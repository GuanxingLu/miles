"""Spawn-as-tool implementation for PARL v2.

Exposes `tool_specs` and `execute_tool` for miles' multi-turn generate loop.
Each `consult_solvers` call fans out N independent SGLang /generate requests
(sharing the orchestrator's engine) and returns their outputs as a single
formatted string. Solver outputs are injected back into the orchestrator's
context via miles' `update_sample_with_tool_responses`, so they get
`loss_mask=0` automatically.

Solver success stats are embedded as a machine-readable footer in the tool
response so `reward.py::reward_func` can recover `valid/total` for the
`r_parallel` term without needing any sample-side plumbing.
"""

import asyncio
import os

from miles.utils.http_utils import post

from .prompts import SOLVER_PROMPT_TEMPLATE

MAX_PARALLEL = 8
MIN_PARALLEL = 1
SOLVER_MAX_NEW_TOKENS = 1024
SOLVER_TEMPERATURE = 1.0
SOLVER_CONCURRENCY = 16
STATS_FOOTER_PREFIX = "<!-- consult_solvers_stats:"
STATS_FOOTER_SUFFIX = "-->"

_solver_semaphore: asyncio.Semaphore | None = None


tool_specs = [
    {
        "type": "function",
        "function": {
            "name": "consult_solvers",
            "description": (
                "Spawn N independent math-solver agents in parallel to "
                "produce candidate solutions for the given problem. Returns "
                "the N candidate solutions concatenated as text. Use when "
                "diverse independent attempts would help you arrive at the "
                "correct answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "problem": {
                        "type": "string",
                        "description": "The problem statement to pass to each solver.",
                    },
                    "num_parallel": {
                        "type": "integer",
                        "description": f"Number of parallel solvers (1 to {MAX_PARALLEL}).",
                        "minimum": MIN_PARALLEL,
                        "maximum": MAX_PARALLEL,
                    },
                },
                "required": ["problem", "num_parallel"],
            },
        },
    }
]


def _router_url() -> str:
    ip = os.environ.get("MILES_SGLANG_ROUTER_IP") or os.environ.get("SGLANG_ROUTER_IP")
    port = os.environ.get("MILES_SGLANG_ROUTER_PORT") or os.environ.get("SGLANG_ROUTER_PORT")
    if not ip or not port:
        raise RuntimeError(
            "consult_solvers needs MILES_SGLANG_ROUTER_{IP,PORT} in env. "
            "Launcher must export these to match --sglang-router-ip/--sglang-router-port."
        )
    return f"http://{ip}:{port}/generate"


def _get_semaphore() -> asyncio.Semaphore:
    global _solver_semaphore
    if _solver_semaphore is None:
        _solver_semaphore = asyncio.Semaphore(SOLVER_CONCURRENCY)
    return _solver_semaphore


async def _solver_call(problem: str) -> tuple[str, int]:
    """Call a single solver and return (text, completion_tokens)."""
    prompt = SOLVER_PROMPT_TEMPLATE.format(problem=problem)
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": SOLVER_MAX_NEW_TOKENS,
            "temperature": SOLVER_TEMPERATURE,
            "top_p": 1.0,
        },
    }
    async with _get_semaphore():
        try:
            output = await post(_router_url(), payload)
        except Exception as e:
            return f"__SOLVER_ERROR__: {e}", 0
    text = output.get("text", "") or ""
    n_tokens = int(output.get("meta_info", {}).get("completion_tokens", 0))
    return text, n_tokens


def _is_valid_solver_output(text: str) -> bool:
    if not isinstance(text, str):
        return False
    if text.startswith("__SOLVER_ERROR__"):
        return False
    return bool(text.strip())


def _format_candidates(candidates: list[tuple[str, int]]) -> str:
    texts = [c[0] for c in candidates]
    token_counts = [c[1] for c in candidates]
    valid = sum(1 for t in texts if _is_valid_solver_output(t))
    total = len(candidates)
    sections = []
    for i, text in enumerate(texts, 1):
        body = text.strip() if _is_valid_solver_output(text) else "(solver returned no usable output)"
        sections.append(f"### Candidate Solution {i}\n{body}")
    body = "\n\n".join(sections)
    tokens_str = ",".join(str(t) for t in token_counts)
    footer = f"{STATS_FOOTER_PREFIX} valid={valid} total={total} solver_tokens={tokens_str} {STATS_FOOTER_SUFFIX}"
    return f"{body}\n\n{footer}"


async def execute_tool(name: str, params: dict) -> str:
    if name != "consult_solvers":
        return f"Error: unknown tool '{name}'"
    problem = params.get("problem", "")
    if not isinstance(problem, str) or not problem.strip():
        return "Error: 'problem' must be a non-empty string."
    try:
        n = int(params.get("num_parallel", 3))
    except (TypeError, ValueError):
        n = 3
    n = max(MIN_PARALLEL, min(MAX_PARALLEL, n))
    candidates = await asyncio.gather(*[_solver_call(problem) for _ in range(n)])
    return _format_candidates(candidates)
