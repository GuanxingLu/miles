"""Dual-tool agent-swarm implementation for PARL v2.

Orchestrator has two tools:
- ``create_subagent(name, system_prompt)``: register a named specialist
  system_prompt in a per-rollout registry. Pure dict write; no inference.
- ``assign_task(agent, prompt)``: look up ``agent`` in the registry, build a
  chat payload with that system_prompt + the given prompt, call SGLang, and
  return the subagent's output text.

Parallelism: the orchestrator is free to emit multiple ``assign_task`` calls
in one turn. ``generate.py``'s custom parallel wrapper runs them via
``asyncio.gather``. Registry state, tokenizer, and the subagent SGLang
router URL are injected as keyword-only args by the wrapper via closure
binding — ``tool.py`` itself is stateless. The router URL points at the
"subagent" model when --sglang-config declares it (frozen mode); otherwise
it falls back to the live router (ablation / shared mode).

See spec: docs/superpowers/specs/2026-04-17-parl-v2-agent-swarm-alignment-design.md
"""

import asyncio

from miles.utils.http_utils import post

MAX_REGISTRY_SIZE = 8
SOLVER_MAX_NEW_TOKENS = 1024
SOLVER_TEMPERATURE = 1.0
SOLVER_CONCURRENCY = 16

_solver_semaphore: asyncio.Semaphore | None = None


tool_specs = [
    {
        "type": "function",
        "function": {
            "name": "create_subagent",
            "description": (
                "Register a specialized subagent with a unique name and a "
                "custom system prompt for later reuse. Returns a short "
                "confirmation. No inference runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique name for this agent configuration.",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "System prompt defining the agent's role, capabilities, and boundaries.",
                    },
                },
                "required": ["name", "system_prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assign_task",
            "description": (
                "Launch a previously-created subagent on a single task and "
                "return its candidate solution. You can emit multiple "
                "assign_task calls in the same turn; they run concurrently."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "description": "Which created agent to use.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The task for the agent to perform.",
                    },
                },
                "required": ["agent", "prompt"],
            },
        },
    },
]



def _get_semaphore() -> asyncio.Semaphore:
    global _solver_semaphore
    if _solver_semaphore is None:
        _solver_semaphore = asyncio.Semaphore(SOLVER_CONCURRENCY)
    return _solver_semaphore


def _create_subagent(params: dict, *, registry: dict[str, str]) -> str:
    name = params.get("name")
    system_prompt = params.get("system_prompt")
    if not isinstance(name, str) or not name.strip():
        return "Error: 'name' must be a non-empty string."
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        return "Error: 'system_prompt' must be a non-empty string."
    if name not in registry and len(registry) >= MAX_REGISTRY_SIZE:
        return (
            f"Error: subagent registry full (max {MAX_REGISTRY_SIZE}). "
            "Reuse an existing agent or avoid creating more."
        )
    registry[name] = system_prompt
    return f"Registered subagent '{name}'."


async def _assign_task_call(
    params: dict, *, registry: dict[str, str], tokenizer, router_url: str
) -> tuple[str, bool]:
    """Return (text, is_valid). ``is_valid`` marks whether the subagent
    produced a non-empty, non-error response (for r_finish)."""
    agent = params.get("agent")
    prompt = params.get("prompt")
    if not isinstance(agent, str) or not agent.strip():
        return "Error: 'agent' must be a non-empty string.", False
    if not isinstance(prompt, str) or not prompt.strip():
        return "Error: 'prompt' must be a non-empty string.", False
    if agent not in registry:
        return f"Error: agent '{agent}' not found. Call create_subagent first.", False

    messages = [
        {"role": "system", "content": registry[agent]},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    payload = {
        "text": text,
        "sampling_params": {
            "max_new_tokens": SOLVER_MAX_NEW_TOKENS,
            "temperature": SOLVER_TEMPERATURE,
            "top_p": 1.0,
        },
    }
    async with _get_semaphore():
        try:
            output = await post(router_url, payload)
        except Exception as e:
            return f"__SOLVER_ERROR__: {e}", False
    body = output.get("text", "") or ""
    is_valid = bool(body.strip()) and not body.startswith("__SOLVER_ERROR__")
    return body, is_valid
