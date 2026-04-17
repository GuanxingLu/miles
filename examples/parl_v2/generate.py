"""PARL v2 custom multi-turn generate (agent-swarm + turn-based critical_steps).

Extends miles' multi_turn.generate with:
- orchestrator system prompt injection
- per-rollout subagent registry (``dict[name, system_prompt]``) closure-bound
  into the tool dispatcher, so ``create_subagent`` / ``assign_task`` share state
- custom parallel tool executor (replacing miles' serial ``execute_tool_calls``):
  ``create_subagent`` calls first (serial, instant), then ``assign_task`` calls
  via ``asyncio.gather`` (capped at ``MAX_CONCURRENT_ASSIGN``)
- K2.5 PARL turn-based ``critical_steps``: each orchestrator turn costs 1, and
  turns with ≥1 executed ``assign_task`` cost 2 (orch + max_i S_sub where S_sub=1
  for single-shot subagents). Running value lives on ``sample.metadata["critical_steps"]``
- structured per-turn stats for reward attribution:
  ``sample.metadata["turns"] = [{n_create, n_assign, n_valid, final}, ...]``
- subagent SGLang router URL discovered via miles.get_model_url("subagent")
  with auto-fallback to the live router when --sglang-config does not
  declare a "subagent" model (= shared/ablation mode)

Design: docs/superpowers/specs/2026-04-17-parl-v2-agent-swarm-alignment-design.md
"""

import argparse
import asyncio
import json
import logging
import uuid
from copy import deepcopy

from openai.types.chat import ChatCompletionMessageToolCall
from sglang.srt.function_call.core_types import ToolCallItem

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    update_sample_from_response,
)
from miles.rollout.generate_utils.tool_call_utils import create_tool_call_parser, update_sample_with_tool_responses
from miles.rollout.sglang_rollout import get_model_url
from miles.utils.http_utils import post
from miles.utils.misc import load_function
from miles.utils.types import Sample

from .prompts import ORCHESTRATOR_SYSTEM_PROMPT
from .tool import _assign_task_call, _create_subagent

logger = logging.getLogger(__name__)
_logged_endpoint = False

MAX_CONCURRENT_ASSIGN = 8


def _with_system_prompt(prompt):
    if isinstance(prompt, str):
        return [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    if isinstance(prompt, list) and (not prompt or prompt[0].get("role") != "system"):
        return [{"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT}] + list(prompt)
    return prompt


def _normalize_tool_call(call) -> tuple[str, dict, str]:
    """Return (name, params, tool_call_id). Mirrors miles' _execute_tool_call."""
    if isinstance(call, ChatCompletionMessageToolCall):
        name = call.function.name
        params = json.loads(call.function.arguments) if call.function.arguments else {}
        tool_call_id = call.id
    elif isinstance(call, ToolCallItem):
        name = call.name
        params = json.loads(call.parameters) if call.parameters else {}
        tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
    else:
        raise TypeError(f"Unsupported tool call type: {type(call)}")
    return name, params, tool_call_id


async def _execute_tool_calls_parallel(
    tool_calls,
    *,
    registry: dict[str, str],
    tokenizer,
    router_url: str,
) -> tuple[list[dict], dict]:
    """Two-phase execution: all ``create_subagent`` serial first, then
    ``assign_task`` in parallel (capped). Returns (tool_messages, per_turn_stats).
    Preserves original tool_call order so response-text ↔ tool_response mapping
    stays intact."""
    normalized = [_normalize_tool_call(c) for c in tool_calls]
    results: list[str | None] = [None] * len(normalized)

    for i, (name, params, _) in enumerate(normalized):
        if name == "create_subagent":
            results[i] = _create_subagent(params, registry=registry)

    assign_indices = [i for i, (n, _, _) in enumerate(normalized) if n == "assign_task"]
    allowed = assign_indices[:MAX_CONCURRENT_ASSIGN]
    denied = assign_indices[MAX_CONCURRENT_ASSIGN:]

    async def run_assign(i: int):
        _, params, _ = normalized[i]
        text, is_valid = await _assign_task_call(params, registry=registry, tokenizer=tokenizer, router_url=router_url)
        return i, text, is_valid

    assign_outputs = await asyncio.gather(*[run_assign(i) for i in allowed])
    n_valid = 0
    for i, text, is_valid in assign_outputs:
        results[i] = text
        if is_valid:
            n_valid += 1

    for i in denied:
        results[i] = (
            f"Error: too many assign_task calls in this turn " f"(cap={MAX_CONCURRENT_ASSIGN}). Retry in a later turn."
        )

    for i, (name, _, _) in enumerate(normalized):
        if results[i] is None:
            results[i] = f"Error: unknown tool '{name}'"

    tool_messages = [
        {"role": "tool", "tool_call_id": tool_call_id, "content": result, "name": name}
        for (name, _, tool_call_id), result in zip(normalized, results)
    ]

    stats = {
        "n_create": sum(1 for n, _, _ in normalized if n == "create_subagent"),
        "n_assign": len(allowed),
        "n_valid": n_valid,
    }
    return tool_messages, stats


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = deepcopy(input.sample)
    sample.prompt = _with_system_prompt(sample.prompt)
    tokenizer = input.state.tokenizer
    subagent_router_url = get_model_url(args, "subagent")
    live_router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    global _logged_endpoint
    if not _logged_endpoint:
        mode = "frozen" if subagent_router_url != live_router_url else "shared (ablation)"
        logger.info(f"[parl_v2] subagent mode: {mode}")
        logger.info(f"[parl_v2] subagent router: {subagent_router_url}")
        logger.info(f"[parl_v2] live router:     {live_router_url}")
        _logged_endpoint = True
    assert not args.partial_rollout, "Partial rollout is not supported"
    assert not args.generate_multi_samples, "generate_multi_samples is not supported in parl_v2 custom multi-turn"

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    tool_specs = load_function(args.generate_tool_specs_path)
    tool_call_parser = create_tool_call_parser(tool_specs, args.generate_tool_call_parser)

    max_cs_raw = getattr(args, "rollout_max_critical_steps", None)
    max_cs = int(max_cs_raw) if max_cs_raw is not None else 2 * int(args.generate_max_turns)

    registry: dict[str, str] = {}
    sample.metadata = dict(sample.metadata or {})
    sample.metadata["critical_steps"] = 0
    sample.metadata["turns"] = []

    prompt_tokens_ids = compute_prompt_ids_from_sample(input.state, sample, tools=tool_specs)
    sample.tokens = prompt_tokens_ids.copy()

    for _turn in range(args.generate_max_turns):
        if sample.metadata["critical_steps"] >= max_cs:
            sample.status = Sample.Status.TRUNCATED
            break

        payload, halt_status = compute_request_payload(args, sample.tokens, input.sampling_params)
        if payload is None:
            sample.status = halt_status
            break

        output = await post(url, payload)
        await update_sample_from_response(args, sample, payload=payload, output=output, update_loss_mask=True)

        if output["meta_info"]["finish_reason"]["type"] in ("abort", "length"):
            sample.metadata["turns"].append({"n_create": 0, "n_assign": 0, "n_valid": 0, "final": False})
            sample.metadata["critical_steps"] += 1
            break

        raw_text = output["text"]
        raw_tool_call_count = raw_text.count(tool_call_parser.detector.bot_token)
        _, tool_calls = tool_call_parser.parse_non_stream(raw_text)
        failed = raw_tool_call_count - len(tool_calls)
        if failed > 0:
            sample.metadata["tool_call_parse_failures"] = sample.metadata.get("tool_call_parse_failures", 0) + failed
            sample.metadata["tool_call_raw_count"] = sample.metadata.get("tool_call_raw_count", 0) + raw_tool_call_count

        if len(tool_calls) == 0:
            sample.metadata["turns"].append({"n_create": 0, "n_assign": 0, "n_valid": 0, "final": False})
            sample.metadata["critical_steps"] += 1
            break

        tool_messages, stats = await _execute_tool_calls_parallel(
            tool_calls,
            registry=registry,
            tokenizer=tokenizer,
            router_url=subagent_router_url,
        )
        sample.metadata["turns"].append({**stats, "final": False})
        sample.metadata["critical_steps"] += 2 if stats["n_assign"] > 0 else 1

        update_sample_with_tool_responses(sample, tool_messages, tokenizer=tokenizer)

    if sample.metadata["turns"]:
        sample.metadata["turns"][-1]["final"] = True
    sample.metadata["registry_size"] = len(registry)

    return GenerateFnOutput(samples=sample)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--generate-max-turns", type=int, default=16)
    parser.add_argument("--generate-tool-specs-path", type=str)
    parser.add_argument("--generate-tool-call-parser", type=str)
    parser.add_argument("--generate-execute-tool-function-path", type=str)
    parser.add_argument("--generate-multi-samples", action="store_true")
    parser.add_argument(
        "--rollout-max-critical-steps",
        type=int,
        default=None,
        help=(
            "K2.5 PARL episode-length budget in TURN units: phase_cost = 1 per "
            "orchestrator turn (final/create-only/length) or 2 per spawn turn "
            "(≥1 executed assign_task). Defaults to 2 * --generate-max-turns."
        ),
    )


generate.add_arguments = _add_arguments
