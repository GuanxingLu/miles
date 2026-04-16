"""PARL v2 custom multi-turn generate with critical_steps budget.

Extends miles' multi_turn.generate with:
- orchestrator system prompt injection (original role of this module)
- K2.5 PARL *critical_steps* tracking: accumulates per-stage
  ``orch_turn_tokens + max_i(solver_tokens_i)``; exposes the running value
  on ``sample.metadata["critical_steps"]`` for logging.
- Dual-budget termination:
  * ``critical_steps < rollout_max_critical_steps`` — paper's episode-length budget
  * ``response_length < rollout_max_response_len`` — context-window cap
    (enforced by miles' existing ``compute_request_payload`` via
    ``rollout_max_context_len``; no change needed here)

The latter keeps context from exploding when orchestrator spawns many small
solvers (critical_steps only counts the longest branch).

Design: docs/superpowers/specs/2026-04-17-parl-v2-critical-steps-refactor-design.md
"""

import argparse
import re
from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    update_sample_from_response,
)
from miles.rollout.generate_utils.tool_call_utils import (
    create_tool_call_parser,
    execute_tool_calls,
    update_sample_with_tool_responses,
)
from miles.utils.http_utils import post
from miles.utils.misc import load_function
from miles.utils.types import Sample

from .prompts import ORCHESTRATOR_SYSTEM_PROMPT
from .tool import STATS_FOOTER_PREFIX

_SOLVER_TOKENS_RE = re.compile(re.escape(STATS_FOOTER_PREFIX) + r"\s*valid=\d+\s+total=\d+\s+solver_tokens=([\d,]+)")


def _max_solver_tokens_from(tool_messages: list[dict]) -> int:
    """Return max solver completion-token count across all consult_solvers
    responses in ``tool_messages``. Returns 0 if no footer is present."""
    all_tokens: list[int] = []
    for msg in tool_messages:
        if msg.get("name") != "consult_solvers":
            continue
        m = _SOLVER_TOKENS_RE.search(msg.get("content") or "")
        if not m:
            continue
        try:
            all_tokens.extend(int(x) for x in m.group(1).split(",") if x)
        except ValueError:
            continue
    return max(all_tokens) if all_tokens else 0


def _with_system_prompt(prompt):
    if isinstance(prompt, str):
        return [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    if isinstance(prompt, list) and (not prompt or prompt[0].get("role") != "system"):
        return [{"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT}] + list(prompt)
    return prompt


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    # ----------------------- Setup -------------------------
    args = input.args
    sample = deepcopy(input.sample)
    sample.prompt = _with_system_prompt(sample.prompt)
    tokenizer = input.state.tokenizer
    assert not args.partial_rollout, "Partial rollout is not supported"
    assert not args.generate_multi_samples, "generate_multi_samples is not supported in parl_math_v2 custom multi-turn"

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    execute_tool_function = load_function(args.generate_execute_tool_function_path)
    tool_specs = load_function(args.generate_tool_specs_path)
    tool_call_parser = create_tool_call_parser(tool_specs, args.generate_tool_call_parser)

    max_cs_raw = getattr(args, "rollout_max_critical_steps", None)
    max_cs = int(max_cs_raw) if max_cs_raw is not None else int(args.rollout_max_response_len)

    sample.metadata = dict(sample.metadata or {})
    sample.metadata["critical_steps"] = 0

    # ----------------------- Initial prompts -------------------------
    prompt_tokens_ids = compute_prompt_ids_from_sample(input.state, sample, tools=tool_specs)
    sample.tokens = prompt_tokens_ids.copy()

    for _turn in range(args.generate_max_turns):
        # K2.5 PARL: stop once critical_steps budget is spent.
        if sample.metadata["critical_steps"] >= max_cs:
            sample.status = Sample.Status.TRUNCATED
            break

        # ----------------------- Call inference endpoint -------------------------
        payload, halt_status = compute_request_payload(args, sample.tokens, input.sampling_params)
        if payload is None:
            sample.status = halt_status
            break

        resp_len_before = sample.response_length
        output = await post(url, payload)
        await update_sample_from_response(args, sample, payload=payload, output=output, update_loss_mask=True)
        orch_new_tokens = sample.response_length - resp_len_before

        if output["meta_info"]["finish_reason"]["type"] in ("abort", "length"):
            # Orchestrator hit its own length cap; account for the tokens and stop.
            sample.metadata["critical_steps"] += orch_new_tokens
            break

        # ----------------------- Execute tools -------------------------
        _, tool_calls = tool_call_parser.parse_non_stream(output["text"])
        if len(tool_calls) == 0:
            # Final-answer turn: orchestrator time only, no solver fan-out.
            sample.metadata["critical_steps"] += orch_new_tokens
            break

        tool_messages = await execute_tool_calls(tool_calls, execute_tool_function)
        max_solver = _max_solver_tokens_from(tool_messages)
        sample.metadata["critical_steps"] += orch_new_tokens + max_solver

        update_sample_with_tool_responses(sample, tool_messages, tokenizer=tokenizer)

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
            "K2.5 PARL episode-length budget: max cumulative critical steps "
            "(orch_turn_tokens + max_i solver_tokens_i, per stage). "
            "Defaults to --rollout-max-response-len."
        ),
    )


generate.add_arguments = _add_arguments
