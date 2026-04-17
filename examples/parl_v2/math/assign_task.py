"""Math assign_task impl: single-turn SGLang call on the subagent engine.

Matches the PARL v2 K2.5 single-shot solver pattern: orchestrator calls
assign_task → subagent gets system_prompt + prompt → one sampling pass →
<result>…</result> extraction → orchestrator sees only that block.

Paired via --assign-task-impl-path (or the default resolution in
run_parl_v2.py env='math'); loaded by examples.parl_v2.generate.
"""

import asyncio

from miles.utils.http_utils import post

from ..tool import _RESULT_RE, SUBAGENT_OUTPUT_SUFFIX, extract_subagent_result

SOLVER_MAX_NEW_TOKENS = 1024
SOLVER_TEMPERATURE = 1.0
SOLVER_CONCURRENCY = 16

_solver_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _solver_semaphore
    if _solver_semaphore is None:
        _solver_semaphore = asyncio.Semaphore(SOLVER_CONCURRENCY)
    return _solver_semaphore


async def call(params: dict, *, registry: dict[str, str], tokenizer, router_url: str) -> tuple[str, bool, int]:
    """Return (text, is_valid, sub_steps). ``is_valid`` marks whether the
    subagent produced a non-empty, non-error response with a <result> block.
    ``sub_steps`` is the number of SGLang generation calls the subagent
    made (always 1 in single-turn math mode; widesearch returns the real
    ReAct turn count)."""
    agent = params.get("agent")
    prompt = params.get("prompt")
    if not isinstance(agent, str) or not agent.strip():
        return "Error: 'agent' must be a non-empty string.", False, 0
    if not isinstance(prompt, str) or not prompt.strip():
        return "Error: 'prompt' must be a non-empty string.", False, 0
    if agent not in registry:
        return f"Error: agent '{agent}' not found. Call create_subagent first.", False, 0

    messages = [
        {"role": "system", "content": registry[agent] + SUBAGENT_OUTPUT_SUFFIX},
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
            return f"__SOLVER_ERROR__: {e}", False, 1
    body = output.get("text", "") or ""
    is_valid = bool(body.strip()) and not body.startswith("__SOLVER_ERROR__") and bool(_RESULT_RE.search(body))
    if is_valid:
        body = extract_subagent_result(body)
    return body, is_valid, 1
