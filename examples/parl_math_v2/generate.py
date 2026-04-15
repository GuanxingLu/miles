"""Thin wrapper around miles' multi_turn.generate that injects a system
prompt teaching the orchestrator about the `consult_solvers` tool.

We skip `--apply-chat-template` at the data layer so sample.prompt arrives
as the raw problem string; here we wrap it into a [system, user] message
list and delegate. multi_turn.generate's compute_prompt_ids_from_sample
handles list prompts via tokenizer.apply_chat_template.
"""
from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.multi_turn import _add_arguments
from miles.rollout.generate_hub.multi_turn import generate as _multi_turn_generate

from .prompts import ORCHESTRATOR_SYSTEM_PROMPT


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    new_sample = deepcopy(input.sample)
    prompt = new_sample.prompt
    if isinstance(prompt, str):
        new_sample.prompt = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    # If already a list of messages, prepend system only if missing.
    elif isinstance(prompt, list) and (not prompt or prompt[0].get("role") != "system"):
        new_sample.prompt = [{"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT}] + list(prompt)

    wrapped_input = GenerateFnInput(
        state=input.state,
        sample=new_sample,
        sampling_params=input.sampling_params,
        evaluation=input.evaluation,
    )
    return await _multi_turn_generate(wrapped_input)


generate.add_arguments = _add_arguments
