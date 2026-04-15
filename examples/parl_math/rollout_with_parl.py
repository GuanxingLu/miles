import copy

from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

from . import agent_system

PARL_CONFIGS = {
    "num_parallel": 5,
    "lambda1_init": 0.3,  # weight on r_parallel
    "lambda2_init": 0.2,  # weight on r_finish
    "anneal_frac": 0.5,   # anneal to 0 over the first half of training
}

_TOKENIZER = None
_STEP_COUNT = 0  # total training-mode rollout calls observed (approximate).


def _update_lambdas(args):
    """Linear decay of λ₁/λ₂ to 0 over anneal_frac * total_calls."""
    global _STEP_COUNT
    _STEP_COUNT += 1

    total_calls = max(
        1,
        int(args.num_rollout) * int(args.rollout_batch_size) * int(args.n_samples_per_prompt),
    )
    frac = _STEP_COUNT / max(1, int(PARL_CONFIGS["anneal_frac"] * total_calls))
    scale = max(0.0, 1.0 - frac)
    agent_system.PARL_STATE["lambda1"] = PARL_CONFIGS["lambda1_init"] * scale
    agent_system.PARL_STATE["lambda2"] = PARL_CONFIGS["lambda2_init"] * scale


async def generate_with_parl(args, sample: Sample, sampling_params, evaluation=False):
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = load_tokenizer(
            args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
        )

    max_context_length = args.rollout_max_context_len if not evaluation else args.eval_max_context_len

    args = copy.copy(args)
    args.sampling_params = sampling_params
    args.rollout_max_context_len = max_context_length
    args.tokenizer = _TOKENIZER

    for key, value in PARL_CONFIGS.items():
        setattr(args, key, value)

    if not evaluation:
        _update_lambdas(args)
    else:
        # Eval: freeze λ = 0 so r_perf is the only reward reported.
        agent_system.PARL_STATE["lambda1"] = 0.0
        agent_system.PARL_STATE["lambda2"] = 0.0

    samples = await agent_system.run_agent_system(args, sample)
    return samples
