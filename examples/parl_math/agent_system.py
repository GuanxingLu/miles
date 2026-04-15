import asyncio
import copy
import time
import traceback
from copy import deepcopy

from miles.rollout.rm_hub import batched_async_rm
from miles.utils.http_utils import post
from miles.utils.types import Sample

from .prompts import SOLVER_PROMPT_TEMPLATE, generate_orchestrator_template


# Populated by rollout_with_parl at every step. Read-only inside agent_system.
PARL_STATE = {
    "lambda1": 0.0,  # weight on r_parallel
    "lambda2": 0.0,  # weight on r_finish
}


def _unwrap_score(reward) -> float:
    if reward is None:
        return 0.0
    if isinstance(reward, dict):
        return float(reward.get("score", 0.0))
    return float(reward)


def _is_valid_solver_sample(sample: Sample) -> bool:
    if sample is None:
        return False
    if sample.status == Sample.Status.ABORTED or sample.status == Sample.Status.FAILED:
        return False
    if not sample.response or not sample.response.strip():
        return False
    return True


def _has_final_answer(text: str) -> bool:
    return isinstance(text, str) and "\\boxed{" in text


async def generate_response(args, prompt, role) -> Sample:
    """Call SGLang /generate, build a Sample, tag loss_mask by role.

    role in {"solver", "orchestrator"}. Solver tokens get loss_mask=0 (frozen
    subagent). Orchestrator tokens get loss_mask=1 (trainable).
    """
    sampling_params = args.sampling_params
    tokenizer = args.tokenizer
    max_context_length = args.rollout_max_context_len
    sample: Sample = deepcopy(args.sample)

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    sample.prompt = prompt
    prompt_token_ids = tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
    sample.tokens = prompt_token_ids
    sample.response_length = 0
    prompt_length = len(prompt_token_ids)

    cur_params = deepcopy(sampling_params)
    cur_params["max_new_tokens"] = min(
        sampling_params["max_new_tokens"], max_context_length - prompt_length
    )
    if cur_params["max_new_tokens"] <= 0:
        sample.status = Sample.Status.ABORTED
        sample.loss_mask = []
        return sample

    payload = {"input_ids": prompt_token_ids, "sampling_params": cur_params, "return_logprob": True}
    output = await post(url, payload)

    if "output_token_logprobs" in output["meta_info"]:
        new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_tokens = []

    sample.tokens = sample.tokens + new_tokens
    sample.response_length = len(new_tokens)
    sample.response = output["text"]

    finish = output["meta_info"]["finish_reason"]["type"]
    if finish == "length":
        sample.status = Sample.Status.TRUNCATED
    elif finish == "stop":
        sample.status = Sample.Status.COMPLETED
    else:
        sample.status = Sample.Status.ABORTED

    mask_val = 0 if role == "solver" else 1
    sample.loss_mask = [mask_val] * sample.response_length
    sample.metadata = {**(sample.metadata or {}), "parl_role": role}
    return sample


async def _solver_worker(args, problem_statement):
    try:
        prompt = SOLVER_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
        return await generate_response(args, prompt, role="solver")
    except Exception as e:
        print(f"[solver] exception: {e}")
        print(traceback.format_exc())
        return None


async def _orchestrator_call(args, problem_statement, candidate_texts):
    template = generate_orchestrator_template(len(candidate_texts))
    fmt = {"problem_statement": problem_statement}
    for i, text in enumerate(candidate_texts):
        fmt[f"solution{i+1}"] = text
    prompt = template.format(**fmt)
    for _ in range(3):
        try:
            return await generate_response(args, prompt, role="orchestrator")
        except Exception as e:
            print(f"[orchestrator] retry due to: {e}")
            time.sleep(1)
    return None


async def run_agent_system(args, sample: Sample):
    """PARL 2-stage rollout: N frozen solvers → 1 trainable orchestrator.

    Returns: list containing a single orchestrator Sample (with composite reward).
    Solver samples are dropped from the returned list so GRPO grouping is clean.
    Their per-sample reward is still computed for r_parallel bookkeeping.
    """
    # Shallow copy so we don't mutate caller args; don't deepcopy (tokenizer!).
    args = copy.copy(args)
    args.sample = sample
    problem_statement = sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt)

    # Stage 1: parallel solvers (frozen / loss_mask=0).
    solver_tasks = [_solver_worker(args, problem_statement) for _ in range(args.num_parallel)]
    solver_results = await asyncio.gather(*solver_tasks, return_exceptions=True)
    solver_samples = [s for s in solver_results if isinstance(s, Sample)]
    valid_solvers = [s for s in solver_samples if _is_valid_solver_sample(s)]

    # Log-only: solver r_perf via RM. Not used for gradient (loss_mask=0).
    if valid_solvers:
        try:
            solver_rewards = await batched_async_rm(args, valid_solvers)
            for s, r in zip(valid_solvers, solver_rewards, strict=False):
                s.reward = r
        except Exception as e:
            print(f"[parl] solver RM failed: {e}")

    # r_parallel ∈ [0, 1]: fraction of solvers returning usable output.
    r_parallel = len(valid_solvers) / max(1, args.num_parallel)

    # Fallback when all solvers failed: run orchestrator on the raw problem alone.
    candidate_texts = [s.response for s in valid_solvers] if valid_solvers else [problem_statement]

    # Stage 2: orchestrator aggregation (trainable / loss_mask=1).
    orch_sample = await _orchestrator_call(args, problem_statement, candidate_texts)

    if orch_sample is None or orch_sample.response_length == 0:
        # Synthesize an aborted orchestrator sample so training pipeline still gets something.
        orch_sample = deepcopy(sample)
        orch_sample.response = ""
        orch_sample.response_length = 0
        orch_sample.loss_mask = []
        orch_sample.status = Sample.Status.ABORTED
        orch_sample.metadata = {**(orch_sample.metadata or {}), "parl_role": "orchestrator"}
        r_perf = 0.0
        r_finish = 0.0
    else:
        try:
            rm_out = await batched_async_rm(args, [orch_sample])
            orch_sample.reward = rm_out[0]
        except Exception as e:
            print(f"[parl] orchestrator RM failed: {e}")
            orch_sample.reward = 0.0
        r_perf = _unwrap_score(orch_sample.reward)
        r_finish = 1.0 if _has_final_answer(orch_sample.response) else 0.0

    lam1 = float(PARL_STATE.get("lambda1", 0.0))
    lam2 = float(PARL_STATE.get("lambda2", 0.0))
    total = r_perf + lam1 * r_parallel + lam2 * r_finish

    # Preserve dict shape if the RM returned a dict (miles unwraps via args.reward_key).
    if isinstance(orch_sample.reward, dict):
        orch_sample.reward = {
            **orch_sample.reward,
            "score": total,
            "r_perf": r_perf,
            "r_parallel": r_parallel,
            "r_finish": r_finish,
            "lambda1": lam1,
            "lambda2": lam2,
        }
    else:
        orch_sample.reward = {
            "score": total,
            "r_perf": r_perf,
            "r_parallel": r_parallel,
            "r_finish": r_finish,
            "lambda1": lam1,
            "lambda2": lam2,
        }

    # Stash unweighted r_perf (0/1 from math grader) so rollout.py overrides
    # train_data["raw_reward"] with it — otherwise pass@k compares the composite
    # score against 1 and degenerates to ~0 whenever λ1/λ2 > 0.
    if orch_sample.metadata is None:
        orch_sample.metadata = {}
    orch_sample.metadata["raw_reward"] = r_perf

    return [orch_sample]
