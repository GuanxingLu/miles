"""PARL v2 composite reward + turn-level credit assignment (K2.5 PARL).

Two modes, both kept in this file so debug_minimal / prod can toggle via
--group-rm:

- **Scalar mode** (no --group-rm): returns the standard composite score per
  sample; miles broadcasts it per-token.
- **Group mode** (--group-rm): additionally computes per-turn rewards,
  group-normalizes them within the prompt, and writes
  `sample.per_token_advantages` so miles' GRPO estimator skips scalar
  broadcasting and uses these directly.

Reward formula (aligned with K2.5 PARL paper, arXiv:2602.02276):

  score = r_perf
        + λ₁ · r_parallel
        + λ₂ · r_finish
        + λ_box · r_box

  r_parallel  = min(Σ num_parallel_i, PARALLEL_CAP) / PARALLEL_CAP   (paper:
                instantiation reward, prevents serial collapse)
  r_finish    = Σ valid_i / Σ total_i                                (paper:
                subagent finish rate)
  r_box       = 1.0 if response has \\boxed{...} else 0.0            (format /
                attempt signal; not from paper, kept as an independent term)

Per-turn decomposition (group mode):
  r_final      = r_perf + λ_box · r_box               (answer + format → final turn)
  per_call_r[i]= λ₁ · (num_parallel_i / PARALLEL_CAP)
               + λ₂ · (valid_i / total_i)             (spawn turn i's local credit)

Critical steps are *not* in the reward. They are an episode-length budget
enforced in generate.py; sample.metadata["critical_steps"] is logged only.

Turn structure is derived from `loss_mask`: contiguous runs of 1s are the
orchestrator's turns. Last run is the final-answer turn; earlier runs are
spawn turns that preceded a `consult_solvers` call (1:1 with per-call stats).
"""

import re

import numpy as np

from miles.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
from miles.utils.types import Sample

from .tool import STATS_FOOTER_PREFIX

LAMBDA1_INIT = 0.3  # r_parallel
LAMBDA2_INIT = 0.2  # r_finish
LAMBDA_BOX = 0.1  # r_box (format / attempt signal)
PARALLEL_CAP = 16
ANNEAL_FRAC = 100.0  # intentionally large — annealing is effectively disabled
GRPO_STD_EPS = 1e-6

# Footer format emitted by tool.py; solver_tokens field is optional (read by
# generate.py for critical_steps, not by reward).
_STATS_FOOTER_RE = re.compile(
    re.escape(STATS_FOOTER_PREFIX) + r"\s*valid=(\d+)\s+total=(\d+)" + r"(?:\s+solver_tokens=[\d,]*)?" + r"\s*-->"
)
_TOOL_CALL_RE = re.compile(r"<tool_call>")
_BOX_RE = re.compile(r"\\boxed\{")

_step = 0


def _read_per_call_stats(response: str) -> list[tuple[int, int]]:
    """Extract (valid, total) for every consult_solvers footer, in order."""
    return [(int(m.group(1)), int(m.group(2))) for m in _STATS_FOOTER_RE.finditer(response or "")]


def _count_tool_calls(response: str) -> int:
    return len(_TOOL_CALL_RE.findall(response or ""))


def _has_boxed(response: str) -> bool:
    return bool(_BOX_RE.search(response or ""))


def _annealed_lambdas(args) -> tuple[float, float, float]:
    global _step
    _step += 1
    total = max(1, int(args.num_rollout) * max(1, int(getattr(args, "rollout_batch_size", 1))))
    frac = _step / max(1, int(ANNEAL_FRAC * total))
    scale = max(0.0, 1.0 - frac)
    return LAMBDA1_INIT * scale, LAMBDA2_INIT * scale, LAMBDA_BOX * scale


def _turn_spans(loss_mask: list[int]) -> list[tuple[int, int]]:
    """Return [start, end) half-open intervals of contiguous loss_mask=1 runs."""
    spans = []
    n = len(loss_mask)
    i = 0
    while i < n:
        if loss_mask[i] == 1:
            j = i
            while j < n and loss_mask[j] == 1:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def _score_one(args, sample: Sample, lam1: float, lam2: float, lam_box: float) -> dict:
    """Compute composite scalar reward + turn decomposition. Pure; no global state."""
    response = sample.response or ""
    label = sample.label if sample.label is not None else ""
    perf = math_dapo_compute_score(response, label, strict_box_verify=True)
    r_perf = 1.0 if float(perf.get("score", 0.0)) > 0 else 0.0

    per_call = _read_per_call_stats(response)
    total_sum = sum(t for _, t in per_call)
    valid_sum = sum(v for v, _ in per_call)
    total_parallel = total_sum  # Σ num_parallel_i equals Σ total_i (each call's "total" == its num_parallel)

    r_parallel = min(total_parallel, PARALLEL_CAP) / PARALLEL_CAP
    r_finish = (valid_sum / total_sum) if total_sum > 0 else 0.0
    r_box = 1.0 if _has_boxed(response) else 0.0
    n_calls = _count_tool_calls(response)

    score = r_perf + lam1 * r_parallel + lam2 * r_finish + lam_box * r_box

    # Per-turn scalar rewards (pre-normalization).
    # Spawn turn i's local credit attributes λ₁·(n_i/cap) + λ₂·(valid_i/total_i)
    # to the decision that produced call i.
    per_call_r = [lam1 * (t / PARALLEL_CAP) + lam2 * (v / t if t > 0 else 0.0) for v, t in per_call]
    r_final = r_perf + lam_box * r_box

    critical_steps = int((sample.metadata or {}).get("critical_steps", 0))

    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["raw_reward"] = r_perf  # keep pass@k clean

    return {
        "score": score,
        "r_perf": r_perf,
        "r_parallel": r_parallel,
        "r_finish": r_finish,
        "r_box": r_box,
        "n_spawn": n_calls,
        "n_solvers_valid": valid_sum,
        "n_solvers_total": total_sum,
        "critical_steps": critical_steps,
        "lambda1": lam1,
        "lambda2": lam2,
        "lambda_box": lam_box,
        "pred": perf.get("pred", "") or "",
        # private: consumed by _fill_per_token_advantages, not logged.
        "_per_call_r": per_call_r,
        "_r_final": r_final,
    }


def _fill_per_token_advantages(samples: list[Sample], score_dicts: list[dict]) -> None:
    """Populate sample.per_token_advantages with group-normalized per-turn advantages.

    Group = the list passed in (expected to be one prompt's rollouts under
    --group-rm). finals form one pool, spawn-call rewards another.
    """
    finals = np.array([sd["_r_final"] for sd in score_dicts], dtype=np.float64)
    spawn_pool = [r for sd in score_dicts for r in sd["_per_call_r"]]

    final_mean = float(finals.mean())
    final_std = float(finals.std())
    final_centered = finals - final_mean
    if final_std > GRPO_STD_EPS:
        final_advs = final_centered / (final_std + GRPO_STD_EPS)
    else:
        final_advs = final_centered

    if spawn_pool:
        spawn_mean = float(np.mean(spawn_pool))
        spawn_std = float(np.std(spawn_pool))
    else:
        spawn_mean = 0.0
        spawn_std = 0.0

    for i, sample in enumerate(samples):
        if sample.loss_mask is None or sample.response_length <= 0:
            continue
        spans = _turn_spans(sample.loss_mask)
        if not spans:
            continue

        advs = [0.0] * sample.response_length  # loss-mask=0 tokens stay 0
        # Last span = final-answer turn.
        fs, fe = spans[-1]
        for t in range(fs, fe):
            advs[t] = float(final_advs[i])

        # All preceding spans = spawn turns; each attributed to one call.
        per_call_r = score_dicts[i]["_per_call_r"]
        for k, (ss, se) in enumerate(spans[:-1]):
            if k < len(per_call_r):
                raw = per_call_r[k]
            else:
                raw = spawn_mean  # truncated without final answer — treat as baseline
            centered = raw - spawn_mean
            adv = centered / (spawn_std + GRPO_STD_EPS) if spawn_std > GRPO_STD_EPS else centered
            for t in range(ss, se):
                advs[t] = float(adv)

        sample.per_token_advantages = advs


async def reward_func(args, sample_or_samples, **kwargs):
    lam1, lam2, lam_box = _annealed_lambdas(args)

    if isinstance(sample_or_samples, list):
        samples = sample_or_samples
        dicts = [_score_one(args, s, lam1, lam2, lam_box) for s in samples]
        _fill_per_token_advantages(samples, dicts)
        return [{k: v for k, v in d.items() if not k.startswith("_")} for d in dicts]

    d = _score_one(args, sample_or_samples, lam1, lam2, lam_box)
    return {k: v for k, v in d.items() if not k.startswith("_")}
