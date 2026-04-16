"""PARL v2 composite reward + turn-level credit assignment (K2.5 PARL agent-swarm).

Two modes, both kept in this file so debug_minimal / prod can toggle via
--group-rm:

- **Scalar mode** (no --group-rm): returns the standard composite score per
  sample; miles broadcasts it per-token.
- **Group mode** (--group-rm): additionally computes per-turn rewards,
  group-normalizes them within the prompt, and writes
  ``sample.per_token_advantages`` so miles' GRPO estimator skips scalar
  broadcasting and uses these directly.

Reward formula (aligned with K2.5 PARL paper, arXiv:2602.02276):

  score = r_perf
        + λ₁ · r_parallel
        + λ₂ · r_finish
        + λ_box · r_box

  r_parallel = min(Σ n_assign_t, PARALLEL_CAP) / PARALLEL_CAP   (paper: instantiation reward)
  r_finish   = Σ n_valid_t / Σ n_assign_t                       (paper: subagent finish rate)
  r_box      = 1.0 if response has \\boxed{...} else 0.0        (format / attempt signal)

Per-turn decomposition (group mode) — drives per_token_advantages:
  r_final        = r_perf + λ_box · r_box               (the turn flagged final=True)
  per_turn_r[t]  = λ₁ · (n_assign_t / PARALLEL_CAP)
                 + λ₂ · (n_valid_t / max(1, n_assign_t))  (pre-final turns)
  pure create turn (n_assign_t = 0, final=False) → per_turn_r[t] = 0

Per-turn stats come from ``sample.metadata["turns"]``, populated by generate.py.
Critical steps is logged-only; the episode-length budget is enforced in
generate.py.
"""

import re

import numpy as np

from miles.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
from miles.utils.types import Sample

LAMBDA1_INIT = 0.3  # r_parallel
LAMBDA2_INIT = 0.2  # r_finish
LAMBDA_BOX = 0.1  # r_box (format / attempt signal)
PARALLEL_CAP = 16
ANNEAL_FRAC = 100.0  # intentionally large — annealing is effectively disabled (rewards parallelism)
GRPO_STD_EPS = 1e-6

_BOX_RE = re.compile(r"\\boxed\{")

_step = 0


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

    turns = (sample.metadata or {}).get("turns") or []
    n_assign_total = sum(int(t.get("n_assign", 0)) for t in turns)
    n_valid_total = sum(int(t.get("n_valid", 0)) for t in turns)
    n_create_total = sum(int(t.get("n_create", 0)) for t in turns)

    r_parallel = min(n_assign_total, PARALLEL_CAP) / PARALLEL_CAP
    r_finish = (n_valid_total / n_assign_total) if n_assign_total > 0 else 0.0
    r_box = 1.0 if _has_boxed(response) else 0.0

    score = r_perf + lam1 * r_parallel + lam2 * r_finish + lam_box * r_box

    # Per-turn scalar rewards (pre-normalization). Final turn's credit is
    # r_final (computed downstream); non-final pre-final turns get per_turn_r;
    # non-final pure-create turns get 0.
    per_turn_r = []
    for t in turns:
        if t.get("final", False):
            per_turn_r.append(None)  # placeholder; filled with r_final downstream
            continue
        n_assign_t = int(t.get("n_assign", 0))
        n_valid_t = int(t.get("n_valid", 0))
        if n_assign_t == 0:
            per_turn_r.append(0.0)
        else:
            per_turn_r.append(lam1 * (n_assign_t / PARALLEL_CAP) + lam2 * (n_valid_t / n_assign_t))
    r_final = r_perf + lam_box * r_box

    critical_steps = int((sample.metadata or {}).get("critical_steps", 0))
    registry_size = int((sample.metadata or {}).get("registry_size", 0))

    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["raw_reward"] = r_perf  # keep pass@k clean

    return {
        "score": score,
        "r_perf": r_perf,
        "r_parallel": r_parallel,
        "r_finish": r_finish,
        "r_box": r_box,
        "n_assign": n_assign_total,
        "n_create": n_create_total,
        "n_valid": n_valid_total,
        "critical_steps": critical_steps,
        "registry_size": registry_size,
        "lambda1": lam1,
        "lambda2": lam2,
        "lambda_box": lam_box,
        "pred": perf.get("pred", "") or "",
        # private: consumed by _fill_per_token_advantages, not logged.
        "_per_turn_r": per_turn_r,
        "_r_final": r_final,
    }


def _fill_per_token_advantages(samples: list[Sample], score_dicts: list[dict]) -> None:
    """Populate sample.per_token_advantages with group-normalized per-turn advantages.

    Group = the list passed in (expected to be one prompt's rollouts under
    --group-rm). finals form one pool; non-final per-turn rewards form another.
    """
    finals = np.array([sd["_r_final"] for sd in score_dicts], dtype=np.float64)
    non_final_pool = [r for sd in score_dicts for r in sd["_per_turn_r"] if r is not None]

    final_mean = float(finals.mean())
    final_std = float(finals.std())
    final_centered = finals - final_mean
    if final_std > GRPO_STD_EPS:
        final_advs = final_centered / (final_std + GRPO_STD_EPS)
    else:
        final_advs = final_centered

    if non_final_pool:
        nf_mean = float(np.mean(non_final_pool))
        nf_std = float(np.std(non_final_pool))
    else:
        nf_mean = 0.0
        nf_std = 0.0

    for i, sample in enumerate(samples):
        if sample.loss_mask is None or sample.response_length <= 0:
            continue
        spans = _turn_spans(sample.loss_mask)
        if not spans:
            continue
        per_turn_r = score_dicts[i]["_per_turn_r"]

        advs = [0.0] * sample.response_length  # loss-mask=0 tokens stay 0
        for k, (ss, se) in enumerate(spans):
            if k >= len(per_turn_r):
                # span without a matching turn entry — treat as non-final baseline.
                raw = nf_mean
                adv = (raw - nf_mean) / (nf_std + GRPO_STD_EPS) if nf_std > GRPO_STD_EPS else 0.0
            elif per_turn_r[k] is None:
                # final turn
                adv = float(final_advs[i])
            else:
                raw = per_turn_r[k]
                adv = (raw - nf_mean) / (nf_std + GRPO_STD_EPS) if nf_std > GRPO_STD_EPS else raw - nf_mean
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
