"""PARL v2 composite reward + turn-level credit assignment (K2.5 PARL).

Two modes, both kept in this file so debug_minimal / prod can toggle via
--group-rm:

- **Scalar mode** (no --group-rm): returns the standard composite score per
  sample; miles broadcasts it per-token. Legacy behavior, kept for eval.
- **Group mode** (--group-rm, reward_func receives the full rollout group):
  additionally computes per-turn rewards, group-normalizes them within the
  prompt, and writes `sample.per_token_advantages` so miles' GRPO estimator
  skips scalar broadcasting and uses these directly.

Turn structure is derived from `loss_mask`: contiguous runs of 1s are the
orchestrator's turns (runs of 0s are frozen solver/tool observations). The
last run is the "final answer" turn; earlier runs are "spawn" turns that
preceded a `consult_solvers` call.

Per-turn reward decomposition:
  r_final = r_perf + λ₂ · r_finish            (credit for answer quality)
  r_spawn = Σ_calls (λ₁·r_parallel_call - cost) / n_spawn_turns
                                              (credit spread over decision turns)

After group-normalization (mean-subtract, optional std-divide per turn
position), we materialize per_token_advantages by broadcasting the
normalized per-turn advantage to every token in that turn's loss-mask=1 span.
"""
import re

import numpy as np

from miles.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
from miles.utils.types import Sample

from .tool import STATS_FOOTER_PREFIX

LAMBDA1_INIT = 0.3
LAMBDA2_INIT = 0.2
LAMBDA3_INIT = 0.1  # critical steps penalty weight
ANNEAL_FRAC = 100.0
COST_PER_CALL = 0.02
GRPO_STD_EPS = 1e-6

# Matches both old format (no solver_tokens) and new format (with solver_tokens).
_STATS_FOOTER_RE = re.compile(
    re.escape(STATS_FOOTER_PREFIX)
    + r"\s*valid=(\d+)\s+total=(\d+)"
    + r"(?:\s+solver_tokens=([\d,]+))?"
    + r"\s*-->"
)
_TOOL_CALL_RE = re.compile(r"<tool_call>")
_BOX_RE = re.compile(r"\\boxed\{")

_step = 0


def _read_per_call_stats(response: str) -> list[dict]:
    """Extract stats for every consult_solvers footer, in order.

    Returns list of dicts with keys: valid, total, solver_tokens (list[int] or None).
    """
    results = []
    for m in _STATS_FOOTER_RE.finditer(response or ""):
        tokens_str = m.group(3)
        solver_tokens = [int(x) for x in tokens_str.split(",")] if tokens_str else None
        results.append({
            "valid": int(m.group(1)),
            "total": int(m.group(2)),
            "solver_tokens": solver_tokens,
        })
    return results


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
    return LAMBDA1_INIT * scale, LAMBDA2_INIT * scale, LAMBDA3_INIT * scale


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


def _compute_critical_steps(per_call: list[dict], loss_mask: list[int] | None) -> int:
    """Compute K2.5-style critical steps: Σ_stage(S_main + max_i(S_sub_i)).

    Each consult_solvers call is a "stage". The critical path for that stage is
    the orchestrator turn tokens (from loss_mask) + max solver token count
    (from footer). If solver_tokens are unavailable, only orchestrator tokens
    are counted (conservative: underestimates critical path).
    """
    if not per_call:
        # No tool calls — critical steps = total orchestrator tokens.
        if loss_mask:
            return sum(1 for v in loss_mask if v == 1)
        return 0

    # Orchestrator turn lengths from loss_mask spans.
    orch_turn_lens = []
    if loss_mask:
        spans = _turn_spans(loss_mask)
        orch_turn_lens = [end - start for start, end in spans]

    critical = 0
    for i, call in enumerate(per_call):
        # Orchestrator turn that preceded this call (spawn turn i).
        orch_tokens = orch_turn_lens[i] if i < len(orch_turn_lens) else 0
        # Parallel solvers: critical path = max solver tokens.
        solver_tokens = call.get("solver_tokens")
        max_solver = max(solver_tokens) if solver_tokens else 0
        critical += orch_tokens + max_solver

    # Add the final-answer turn (last orchestrator turn, after all calls).
    if orch_turn_lens and len(orch_turn_lens) > len(per_call):
        critical += orch_turn_lens[-1]

    return critical


def _score_one(args, sample: Sample, lam1: float, lam2: float, lam3: float) -> dict:
    """Compute composite scalar reward + turn decomposition. Pure; no global state."""
    response = sample.response or ""
    label = sample.label if sample.label is not None else ""
    perf = math_dapo_compute_score(response, label, strict_box_verify=True)
    r_perf = 1.0 if float(perf.get("score", 0.0)) > 0 else 0.0

    per_call = _read_per_call_stats(response)
    valid = sum(c["valid"] for c in per_call)
    total = sum(c["total"] for c in per_call)
    r_parallel = (valid / total) if total > 0 else 0.0
    r_finish = 1.0 if _has_boxed(response) else 0.0
    n_calls = _count_tool_calls(response)

    # Critical steps: normalized by max_response_len to get [0, 1] range.
    critical_steps = _compute_critical_steps(per_call, sample.loss_mask)
    max_resp_len = max(1, int(getattr(args, "rollout_max_response_len", 8192)))
    critical_steps_ratio = critical_steps / max_resp_len

    score = (
        r_perf
        + lam1 * r_parallel
        + lam2 * r_finish
        - COST_PER_CALL * n_calls
        - lam3 * critical_steps_ratio
    )

    # Per-turn scalar rewards (pre-normalization). Split spawn credit into a
    # per-call contribution so reward.parallel attributes correctly to the
    # turn that spawned it (loss_mask turn order == tool_call order).
    per_call_r = [
        lam1 * (c["valid"] / c["total"] if c["total"] > 0 else 0.0) - COST_PER_CALL
        for c in per_call
    ]
    r_final = r_perf + lam2 * r_finish

    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["raw_reward"] = r_perf  # keep pass@k clean

    return {
        "score": score,
        "r_perf": r_perf,
        "r_parallel": r_parallel,
        "r_finish": r_finish,
        "n_spawn": n_calls,
        "n_solvers_valid": valid,
        "n_solvers_total": total,
        "critical_steps": critical_steps,
        "critical_steps_ratio": critical_steps_ratio,
        "lambda1": lam1,
        "lambda2": lam2,
        "lambda3": lam3,
        "pred": perf.get("pred", "") or "",
        # private: consumed by _fill_per_token_advantages, not logged.
        "_per_call_r": per_call_r,
        "_r_final": r_final,
    }


def _fill_per_token_advantages(samples: list[Sample], score_dicts: list[dict]) -> None:
    """Populate sample.per_token_advantages with group-normalized per-turn advantages.

    Group = the list passed in (expected to be one prompt's rollouts under
    --group-rm). Normalization is per turn *position*: final turn is its own
    slot; spawn turns share a slot (they're conceptually "decide to spawn").
    """
    # --- collect per-sample per-turn reward vectors -------------------------
    n = len(samples)
    finals = np.array([sd["_r_final"] for sd in score_dicts], dtype=np.float64)

    # Pool all spawn-call rewards across the group to form the spawn baseline.
    # Using a single pooled baseline avoids misaligning "call #2 vs call #3"
    # across rollouts that made different numbers of calls.
    spawn_pool = [r for sd in score_dicts for r in sd["_per_call_r"]]

    final_mean = float(finals.mean())
    final_std = float(finals.std())
    final_centered = finals - final_mean
    if final_std > GRPO_STD_EPS:
        final_advs = final_centered / (final_std + GRPO_STD_EPS)
    else:
        final_advs = final_centered  # degenerate group → no signal

    if spawn_pool:
        spawn_mean = float(np.mean(spawn_pool))
        spawn_std = float(np.std(spawn_pool))
    else:
        spawn_mean = 0.0
        spawn_std = 0.0

    # --- materialize per-token advantages -----------------------------------
    for i, sample in enumerate(samples):
        if sample.loss_mask is None or sample.response_length <= 0:
            continue
        spans = _turn_spans(sample.loss_mask)
        if not spans:
            continue

        advs = [0.0] * sample.response_length  # loss-mask=0 tokens stay 0
        # Last span is the final-answer turn.
        fs, fe = spans[-1]
        for t in range(fs, fe):
            advs[t] = float(final_advs[i])

        # All preceding spans are spawn turns; each attributed to one call.
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
    lam1, lam2, lam3 = _annealed_lambdas(args)

    if isinstance(sample_or_samples, list):
        samples = sample_or_samples
        dicts = [_score_one(args, s, lam1, lam2, lam3) for s in samples]
        _fill_per_token_advantages(samples, dicts)
        # Strip private keys before returning — miles writes this dict to sample.reward.
        return [{k: v for k, v in d.items() if not k.startswith("_")} for d in dicts]

    d = _score_one(args, sample_or_samples, lam1, lam2, lam3)
    return {k: v for k, v in d.items() if not k.startswith("_")}
