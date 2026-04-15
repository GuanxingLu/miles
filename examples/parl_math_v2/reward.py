"""PARL v2 composite reward: r_perf + λ₁·r_parallel + λ₂·r_finish - cost·n_spawn.

- r_perf: dapo math grader on the orchestrator's final answer.
- r_parallel: aggregated fraction of valid solver outputs across all
  consult_solvers calls in this rollout, parsed from stats footers that
  `tool.py::_format_candidates` embeds in each tool response.
- r_finish: 1 if the orchestrator emitted \\boxed{...} anywhere.
- cost: small negative term per tool call, protects against infinite spawn
  under λ₁ > 0.

λ₁/λ₂ linearly anneal to 0 over the first `ANNEAL_FRAC * num_rollout` steps.
Step is approximated with a module-level counter (no rollout_id is passed to
the reward fn), which drifts from true training step but preserves the
monotone-decay shape — good enough for MVR.
"""
import re

from miles.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
from miles.utils.types import Sample

from .tool import STATS_FOOTER_PREFIX

LAMBDA1_INIT = 0.3   # weight on r_parallel
LAMBDA2_INIT = 0.2   # weight on r_finish
ANNEAL_FRAC = 0.5    # fraction of num_rollout over which λ decays to 0
COST_PER_CALL = 0.02 # per-tool-call penalty (r_perf ∈ {0,1} dominates this)

_STATS_FOOTER_RE = re.compile(
    re.escape(STATS_FOOTER_PREFIX) + r"\s*valid=(\d+)\s+total=(\d+)\s*-->"
)
_TOOL_CALL_RE = re.compile(r"<tool_call>")
_BOX_RE = re.compile(r"\\boxed\{")

_step = 0


def _read_solver_stats(response: str) -> tuple[int, int]:
    """Sum (valid, total) over all stats footers in the response."""
    valid = 0
    total = 0
    for m in _STATS_FOOTER_RE.finditer(response or ""):
        valid += int(m.group(1))
        total += int(m.group(2))
    return valid, total


def _count_tool_calls(response: str) -> int:
    return len(_TOOL_CALL_RE.findall(response or ""))


def _has_boxed(response: str) -> bool:
    return bool(_BOX_RE.search(response or ""))


def _annealed_lambdas(args) -> tuple[float, float]:
    global _step
    _step += 1
    total = max(1, int(args.num_rollout) * max(1, int(getattr(args, "rollout_batch_size", 1))))
    frac = _step / max(1, int(ANNEAL_FRAC * total))
    scale = max(0.0, 1.0 - frac)
    return LAMBDA1_INIT * scale, LAMBDA2_INIT * scale


def _score_one(args, sample: Sample) -> dict:
    response = sample.response or ""
    label = sample.label if sample.label is not None else ""
    perf = math_dapo_compute_score(response, label, strict_box_verify=True)
    r_perf = float(perf.get("score", 0.0))
    r_perf = 1.0 if r_perf > 0 else 0.0  # clamp dapo's possible negatives to {0,1}

    valid, total = _read_solver_stats(response)
    r_parallel = (valid / total) if total > 0 else 0.0
    r_finish = 1.0 if _has_boxed(response) else 0.0
    n_calls = _count_tool_calls(response)

    lam1, lam2 = _annealed_lambdas(args)
    score = r_perf + lam1 * r_parallel + lam2 * r_finish - COST_PER_CALL * n_calls

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
        "lambda1": lam1,
        "lambda2": lam2,
        "pred": perf.get("pred", "") or "",
    }


async def reward_func(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        return [_score_one(args, s) for s in sample_or_samples]
    return _score_one(args, sample_or_samples)
