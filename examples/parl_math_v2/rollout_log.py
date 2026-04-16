"""PARL v2 custom rollout logger.

Bolts two things onto miles' default wandb logging:

1. Per-component reward stats. `reward.py::reward_func` returns a dict
   (`r_perf`, `r_parallel`, `r_finish`, `n_spawn`, `n_solvers_valid/total`,
   `lambda1/2`) but miles only pipes `args.reward_key=score` to wandb —
   everything else is invisible. We unpack the dict and log mean/std/p50/
   max/min per key under `reward/<key>/…`.

2. Critical-step (tool-decision) metrics for K2.5 PARL. Each turn in the
   orchestrator is a "spawn or answer" decision, so we aggregate:
     - spawn_rate, turns_per_rollout, effective_response_ratio (loss-mask
       coverage — drops if solver observation tokens leak into loss);
     - n_spawn / r_parallel **within-group std** averaged across prompts —
       the single hardest check that GRPO advantages on dispatch are
       non-degenerate (see plan §"风险与对策");
     - spawn-by-difficulty: bucket prompts by group r_perf pass rate and
       report mean n_spawn per bucket — direct proxy for "learned
       scheduling" (plan验收 Prod §2);
     - solver aggregate success rate = Σvalid / Σtotal.

Returns False so miles' default logger still runs after this hook — we
want `rollout/reward/...` (the scalar score) and perf metrics too.
"""

from __future__ import annotations

import numpy as np

from miles.utils import tracking_utils
from miles.utils.iter_utils import group_by
from miles.utils.metric_utils import compute_rollout_step

_REWARD_KEYS = (
    "r_perf",
    "r_parallel",
    "r_finish",
    "n_spawn",
    "n_solvers_valid",
    "n_solvers_total",
    "critical_steps",
    "critical_steps_ratio",
    "lambda1",
    "lambda2",
    "lambda3",
)


def _stats(values, prefix):
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}/mean": float(arr.mean()),
        f"{prefix}/std": float(arr.std()),
        f"{prefix}/p50": float(np.median(arr)),
        f"{prefix}/max": float(arr.max()),
        f"{prefix}/min": float(arr.min()),
    }


def _extract(samples, key):
    out = []
    for s in samples:
        r = s.reward
        if isinstance(r, dict) and key in r and r[key] is not None:
            out.append(float(r[key]))
    return out


def _compute_reward_component_metrics(samples):
    log_dict = {}
    for key in _REWARD_KEYS:
        log_dict |= _stats(_extract(samples, key), f"reward/{key}")

    valid = sum(_extract(samples, "n_solvers_valid"))
    total = sum(_extract(samples, "n_solvers_total"))
    if total > 0:
        log_dict["reward/solver_success_rate"] = float(valid / total)
    return log_dict


def _compute_multi_turn_metrics(args, samples):
    log_dict = {}

    n_spawn = _extract(samples, "n_spawn")
    if n_spawn:
        arr = np.asarray(n_spawn)
        log_dict["multi_turn/spawn_rate"] = float((arr > 0).mean())
        log_dict |= _stats(n_spawn, "multi_turn/n_spawn")

    # turns_per_rollout ≈ 1 (final answer) + n_spawn (each spawn is its own turn)
    if n_spawn:
        log_dict |= _stats([1 + v for v in n_spawn], "multi_turn/turns_per_rollout")

    # loss-mask coverage: effective / total response length.
    ratios = []
    for s in samples:
        if s.response_length and s.response_length > 0:
            ratios.append(s.effective_response_length / s.response_length)
    if ratios:
        log_dict |= _stats(ratios, "multi_turn/effective_response_ratio")

    # GRPO within-group std: if this is ~0, dispatch gradients collapse.
    if args.advantage_estimator != "ppo":
        groups = group_by(samples, lambda s: s.group_index)
        for key in ("n_spawn", "r_parallel", "r_perf"):
            per_group_std = []
            for g in groups.values():
                vals = _extract(g, key)
                if len(vals) >= 2:
                    per_group_std.append(float(np.std(vals)))
            if per_group_std:
                log_dict[f"multi_turn/group_std/{key}/mean"] = float(np.mean(per_group_std))
                log_dict[f"multi_turn/group_std/{key}/frac_nonzero"] = float(
                    np.mean([v > 1e-6 for v in per_group_std])
                )

        # Spawn-by-difficulty: bucket prompts by group mean r_perf, report n_spawn.
        for label, lo, hi in (("hard", 0.0, 0.25), ("mid", 0.25, 0.75), ("easy", 0.75, 1.01)):
            bucket_spawn = []
            for g in groups.values():
                perfs = _extract(g, "r_perf")
                spawns = _extract(g, "n_spawn")
                if not perfs or not spawns:
                    continue
                pass_rate = float(np.mean(perfs))
                if lo <= pass_rate < hi:
                    bucket_spawn.extend(spawns)
            if bucket_spawn:
                log_dict[f"multi_turn/n_spawn_by_diff/{label}/mean"] = float(np.mean(bucket_spawn))
                log_dict[f"multi_turn/n_spawn_by_diff/{label}/count"] = len(bucket_spawn)

    return log_dict


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    log_dict = {}
    log_dict |= _compute_reward_component_metrics(samples)
    log_dict |= _compute_multi_turn_metrics(args, samples)

    if log_dict:
        step = compute_rollout_step(args, rollout_id)
        log_dict["rollout/step"] = step
        tracking_utils.log(args, log_dict, step_key="rollout/step")

    # Return False so miles' default _log_rollout_data still runs (we want
    # response_len, truncated_ratio, the scalar rollout/reward, perf/*, etc.).
    return False


def log_eval_rollout_data(rollout_id, args, data, extra_metrics):
    log_dict = {}
    for eval_key, payload in data.items():
        samples = payload.get("samples")
        if not samples:
            continue
        log_dict |= {
            f"eval/{eval_key}/{k}": v
            for k, v in _compute_reward_component_metrics(samples).items()
        }
        log_dict |= {
            f"eval/{eval_key}/{k}": v
            for k, v in _compute_multi_turn_metrics(args, samples).items()
        }

    if log_dict:
        step = compute_rollout_step(args, rollout_id)
        log_dict["eval/step"] = step
        tracking_utils.log(args, log_dict, step_key="eval/step")

    return False
