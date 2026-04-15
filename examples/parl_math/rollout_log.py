"""Custom rollout log fn: aggregates PARL-specific metadata to wandb.

Returns False so miles' default rollout logging still runs after this hook.
"""

from __future__ import annotations

import numpy as np

from miles.utils import tracking_utils
from miles.utils.metric_utils import compute_rollout_step


def _stat(values, prefix):
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}/mean": float(arr.mean()),
        f"{prefix}/p50": float(np.median(arr)),
        f"{prefix}/max": float(arr.max()),
        f"{prefix}/min": float(arr.min()),
    }


def log_rollout_with_parl(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    log_dict = {}

    crit = [s.metadata.get("critical_steps") for s in samples if s.metadata]
    crit = [v for v in crit if v is not None]
    log_dict |= _stat(crit, "rollout/parl/critical_steps")

    num_stages = [s.metadata.get("num_stages") for s in samples if s.metadata]
    num_stages = [v for v in num_stages if v is not None]
    log_dict |= _stat(num_stages, "rollout/parl/num_stages")

    # Per-stage breakdown: stage_i/max-token across batch. Pad-aware.
    per_stage = [s.metadata.get("critical_steps_per_stage") for s in samples if s.metadata]
    per_stage = [v for v in per_stage if v]
    if per_stage:
        max_depth = max(len(v) for v in per_stage)
        for i in range(max_depth):
            stage_vals = [v[i] for v in per_stage if len(v) > i]
            log_dict |= _stat(stage_vals, f"rollout/parl/stage_{i}_tokens")

    if log_dict:
        step = compute_rollout_step(args, rollout_id)
        log_dict["rollout/step"] = step
        tracking_utils.log(args, log_dict, step_key="rollout/step")

    return False
