"""PARL v2 custom rollout logger (agent-swarm flavor).

Bolts two things onto miles' default wandb logging:

1. Per-component reward stats. ``reward.py::reward_func`` returns a dict
   (``r_perf``, ``r_parallel``, ``r_finish``, ``r_box``, ``n_assign``,
   ``n_create``, ``n_valid``, ``registry_size``, ``critical_steps``,
   ``lambda1/2/_box``) but miles only pipes ``args.reward_key=score`` to
   wandb. We unpack and log mean/std/p50/max/min per key under
   ``reward/<key>/…``.

2. Agent-swarm multi-turn metrics. Each orchestrator turn is a
   "spawn / create / answer" decision; we aggregate from
   ``sample.metadata["turns"]`` and friends:
     - ``assign_rate`` (any turn with n_assign > 0) and per-turn n_assign
       distribution;
     - ``turns_per_rollout`` = len(metadata["turns"]);
     - ``effective_response_ratio`` (loss-mask coverage — drops if subagent
       observation tokens leak into loss);
     - ``n_assign`` / ``r_parallel`` within-group std averaged across prompts
       — the single hardest check that GRPO advantages on dispatch are
       non-degenerate (see plan §"风险与对策");
     - assign-by-difficulty: bucket prompts by group r_perf pass rate and
       report mean ``n_assign`` per bucket — direct proxy for "learned
       scheduling";
     - ``n_unique_agents_used`` per rollout (distinct agent names appearing
       in valid assign_task calls, approximated as ``registry_size``);
     - ``solver_success_rate`` = Σ n_valid / Σ n_assign.

Returns False so miles' default logger still runs after this hook.
"""

from __future__ import annotations

import numpy as np

from miles.utils import tracking_utils
from miles.utils.iter_utils import group_by
from miles.utils.metric_utils import compute_rollout_step
from miles.rollout.sglang_rollout import get_model_url

_REWARD_KEYS = (
    "r_perf",
    "r_parallel",
    "r_finish",
    "r_box",
    "n_assign",
    "n_create",
    "n_valid",
    "registry_size",
    "critical_steps",
    "lambda1",
    "lambda2",
    "lambda_box",
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


def _per_turn_assign_counts(samples):
    """Flatten per-turn n_assign across all samples, for distribution stats."""
    out = []
    for s in samples:
        turns = (s.metadata or {}).get("turns") or []
        for t in turns:
            out.append(float(t.get("n_assign", 0)))
    return out


def _compute_reward_component_metrics(samples):
    log_dict = {}
    for key in _REWARD_KEYS:
        log_dict |= _stats(_extract(samples, key), f"reward/{key}")

    valid = sum(_extract(samples, "n_valid"))
    total = sum(_extract(samples, "n_assign"))
    if total > 0:
        log_dict["reward/solver_success_rate"] = float(valid / total)
    return log_dict


def _compute_multi_turn_metrics(args, samples):
    log_dict = {}

    n_assign = _extract(samples, "n_assign")
    if n_assign:
        arr = np.asarray(n_assign)
        log_dict["multi_turn/assign_rate"] = float((arr > 0).mean())
        log_dict |= _stats(n_assign, "multi_turn/n_assign")

    # per-turn n_assign distribution (how many subagents spawned per spawn turn).
    per_turn_assigns = _per_turn_assign_counts(samples)
    if per_turn_assigns:
        log_dict |= _stats(per_turn_assigns, "multi_turn/assign_per_turn")

    # turns_per_rollout — total orchestrator turns actually taken.
    turn_counts = [len((s.metadata or {}).get("turns") or []) for s in samples]
    if turn_counts:
        log_dict |= _stats(turn_counts, "multi_turn/turns_per_rollout")

    # loss-mask coverage: effective / total response length.
    ratios = []
    for s in samples:
        if s.response_length and s.response_length > 0:
            ratios.append(s.effective_response_length / s.response_length)
    if ratios:
        log_dict |= _stats(ratios, "multi_turn/effective_response_ratio")

    # Unique specialists created per rollout (registry_size proxy).
    registry_sizes = _extract(samples, "registry_size")
    if registry_sizes:
        log_dict |= _stats(registry_sizes, "multi_turn/n_unique_agents_used")

    # GRPO within-group std: if this is ~0, dispatch gradients collapse.
    if args.advantage_estimator != "ppo":
        groups = group_by(samples, lambda s: s.group_index)
        for key in ("n_assign", "r_parallel", "r_perf"):
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

        # Assign-by-difficulty: bucket prompts by group mean r_perf, report n_assign.
        for label, lo, hi in (("hard", 0.0, 0.25), ("mid", 0.25, 0.75), ("easy", 0.75, 1.01)):
            bucket = []
            for g in groups.values():
                perfs = _extract(g, "r_perf")
                assigns = _extract(g, "n_assign")
                if not perfs or not assigns:
                    continue
                pass_rate = float(np.mean(perfs))
                if lo <= pass_rate < hi:
                    bucket.extend(assigns)
            if bucket:
                log_dict[f"multi_turn/n_assign_by_diff/{label}/mean"] = float(np.mean(bucket))
                log_dict[f"multi_turn/n_assign_by_diff/{label}/count"] = len(bucket)

    return log_dict


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    log_dict = {}
    log_dict |= _compute_reward_component_metrics(samples)
    log_dict |= _compute_multi_turn_metrics(args, samples)

    sub_url = get_model_url(args, "subagent")
    live_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    log_dict["parl/subagent_endpoint_distinct"] = int(sub_url != live_url)

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
        log_dict |= {f"eval/{eval_key}/{k}": v for k, v in _compute_reward_component_metrics(samples).items()}
        log_dict |= {f"eval/{eval_key}/{k}": v for k, v in _compute_multi_turn_metrics(args, samples).items()}

    if log_dict:
        step = compute_rollout_step(args, rollout_id)
        log_dict["eval/step"] = step
        tracking_utils.log(args, log_dict, step_key="eval/step")

    return False
