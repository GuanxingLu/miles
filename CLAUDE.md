# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Miles** is an enterprise-grade RL post-training framework for large language models, forked from [slime](https://github.com/THUDM/slime). It couples **SGLang** (rollout/inference) with **Megatron-LM** (training) and also supports an FSDP backend. Core focus: stability and efficiency for large MoE + low-precision (FP8 / INT4 QAT) training, and eliminating train-inference mismatch.

## Entry points

- `train.py` — sync / colocated RL training loop (rollout → train → weight update).
- `train_async.py` — fully async variant; prefetches the next rollout while training. Requires `--colocate` off.

Both call into `miles.ray.placement_group.create_placement_groups` → `create_rollout_manager` → `create_training_models`, then iterate `rollout_manager.generate` → `actor_model.train` → `actor_model.update_weights`. Args are parsed by `miles.utils.arguments.parse_args`.

Launches go through Ray: scripts do `ray start --head … && ray job submit … -- python3 train.py …`.

## Commands

### Run training
Training is driven by launcher shell scripts under `scripts/` (e.g. `scripts/run-qwen3-4B.sh`, `scripts/run-qwen3-235B-A22B.sh`). Each script sources a model config from `scripts/models/` and assembles arg groups (`CKPT_ARGS`, `ROLLOUT_ARGS`, `PERF_ARGS`, `GRPO_ARGS`, …) passed to `train.py`. When behavior needs to change per-run, edit the launch script rather than patching library code in `miles/utils/`.

Example configs also live under `examples/` (multi-agent, VLM, fully_async, retool, true_on_policy, etc.) — those are the canonical references for non-trivial rollout/reward customization.

### Tests
Pytest is configured via `pyproject.toml` (`asyncio_mode = "auto"`, `testpaths = ./tests`). Tests are split by folder, not markers:
- `tests/fast/` — fast unit tests (backends, rollout, router, utils). CI runs with `pytest tests/fast`.
- `tests/e2e/` — GPU end-to-end training runs (short/long, fsdp, megatron, sglang, precision, ckpt, lora). Run individual files, e.g. `pytest tests/e2e/short/test_qwen2.5_0.5B_gsm8k_short.py`.
- `tests/ci/` — GitHub self-hosted runner infra (not a test target).

`tests/fast/conftest.py` auto-sets `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1` — fast tests exercise the experimental rollout path.

Run a single test: `pytest tests/fast/path/to/test_file.py::test_name -vv`.

### Lint / format
Pre-commit is the source of truth (`.pre-commit-config.yaml`): ruff (with `--fix`), autoflake, isort (black profile), black (line length 119). Run `pre-commit run --all-files`.

There is a local pygrep hook **banning direct `mpu.get_data_parallel*` calls** — always go through `miles/backends/megatron_utils/parallel.py` wrappers (the two allowed exceptions are that file itself and `tools/convert_to_hf.py`).

## Code layout

- `miles/ray/` — Ray placement groups and actor wiring. `placement_group.py` is the entry glue used by both `train.py` and `train_async.py`. `actor_group.py`, `train_actor.py`, `rollout.py` define the ray actor classes.
- `miles/backends/` — training + inference integrations.
  - `megatron_utils/` — Megatron actor, model provider, parallel wrappers, checkpoint, LoRA, update-weight paths, `megatron_to_hf` conversion. This is where Megatron-specific changes go.
  - `sglang_utils/` — SGLang engine wrapper + SGLang arg plumbing.
  - `training_utils/` — backend-agnostic training helpers (loss, CP, parallel, data, logging).
  - `experimental/` — in-flight backend work (e.g. FSDP); not stable.
- `miles/rollout/` — rollout orchestration. `sglang_rollout.py` is the primary path; `inference_rollout/`, `generate_hub/`, `generate_utils/`, `session/` handle multi-turn / agent loops; `rm_hub/` + `filter_hub/` hold reward models and rollout filters; `sft_rollout.py` / `sleep_rollout.py` are specialized modes.
- `miles/router/` — request routing (referenced by R3 / routing-replay features).
- `miles/utils/` — shared utilities (arg parsing, PPO math, metrics, dumpers, FP8 kernels, tracking, tensor backper, etc.). `arguments.py` owns the CLI schema.
- `miles_plugins/` — pluggable integrations (`mbridge`, `megatron_bridge`, custom models) registered at runtime.
- `examples/` — canonical end-to-end recipes (multi-agent, VLM multi-turn, fully_async, retool, true-on-policy, low_precision, DrGRPO, …). Prefer copying from here when building new configurations.
- `scripts/` — production launcher shell scripts + per-model env configs in `scripts/models/`.
- `tools/` — one-off utilities (e.g. `convert_to_hf.py`).
- `docker/` — Dockerfiles (H/B/GB300 CUDA + ROCm MI300/MI350 variants) and build helpers.
- `docs/en/` — user docs (`get_started/`, `advanced/`, `developer_guide/`, `agentic/`, `platform_support/`).

## Conventions specific to this repo

- **Never call `mpu.get_data_parallel*` directly** (enforced by pre-commit); use the wrappers in `miles/backends/megatron_utils/parallel.py`.
- When a training run misbehaves and the root cause is configuration, fix the launch script in `scripts/` or the example config rather than adding workarounds inside `miles/utils/`.
- Launch scripts always preamble with `pkill -9 sglang && ray stop --force && pkill -9 ray && pkill -9 python` — previous runs leave Ray/SGLang processes around; reuse this pattern for new launchers.
- Async training (`train_async.py`) asserts `not args.colocate` — colocation is a sync-only path.
