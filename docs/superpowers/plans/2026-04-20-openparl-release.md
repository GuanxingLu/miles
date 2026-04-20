# OpenPARL Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Open-source the WideSearch PARL reproduction as two GitHub artifacts — a clean `miles@openparl-v1` branch carrying ~193 LOC of framework hooks, and a standalone `OpenPARL` repo with the agent code, RAG server, launchers, and docs.

**Architecture:** Two-repo split. `GuanxingLu/miles` hosts a new branch `openparl-v1` (based on `radixark/miles@5d11fe2f0`) containing 6 paper-legible hook commits distilled from `dev/guanxing`. `GuanxingLu/OpenPARL` is a fresh standalone Python package (`src/openparl/`) that depends on the miles tag via pip-install-from-git. All blog prose and figures are out of scope for v0.1; author fills them from existing wandb runs at blog-writing time.

**Tech Stack:** Python 3.10+, miles (fork), SGLang, Megatron-LM, Ray, pytest, Apache-2.0 license.

**Spec:** `docs/superpowers/specs/2026-04-20-openparl-release-design.md`

**Working directories:**
- `/Users/a74739/code/miles/` — source checkout (dev/guanxing), read-only reference
- `/Users/a74739/code/miles-openparl/` — NEW: git worktree for the clean miles branch
- `/Users/a74739/code/OpenPARL/` — NEW: standalone repo

---

## File Structure

### Miles fork branch (`GuanxingLu/miles@openparl-v1`)

12 files touched relative to `radixark/miles@5d11fe2f0`, +193/−27 LOC total. Split into 6 commits:

| Commit | Files |
|---|---|
| `feat(sample): per-token advantages` | `miles/utils/types.py`, `miles/utils/data.py`, `miles/backends/training_utils/data.py` |
| `feat(loss): critical-step + token-level clip` | `miles/backends/training_utils/loss.py` |
| `feat(args): --disable-entropy-computation` | `miles/utils/arguments.py` |
| `feat(rollout): frozen-subagent weight sync` | `miles/ray/rollout.py`, `miles/rollout/sglang_rollout.py`, `miles/rollout/generate_hub/multi_turn.py` |
| `feat(metrics): multi-agent + false-tool-call + critical-step` | `miles/utils/metric_utils.py`, `miles/backends/training_utils/log_utils.py` |
| `feat(rollout): Orchestrator→Subagent inference hooks` | `miles/rollout/inference_rollout/inference_rollout_common.py`, `miles/rollout/inference_rollout/inference_rollout_eval.py` |

### `GuanxingLu/OpenPARL` repo

```
OpenPARL/
├── README.md                              # hero + install + result-table placeholder
├── BLOG.md                                # skeleton (author fills Section 5 later)
├── LICENSE                                # Apache-2.0
├── NOTICE                                 # attribution
├── pyproject.toml                         # package metadata
├── install.sh                             # pip install miles@tag + -e .
├── .gitignore
│
├── src/openparl/
│   ├── __init__.py
│   ├── prompts.py                         # from examples/parl_v2/prompts.py
│   ├── generate.py                        # from examples/parl_v2/generate.py
│   ├── rollout_log.py                     # from examples/parl_v2/rollout_log.py
│   ├── run.py                             # from examples/parl_v2/run_parl_v2.py (renamed)
│   ├── tool.py                            # from examples/parl_v2/tool.py
│   └── widesearch/
│       ├── __init__.py
│       ├── assign_task.py
│       ├── orchestrator_tools.py
│       ├── prepare_data.py
│       ├── reward.py
│       ├── reward_utils.py
│       ├── search_client.py
│       └── subagent_prompts.py
│
├── third_party/
│   └── rag_server/
│       ├── CREDITS.md                     # cites RLinf source
│       ├── build_index.py
│       ├── local_retrieval_server.py
│       └── qdrant_encoder.py
│
├── configs/
│   ├── sglang_4B.yaml
│   └── sglang_0.6B.yaml
│
├── scripts/
│   ├── run-qwen3-4B-parl.sh               # renamed from run-qwen3-4B-widesearch.sh
│   ├── run-qwen3-4B-single.sh             # renamed from run-qwen3-4B-widesearch-single.sh
│   ├── run-qwen3-4B-orchestrator_only.sh  # renamed from run-qwen3-4B-widesearch-paper.sh
│   ├── run-qwen3-0.6B-parl.sh             # renamed from run-qwen3-0.6B-widesearch.sh
│   └── launch_rag_server.sh
│
├── tests/
│   ├── __init__.py
│   ├── test_reward.py                     # from tests/fast/examples/parl_v2/widesearch/
│   └── test_reward_utils.py
│
├── docs/
│   ├── architecture.md
│   ├── reward.md
│   └── reproducibility.md
│
└── .github/workflows/
    └── fast-tests.yml                     # pytest tests/ on CPU
```

### Files NOT copied from source

| Source path | Reason |
|---|---|
| `examples/parl_v2/widesearch/robbyctl_remote/` | Ant Group internal launcher |
| `examples/parl_v2/widesearch/job_config_robbys3.yml` | Ant Group internal AI Studio config |
| `examples/parl_v2/math/` | Out of v0.1 scope (WideSearch only) |
| `examples/parl_v2/sglang_config_30B_A3B.yaml` | 30B untested at release |
| `examples/parl_v2/sglang_config_30B_A3B_2node.yaml` | same |
| `.claude/` | internal agent scratchpads |

---

## Phase A — Miles fork: clean branch (Tasks 1–8)

### Task 1: Create worktree for clean miles branch

**Files:** none modified in source; produces new worktree at `/Users/a74739/code/miles-openparl/`.

- [ ] **Step 1: Create worktree off the pinned base SHA**

```bash
cd /Users/a74739/code/miles
git worktree add -b openparl-v1 /Users/a74739/code/miles-openparl 5d11fe2f0
```

- [ ] **Step 2: Verify the worktree is clean and at the right SHA**

```bash
cd /Users/a74739/code/miles-openparl
git log --oneline -1
git status
```

Expected output: HEAD is `5d11fe2f0 fix: handle non-tool appended messages in TITO incremental tokenization (#949)` (or whichever subject `5d11fe2f0` carries), working tree clean.

- [ ] **Step 3: Confirm upstream `radixark/miles` is reachable at this SHA**

```bash
git log -1 5d11fe2f0 --format='%H %s'
```

Expected: prints full SHA + subject. No push needed yet.

### Task 2: Build commit 1 — per-token advantages

**Files (checkout from `dev/guanxing`):**
- Modify: `miles/utils/types.py`
- Modify: `miles/utils/data.py`
- Modify: `miles/backends/training_utils/data.py`

- [ ] **Step 1: Checkout the three files from dev/guanxing**

```bash
cd /Users/a74739/code/miles-openparl
git checkout dev/guanxing -- miles/utils/types.py miles/utils/data.py miles/backends/training_utils/data.py
```

- [ ] **Step 2: Review diff and confirm only per-token-advantage bits are staged**

```bash
git diff --cached miles/utils/types.py miles/utils/data.py miles/backends/training_utils/data.py
```

Expected hunks:
- `types.py`: `per_token_advantages: list[float] | None` field, post-init length assertion, updates in `strip_last_output_tokens` + `reset_for_retry`.
- `utils/data.py`: small update so `per_token_advantages` flows through whatever serialization path `utils/data.py` handles (re-check; if the hunk turns out to be unrelated, revert that file via `git restore --staged --worktree miles/utils/data.py`).
- `backends/training_utils/data.py`: per-token-advantage propagation into training batch.

If any hunk is unrelated (e.g. a log_utils change bled in), use `git restore --patch --staged --worktree <file>` to drop it.

- [ ] **Step 3: Run the fast-path unit tests for types**

```bash
cd /Users/a74739/code/miles-openparl
pip install -e . --quiet
pytest tests/fast/ -k "types or sample" -x -q
```

Expected: PASS (or "no tests collected" if none match — acceptable; the real validation happens at Task 8 when running the whole fast suite).

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(sample): per-token advantages for turn-level credit assignment

Adds `Sample.per_token_advantages` (list[float] | None). When populated,
the GRPO/GSPO advantage estimator uses it directly instead of broadcasting
a scalar reward across the response, enabling segment- / turn-level credit
assignment for multi-agent rollouts (K2.5 PARL, arXiv:2602.02276).

* `miles/utils/types.py`: field + post-init length check + maintenance in
  `strip_last_output_tokens` / `reset_for_retry`.
* `miles/backends/training_utils/data.py`: batch-building path forwards the
  array when present, falls back to scalar advantage otherwise.
* `miles/utils/data.py`: serialization hook.

Whoever populates the field owns any group baseline / normalization they
want; downstream `normalize_advantages` still whitens as before.
EOF
)"
```

### Task 3: Build commit 2 — critical-step + token-level clip

**Files:** `miles/backends/training_utils/loss.py`

- [ ] **Step 1: Checkout loss.py**

```bash
cd /Users/a74739/code/miles-openparl
git checkout dev/guanxing -- miles/backends/training_utils/loss.py
```

- [ ] **Step 2: Review staged hunks**

```bash
git diff --cached miles/backends/training_utils/loss.py
```

Expected: ~34 added lines implementing (a) critical-step reward weighting, (b) K2.5 token-level gradient masking (log-ratio out of `[α, β]` → grad zero; advantage-agnostic — distinct from PPO's clip-min).

- [ ] **Step 3: Smoke-test**

```bash
pytest tests/fast/backends/ -k "loss" -x -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(loss): critical-step reward weighting + token-level clip

Two policy-loss additions needed for K2.5 PARL reproduction
(arXiv:2602.02276, §4.2):

* Critical-step reward weighting: response tokens belonging to the critical
  path (Orchestrator + longest subagent in each parallel group) carry the
  full advantage; non-critical tokens are zeroed out.
* Token-level gradient masking: log-ratio outside `[α, β]` → grad set to
  zero. Advantage-agnostic, distinct from PPO's clip-min semantics.
  Mitigates train-inference framework numerical discrepancy.
EOF
)"
```

### Task 4: Build commit 3 — `--disable-entropy-computation`

**Files:** `miles/utils/arguments.py`

- [ ] **Step 1: Checkout**

```bash
cd /Users/a74739/code/miles-openparl
git checkout dev/guanxing -- miles/utils/arguments.py
```

- [ ] **Step 2: Review staged hunks**

```bash
git diff --cached miles/utils/arguments.py
```

Expected: +12 lines adding `--disable-entropy-computation` flag (sets `dest="compute_entropy"` false, default True).

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(args): --disable-entropy-computation flag

Skips the entropy term in the policy loss entirely, avoiding the fp32
`N * (vocab/TP) * 4 byte` peak allocation inside
`compute_entropy_from_logits` (plus its bf16 `logits.clone()`).

Useful when `--entropy-coef 0` and the entropy metric is not needed —
specifically required to fit Qwen3-4B trainable Orchestrator + Qwen3-0.6B
frozen Subagent into a single H200 node. When disabled, the `entropy_loss`
metric is reported as 0.
EOF
)"
```

### Task 5: Build commit 4 — frozen-subagent weight sync

**Files:** `miles/ray/rollout.py`, `miles/rollout/sglang_rollout.py`, `miles/rollout/generate_hub/multi_turn.py`

- [ ] **Step 1: Checkout**

```bash
cd /Users/a74739/code/miles-openparl
git checkout dev/guanxing -- miles/ray/rollout.py miles/rollout/sglang_rollout.py miles/rollout/generate_hub/multi_turn.py
```

- [ ] **Step 2: Review diff**

```bash
git diff --cached miles/ray/rollout.py miles/rollout/sglang_rollout.py miles/rollout/generate_hub/multi_turn.py
```

Expected:
- `ray/rollout.py` (+27): enables CPU backup for frozen-subagent weights across actor `update_weights` cycles (colocate path).
- `sglang_rollout.py` (+14): hook point for the separate frozen-subagent SGLang engine.
- `multi_turn.py` (+11): `assign_task` dispatch honoring the frozen subagent's router.

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(rollout): frozen-subagent weight sync for colocate mode

Keeps frozen Subagent SGLang engine weights alive across trainable-
Orchestrator `update_weights` cycles. Without this, the Subagent engine's
weights get nulled when the Actor sends new Orchestrator weights in
colocate mode, silently breaking multi-agent rollout.

* `miles/ray/rollout.py`: enable_weights_cpu_backup plumbing.
* `miles/rollout/sglang_rollout.py`: two-engine-aware update path.
* `miles/rollout/generate_hub/multi_turn.py`: `assign_task` routes to the
  frozen-subagent router URL.
EOF
)"
```

### Task 6: Build commit 5 — multi-agent metrics

**Files:** `miles/utils/metric_utils.py`, `miles/backends/training_utils/log_utils.py`

- [ ] **Step 1: Checkout**

```bash
cd /Users/a74739/code/miles-openparl
git checkout dev/guanxing -- miles/utils/metric_utils.py miles/backends/training_utils/log_utils.py
```

- [ ] **Step 2: Review diff**

```bash
git diff --cached miles/utils/metric_utils.py miles/backends/training_utils/log_utils.py
```

Expected:
- `metric_utils.py` (+59): per-subagent reward aggregation, false-tool-call rate (JSON-parse failures + schema violations), critical-step distribution helpers.
- `log_utils.py` (+13): wandb / tracker wiring for the new metrics.

- [ ] **Step 3: Run metric tests**

```bash
pytest tests/fast/utils/test_metric_utils.py -x -q
```

These tests already exist on `dev/guanxing` at `tests/fast/utils/test_metric_utils.py` (+99 lines). Since we only checked out the library files, the test file is NOT on this branch yet. Expected: `no tests ran` or `FileNotFoundError`. That's fine — we do not ship these tests on the miles branch; they belong in OpenPARL (Task 14 copies them to a different path).

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(metrics): multi-agent reward + false-tool-call + critical-step metrics

* Per-subagent reward aggregation: tracks r_perf / r_finish / r_parallel
  contribution per rollout.
* False-tool-call rate: fraction of model-emitted tool calls that fail
  JSON parse or schema validation. Useful signal for agent-swarm training
  where bad `create_subagent` / `assign_task` calls are common failure mode.
* Critical-step distribution: histograms of per-episode critical-step count
  for the reward weighting in `loss.py`.

Plumbed into the existing wandb tracker via `log_utils.py`.
EOF
)"
```

### Task 7: Build commit 6 — Orchestrator → Subagent inference hooks

**Files:** `miles/rollout/inference_rollout/inference_rollout_common.py`, `miles/rollout/inference_rollout/inference_rollout_eval.py`

- [ ] **Step 1: Checkout**

```bash
cd /Users/a74739/code/miles-openparl
git checkout dev/guanxing -- miles/rollout/inference_rollout/inference_rollout_common.py miles/rollout/inference_rollout/inference_rollout_eval.py
```

- [ ] **Step 2: Review diff**

```bash
git diff --cached miles/rollout/inference_rollout/
```

Expected: ~6 net lines. Thin hook points so user-space `assign_task` can route to the frozen-Subagent SGLang endpoint during both training and eval rollouts.

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(rollout): Orchestrator → Subagent inference hooks

Minimal hook points in `inference_rollout_common` / `inference_rollout_eval`
so user-space `assign_task` (in examples / downstream code) can route to
the frozen-Subagent SGLang endpoint during both training and eval rollouts.
EOF
)"
```

### Task 8: Verify miles branch + tag + push

- [ ] **Step 1: Confirm diff against base matches the spec's +193/−27 envelope**

```bash
cd /Users/a74739/code/miles-openparl
git diff --shortstat 5d11fe2f0..HEAD
```

Expected: `12 files changed, <≈193> insertions(+), <≈27> deletions(-)`. Minor deviation (±10 lines) is fine; major deviation (>50) means a hunk crept in from an unrelated commit — investigate.

- [ ] **Step 2: Confirm commit list is clean and paper-legible**

```bash
git log --oneline 5d11fe2f0..HEAD
```

Expected: exactly 6 commits, titles:
- `feat(sample): per-token advantages for turn-level credit assignment`
- `feat(loss): critical-step reward weighting + token-level clip`
- `feat(args): --disable-entropy-computation flag`
- `feat(rollout): frozen-subagent weight sync for colocate mode`
- `feat(metrics): multi-agent reward + false-tool-call + critical-step metrics`
- `feat(rollout): Orchestrator → Subagent inference hooks`

- [ ] **Step 3: Run full fast test suite**

```bash
MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1 pytest tests/fast -q
```

Expected: all PASS (or same pass rate as base — compare with `5d11fe2f0` baseline if anything fails). If failures appear only on the new branch, inspect the offending hook commit.

- [ ] **Step 4: Tag**

```bash
git tag -a v0.1-openparl -m "OpenPARL v0.1: PARL framework hooks on radixark/miles@5d11fe2f0"
```

- [ ] **Step 5: Push branch and tag to `guanxing` remote**

```bash
git push guanxing openparl-v1
git push guanxing v0.1-openparl
```

Verify at `https://github.com/GuanxingLu/miles/tree/openparl-v1` — branch visible, tag visible. **STOP and ask the author to confirm** before proceeding to Phase B (so they can spot-check the GitHub compare view).

---

## Phase B — OpenPARL standalone repo (Tasks 9–21)

### Task 9: Initialize OpenPARL repo skeleton

- [ ] **Step 1: Create directory + git init**

```bash
mkdir -p /Users/a74739/code/OpenPARL
cd /Users/a74739/code/OpenPARL
git init -b main
```

- [ ] **Step 2: Create top-level directory skeleton**

```bash
mkdir -p src/openparl/widesearch third_party/rag_server configs scripts tests docs .github/workflows
touch src/openparl/__init__.py src/openparl/widesearch/__init__.py tests/__init__.py
```

- [ ] **Step 3: Write `.gitignore`**

File: `/Users/a74739/code/OpenPARL/.gitignore`
```gitignore
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.venv/
venv/
env/
build/
dist/
wandb/
outputs/
saves/
*.log
.DS_Store
.idea/
.vscode/
```

- [ ] **Step 4: First commit**

```bash
git add .
git commit -m "chore: init OpenPARL repo skeleton"
```

### Task 10: Copy core `openparl` package files

**Files:**
- Create: `src/openparl/prompts.py` (from `/Users/a74739/code/miles/examples/parl_v2/prompts.py`)
- Create: `src/openparl/generate.py` (from `/Users/a74739/code/miles/examples/parl_v2/generate.py`)
- Create: `src/openparl/rollout_log.py` (from `/Users/a74739/code/miles/examples/parl_v2/rollout_log.py`)
- Create: `src/openparl/run.py` (from `/Users/a74739/code/miles/examples/parl_v2/run_parl_v2.py`)
- Create: `src/openparl/tool.py` (from `/Users/a74739/code/miles/examples/parl_v2/tool.py`)

- [ ] **Step 1: Copy**

```bash
SRC=/Users/a74739/code/miles/examples/parl_v2
DST=/Users/a74739/code/OpenPARL/src/openparl

cp "$SRC/prompts.py" "$DST/prompts.py"
cp "$SRC/generate.py" "$DST/generate.py"
cp "$SRC/rollout_log.py" "$DST/rollout_log.py"
cp "$SRC/run_parl_v2.py" "$DST/run.py"
cp "$SRC/tool.py" "$DST/tool.py"
```

- [ ] **Step 2: Verify all five files exist**

```bash
ls -la /Users/a74739/code/OpenPARL/src/openparl/
```

Expected: `__init__.py`, `prompts.py`, `generate.py`, `rollout_log.py`, `run.py`, `tool.py`, `widesearch/`.

- [ ] **Step 3: Commit**

```bash
cd /Users/a74739/code/OpenPARL
git add src/openparl/
git commit -m "feat: vendor parl_v2 core modules (prompts, generate, rollout_log, run, tool)"
```

### Task 11: Copy widesearch submodule

**Files:**
- Create: `src/openparl/widesearch/*.py` (from `examples/parl_v2/widesearch/`)

- [ ] **Step 1: Copy all widesearch .py files**

```bash
SRC=/Users/a74739/code/miles/examples/parl_v2/widesearch
DST=/Users/a74739/code/OpenPARL/src/openparl/widesearch

for f in assign_task.py orchestrator_tools.py prepare_data.py reward.py reward_utils.py search_client.py subagent_prompts.py; do
  cp "$SRC/$f" "$DST/$f"
done
```

- [ ] **Step 2: Verify excluded paths are NOT present**

```bash
ls -la /Users/a74739/code/OpenPARL/src/openparl/widesearch/
```

Expected: 7 .py files + `__init__.py`. No `robbyctl_remote/`, no `job_config_robbys3.yml`, no `rag_server/` (that's Task 12).

- [ ] **Step 3: Commit**

```bash
cd /Users/a74739/code/OpenPARL
git add src/openparl/widesearch/
git commit -m "feat: vendor widesearch agent code (reward, prompts, tools, prepare-data)"
```

### Task 12: Vendor RAG server with CREDITS

**Files:**
- Create: `third_party/rag_server/*.py`
- Create: `third_party/rag_server/CREDITS.md`

- [ ] **Step 1: Copy RAG server files**

```bash
SRC=/Users/a74739/code/miles/examples/parl_v2/widesearch/rag_server
DST=/Users/a74739/code/OpenPARL/third_party/rag_server

cp "$SRC/build_index.py" "$DST/build_index.py"
cp "$SRC/local_retrieval_server.py" "$DST/local_retrieval_server.py"
cp "$SRC/qdrant_encoder.py" "$DST/qdrant_encoder.py"
```

- [ ] **Step 2: Write CREDITS.md**

File: `/Users/a74739/code/OpenPARL/third_party/rag_server/CREDITS.md`
```markdown
# RAG server credits

The code in this directory was vendored from the RLinf project
(<https://github.com/RLinf/RLinf>) to avoid an external runtime dependency
during OpenPARL's WideSearch training. Minor changes to path handling and
import style were applied on top; core retrieval / encoder logic is
unchanged.

Upstream source commit: (to be filled in — run `git log --all --author= --oneline
-- examples/parl_v2/widesearch/rag_server/` in the miles checkout and record the
commit hash that introduced these files, which was `5788fa971` in
`dev/guanxing`).

RLinf is distributed under its own license — users of OpenPARL must ensure
their use of this RAG server complies with RLinf's license, which is
reproduced below:

<!-- PASTE RLinf LICENSE BODY HERE -->
```

> **Author action required:** fetch the RLinf license text and paste it into the placeholder before pushing. If uncertain about compatibility, replace `third_party/rag_server/` with a `scripts/fetch_rag_server.sh` that clones RLinf at install time instead of vendoring.

- [ ] **Step 3: Commit**

```bash
cd /Users/a74739/code/OpenPARL
git add third_party/
git commit -m "feat: vendor RAG server from RLinf with CREDITS attribution"
```

### Task 13: Copy configs + launchers with renames

**Files:**
- Create: `configs/sglang_4B.yaml`, `configs/sglang_0.6B.yaml`
- Create: `scripts/run-qwen3-4B-parl.sh`, `scripts/run-qwen3-4B-single.sh`, `scripts/run-qwen3-4B-orchestrator_only.sh`, `scripts/run-qwen3-0.6B-parl.sh`, `scripts/launch_rag_server.sh`

- [ ] **Step 1: Copy two sglang configs (drop the 30B pair)**

```bash
SRC_P=/Users/a74739/code/miles/examples/parl_v2
DST=/Users/a74739/code/OpenPARL/configs

cp "$SRC_P/sglang_config_4B.yaml" "$DST/sglang_4B.yaml"
cp "$SRC_P/sglang_config_0.6B.yaml" "$DST/sglang_0.6B.yaml"
```

- [ ] **Step 2: Copy + rename launchers (skip `-paper` → `orchestrator_only`)**

```bash
SRC_W=/Users/a74739/code/miles/examples/parl_v2/widesearch
DST=/Users/a74739/code/OpenPARL/scripts

cp "$SRC_W/run-qwen3-4B-widesearch.sh"         "$DST/run-qwen3-4B-parl.sh"
cp "$SRC_W/run-qwen3-4B-widesearch-single.sh"  "$DST/run-qwen3-4B-single.sh"
cp "$SRC_W/run-qwen3-4B-widesearch-paper.sh"   "$DST/run-qwen3-4B-orchestrator_only.sh"
cp "$SRC_W/run-qwen3-0.6B-widesearch.sh"       "$DST/run-qwen3-0.6B-parl.sh"
cp "$SRC_W/launch_rag_server.sh"               "$DST/launch_rag_server.sh"
chmod +x "$DST"/*.sh
```

- [ ] **Step 3: Verify none of the copied launchers references `robbyctl_remote/`**

```bash
grep -R "robbyctl\|job_config_robbys3\|robbys3" /Users/a74739/code/OpenPARL/scripts/ || echo "clean"
```

Expected: `clean`. If any match, clean it up by hand (e.g. drop the `robbyctl launch` line from that launcher).

- [ ] **Step 4: Commit**

```bash
cd /Users/a74739/code/OpenPARL
git add configs/ scripts/
git commit -m "feat: vendor sglang configs + rename launchers (parl / single / orchestrator_only)"
```

### Task 14: Copy tests

**Files:**
- Create: `tests/test_reward.py`, `tests/test_reward_utils.py`

- [ ] **Step 1: Copy**

```bash
SRC=/Users/a74739/code/miles/tests/fast/examples/parl_v2/widesearch
DST=/Users/a74739/code/OpenPARL/tests

cp "$SRC/test_reward.py" "$DST/test_reward.py"
cp "$SRC/test_reward_utils.py" "$DST/test_reward_utils.py"
```

- [ ] **Step 2: Commit (imports will still point at `examples.parl_v2.*` — fixed in Task 15)**

```bash
cd /Users/a74739/code/OpenPARL
git add tests/
git commit -m "chore: vendor widesearch reward tests (imports fixed in next commit)"
```

### Task 15: Rewrite `examples.parl_v2` imports → `openparl`

**Files touched:** all `.py` files under `src/` + `tests/`; launchers under `scripts/`; yaml under `configs/`.

This task is one sed pass + careful spot-check. `examples.parl_v2.widesearch.*` maps to `openparl.widesearch.*`; everything else under `examples.parl_v2.*` maps to `openparl.*`. Also the top-level module rename `run_parl_v2` → `run`.

- [ ] **Step 1: Preview candidate matches**

```bash
cd /Users/a74739/code/OpenPARL
grep -rn 'examples\.parl_v2\|run_parl_v2' src/ tests/ scripts/ configs/ || echo "none"
```

Expected: list of ~30 hits (in `generate.py`, `run.py`, launchers, yaml, tests).

- [ ] **Step 2: Run the rewrite (mac BSD-sed needs `-i ''`)**

```bash
cd /Users/a74739/code/OpenPARL

# Widesearch-qualified first so the bare form doesn't shadow.
find src tests scripts configs -type f \( -name '*.py' -o -name '*.sh' -o -name '*.yaml' \) -print0 \
  | xargs -0 sed -i '' -e 's|examples\.parl_v2\.widesearch|openparl.widesearch|g'

# Then the bare form.
find src tests scripts configs -type f \( -name '*.py' -o -name '*.sh' -o -name '*.yaml' \) -print0 \
  | xargs -0 sed -i '' -e 's|examples\.parl_v2|openparl|g'

# Rename the run_parl_v2 module reference.
find src tests scripts configs -type f \( -name '*.py' -o -name '*.sh' -o -name '*.yaml' \) -print0 \
  | xargs -0 sed -i '' -e 's|openparl\.run_parl_v2|openparl.run|g'

# Cosmetic: drop stale "python -m examples.parl_v2.*" remnants (should be gone; this is a sanity sweep).
grep -rn 'examples\.parl_v2\|run_parl_v2' src/ tests/ scripts/ configs/ || echo "all clean"
```

Expected tail: `all clean`.

- [ ] **Step 3: Spot-check `src/openparl/run.py` — the launcher has hardcoded `/workspace/miles` paths that may need adjusting**

```bash
grep -n 'workspace/miles\|/root/miles' /Users/a74739/code/OpenPARL/src/openparl/run.py
```

Expected: hits on lines around 28 (`DEFAULT_DEV_REPO_DIR`), 411 (`examples.parl_v2.*` comment) and possibly others. Replace with an env var default:

```bash
# Manual edit (no sed — this is one line in a comment + one default value):
# 1. Change `DEFAULT_DEV_REPO_DIR = "/workspace/miles"` to
#    `DEFAULT_DEV_REPO_DIR = os.environ.get("OPENPARL_DEV_REPO_DIR", os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))`.
# 2. Scrub the 2 comment lines that still say `examples.parl_v2.*`.
```

Alternative (simpler): leave `DEFAULT_DEV_REPO_DIR` as-is and document in `docs/reproducibility.md` that users must export `OPENPARL_DEV_REPO_DIR`. Keep the launcher behavior unchanged; this is a doc fix, not a code fix.

- [ ] **Step 4: Spot-check launchers still source from `REPO_DIR=...`**

```bash
grep -n 'REPO_DIR\|DEV_REPO_DIR' /Users/a74739/code/OpenPARL/scripts/run-qwen3-4B-parl.sh
```

Expected: launcher sets `REPO_DIR` relative to its own location (`cd "$(dirname "$0")/.."`). If it hardcodes an internal path, edit to use the script-relative form.

- [ ] **Step 5: Commit**

```bash
cd /Users/a74739/code/OpenPARL
git add -A
git commit -m "refactor: rewrite examples.parl_v2 imports to openparl"
```

### Task 16: Write `pyproject.toml`

**Files:** Create `/Users/a74739/code/OpenPARL/pyproject.toml`

- [ ] **Step 1: Write file**

```toml
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "openparl"
version = "0.1.0"
description = "Reproduction of Kimi K2.5 PARL Agent Swarm on WideSearch"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "Apache-2.0" }
authors = [{ name = "Guanxing Lu" }]
dependencies = [
    # miles is installed separately by install.sh (git+https + tag).
    "typer",
    "numpy",
    "pytest",
]

[project.urls]
Homepage = "https://github.com/GuanxingLu/OpenPARL"
Issues = "https://github.com/GuanxingLu/OpenPARL/issues"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 2: Commit**

```bash
cd /Users/a74739/code/OpenPARL
git add pyproject.toml
git commit -m "feat: add pyproject.toml (src-layout, Python 3.10+)"
```

### Task 17: Write `install.sh`

**Files:** Create `/Users/a74739/code/OpenPARL/install.sh`

- [ ] **Step 1: Write file**

```bash
#!/usr/bin/env bash
# Install OpenPARL + its pinned miles fork.
# Requirements: Python 3.10+, pip, git.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/2] Installing miles@v0.1-openparl (PARL framework hooks on radixark/miles@5d11fe2f0)..."
pip install "git+https://github.com/GuanxingLu/miles.git@v0.1-openparl"

echo "[2/2] Installing OpenPARL in editable mode..."
pip install -e "${HERE}"

cat <<'EOF'

OpenPARL installed. Next steps:
  1. (Optional) Launch the local RAG server:
       bash scripts/launch_rag_server.sh
  2. Run the WideSearch training:
       bash scripts/run-qwen3-4B-parl.sh
  See docs/reproducibility.md for hardware / environment details.
EOF
```

- [ ] **Step 2: Make executable + commit**

```bash
cd /Users/a74739/code/OpenPARL
chmod +x install.sh
git add install.sh
git commit -m "feat: add install.sh (miles@v0.1-openparl + editable install)"
```

### Task 18: Write `LICENSE` + `NOTICE`

**Files:** `LICENSE`, `NOTICE`

- [ ] **Step 1: Write `LICENSE` — Apache-2.0 full text**

Copy from miles: `cp /Users/a74739/code/miles/LICENSE /Users/a74739/code/OpenPARL/LICENSE` — verify it IS Apache-2.0 first:

```bash
head -5 /Users/a74739/code/miles/LICENSE
```

Expected: `Apache License / Version 2.0, January 2004`. If that's the case, copy it. If miles has a different LICENSE, fetch a canonical Apache-2.0 body from <https://www.apache.org/licenses/LICENSE-2.0.txt>.

- [ ] **Step 2: Write `NOTICE`**

File: `/Users/a74739/code/OpenPARL/NOTICE`
```text
OpenPARL
Copyright 2026 Guanxing Lu

This product is a research reproduction of Kimi K2.5 Agent Swarm (PARL),
arXiv:2602.02276 (Kimi Team, 2026). It is not an official Kimi or Moonshot
product, and carries no endorsement from the paper authors.

OpenPARL incorporates code from:

  * miles (Apache License 2.0)
    https://github.com/radixark/miles
    Used as the training framework. OpenPARL's installer pulls a specific
    fork with PARL-specific hooks: https://github.com/GuanxingLu/miles/tree/openparl-v1

  * slime (Apache License 2.0)
    https://github.com/THUDM/slime
    Upstream project from which miles was forked.

  * RLinf (see third_party/rag_server/CREDITS.md for license)
    https://github.com/RLinf/RLinf
    Vendored RAG server (local retrieval + qdrant encoder + index builder).
```

- [ ] **Step 3: Commit**

```bash
cd /Users/a74739/code/OpenPARL
git add LICENSE NOTICE
git commit -m "chore: add LICENSE (Apache-2.0) and NOTICE (attribution)"
```

### Task 19: Write `README.md`

**Files:** `README.md`

- [ ] **Step 1: Write file**

File: `/Users/a74739/code/OpenPARL/README.md`
````markdown
# OpenPARL

**Research reproduction of Kimi K2.5 Agent Swarm (PARL) on WideSearch.**
arXiv:2602.02276 — "Kimi K2.5: Visual Agentic Intelligence" (Kimi Team, 2026).

> ⚠️ OpenPARL is an independent reproduction, **not** an official Kimi / Moonshot
> product. No endorsement by the paper authors is implied.

## What it does

Trains a **Qwen3-4B Orchestrator** with RL on the WideSearch benchmark while a
**Qwen3-0.6B Subagent** stays frozen. The Orchestrator uses `create_subagent` +
`assign_task` tools to dispatch parallel sub-queries; per-token credit assignment
routes advantages only to Orchestrator tokens.

Three launchers compare three agent configurations:

| Launcher | Agent mode | Purpose |
|---|---|---|
| `scripts/run-qwen3-4B-parl.sh` | Trainable Orch + frozen Subagent | Headline result |
| `scripts/run-qwen3-4B-single.sh` | Single agent (no subagents) | Baseline |
| `scripts/run-qwen3-4B-orchestrator_only.sh` | Orchestrator w/o subagents, paper prompt | Ablation |

## Install

```bash
git clone https://github.com/GuanxingLu/OpenPARL.git
cd OpenPARL
./install.sh
```

`install.sh` pulls a pinned miles fork (`GuanxingLu/miles@v0.1-openparl`,
~193 LOC of PARL hooks on top of `radixark/miles@5d11fe2f0`) and installs
OpenPARL in editable mode.

## Reproduce

```bash
# 1. Launch local RAG server on :8000
bash scripts/launch_rag_server.sh

# 2. Run the headline launcher
bash scripts/run-qwen3-4B-parl.sh
```

See [`docs/reproducibility.md`](docs/reproducibility.md) for hardware,
seeds, wall-clock, and expected numbers.

## Results

*(Populate from wandb runs at blog-writing time.)*

| Config | item-F1 | row-F1 | is_success | Avg@N | Max@N | Pass@N |
|---|---|---|---|---|---|---|
| Single | — | — | — | — | — | — |
| Orch-only | — | — | — | — | — | — |
| PARL | — | — | — | — | — | — |

## Repository map

```
src/openparl/          agent code (prompts, generate, rollout_log, run, tool)
  widesearch/          widesearch-specific (reward, prompts, tools, prepare-data)
third_party/rag_server/ RAG server vendored from RLinf
configs/               sglang configs (4B + 0.6B)
scripts/               launchers (.sh)
tests/                 CPU unit tests
docs/                  architecture / reward / reproducibility
```

## Framework hooks

The PARL training recipe needs ~193 LOC of hooks in miles. They ship as 6
commits on [`GuanxingLu/miles@openparl-v1`](https://github.com/GuanxingLu/miles/tree/openparl-v1):

| Hook | What it enables |
|---|---|
| per-token advantages | turn-level credit assignment for Orchestrator tokens only |
| critical-step + token-level clip | K2.5 policy loss (grad-mask out of `[α,β]`) |
| `--disable-entropy-computation` | fits 4B+0.6B colocate into 1×H200 |
| frozen-subagent weight sync | keeps Subagent weights across Actor update cycles |
| multi-agent metrics | false-tool-call + critical-step + per-subagent rewards |
| Orch→Subagent inference hooks | user-space `assign_task` routing |

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Cite

If you use OpenPARL, please cite the Kimi K2.5 paper and this repository.
````

- [ ] **Step 2: Commit**

```bash
cd /Users/a74739/code/OpenPARL
git add README.md
git commit -m "docs: add README (install, reproduce, hooks)"
```

### Task 20: Write `docs/architecture.md`, `docs/reward.md`, `docs/reproducibility.md`

**Files:** three markdown files under `docs/`.

- [ ] **Step 1: Write `docs/architecture.md`**

File: `/Users/a74739/code/OpenPARL/docs/architecture.md`
```markdown
# Architecture

## Decoupled Orchestrator + frozen Subagent

PARL separates roles:

* **Orchestrator** (trainable) — receives the user query, decides which
  sub-queries to spawn in parallel, aggregates results, emits the final
  answer. Gets RL gradient on its tokens.
* **Subagent** (frozen) — instantiated from a fixed checkpoint; executes
  one assigned sub-query at a time. Its tokens are **environmental
  observations** to the Orchestrator, not gradient-bearing.

Rationale: joint optimization over Orchestrator + Subagent has sparse +
ambiguous credit assignment ("correct final answer ≠ flawless subagent
execution"); freezing Subagent collapses the problem to single-agent RL on
the Orchestrator, conditioned on a stable tool-use environment.

## SGLang deployment (colocate)

OpenPARL runs two SGLang engines in colocate mode:

1. Orchestrator engine — weights track the training actor (updated each
   `update_weights` cycle).
2. Subagent engine — weights pinned at the 0.6B checkpoint, never touched.

The frozen-subagent-weight-sync hook in
[`miles/ray/rollout.py`](https://github.com/GuanxingLu/miles/blob/openparl-v1/miles/ray/rollout.py)
is what keeps the Subagent weights alive across Actor pushes.

## Tools

The Orchestrator is wired to three tool classes:

1. Search (`search`) — backed by the local RAG server in `third_party/rag_server/`
2. `create_subagent(name, system_prompt)` — registers a named subagent configuration
3. `assign_task(agent, prompt)` — dispatches a task to one registered subagent

`assign_task` supports concurrent dispatch — the Orchestrator emits multiple
tool calls in one turn, and OpenPARL's rollout driver awaits them in parallel.

## Critical steps

Episode length is bounded by **critical steps**, not total tool calls:

```
CriticalSteps = Σ_t ( S_main^(t) + max_i S_sub,i^(t) )
```

This directly rewards splitting work across parallel subagents, because
spawning N parallel subagents costs `max_i steps`, not `Σ_i steps`. See
[`docs/reward.md`](reward.md) for how this interacts with the reward.
```

- [ ] **Step 2: Write `docs/reward.md`**

File: `/Users/a74739/code/OpenPARL/docs/reward.md`
````markdown
# Reward

```
score = r_perf  +  λ₁ · r_parallel  +  λ₂ · r_finish
```

## `r_perf` — task-level outcome

Rule-based, varies by eval set:

* WideSearch / WideSeek-R1 train: `item_f1` over required columns × rows,
  with `unique_columns` row-key alignment, URL + multi-value cell
  canonicalization.
* ASearcher QA (HotpotQA, 2Wiki, Bamboogle): normalized exact-match on the
  boxed answer.

## `r_finish` — sub-agent completion

Fraction of `assign_task` calls that returned a parseable result. Discourages
**spurious parallelism** (spawning many nonsense subagents to inflate the
parallel term).

## `r_parallel` (default OFF)

Discourages **serial collapse** (Orchestrator using only 1 subagent).
Paper omits the explicit formula; OpenPARL's default is `LAMBDA1_INIT = 0.3,
LAMBDA2_INIT = 0.2` with `ANNEAL_FRAC = 100.0` (effectively no anneal —
flip this when `r_perf` stops being sparse).

> Critical-step budget implicitly rewards parallelism: at fixed critical
> budget, more parallel subagents = more total work done = higher `r_perf`.
> So `r_parallel` is often redundant and can be disabled.

## Annealing

`λ₁` and `λ₂` anneal to 0 over training so the final policy optimizes
`r_perf` alone. See `src/openparl/widesearch/reward.py` for the exact
schedule knobs.
````

- [ ] **Step 3: Write `docs/reproducibility.md`**

File: `/Users/a74739/code/OpenPARL/docs/reproducibility.md`
```markdown
# Reproducibility

## Hardware

* 1 × H200 (80 GB, SXM) node — 8 GPUs
* CUDA 12.x, driver 550+
* ~200 GB disk for MODEL / DATA roots

## Software

Pinned via `install.sh`:

* miles @ `v0.1-openparl` (= `radixark/miles@5d11fe2f0` + 6 PARL hook commits)
* SGLang, Megatron-LM — transitive deps of miles
* Python 3.10+

## Environment variables

Launchers read these (defaults in each script; export to override):

* `DATA_ROOT` — dataset root (wideseek-r1-train, widesearch-test, asearcher-test)
* `MODEL_ROOT` — HF checkpoint root (Qwen3-4B, Qwen3-0.6B)
* `REPO_DIR` — this checkout (auto-derived from `$(dirname $0)`)
* `WANDB_API_KEY`, `WANDB_BASE_URL` — tracking
* `HF_HOME` — HuggingFace cache

## Seeds

All launchers pin seeds; record the three run seeds used for the blog
numbers here: **_TBD_** (author to fill).

## Wall-clock

* WideSearch PARL 4B: **_N hours for M steps_** on 1×H200 node (author to fill).
* Single-agent baseline: **_N'_** (author to fill).

## Data

* `wideseek-r1-train/hybrid_20k.miles.jsonl` — training data (prepared via
  `src/openparl/widesearch/prepare_data.py`)
* `widesearch-test/test.miles.jsonl` — WideSearch eval
* `asearcher-test/{HotpotQA_rand1000, 2WikiMultihopQA_rand1000, Bamboogle}/test.miles.jsonl` — QA evals

## Known issues

* Colocate actor size must be ≤ rollout size (see frozen-subagent weight sync hook).
* OOM at `max-tokens-per-gpu >= 32768`; 20480 is a safe default for 4B + 0.6B.
```

- [ ] **Step 4: Commit**

```bash
cd /Users/a74739/code/OpenPARL
git add docs/
git commit -m "docs: add architecture / reward / reproducibility guides"
```

### Task 21: Write `BLOG.md` skeleton

**Files:** `BLOG.md`

- [ ] **Step 1: Write skeleton with section headers only**

File: `/Users/a74739/code/OpenPARL/BLOG.md`
```markdown
# Reproducing Kimi K2.5's Agent Swarm with Qwen3-4B

> Draft skeleton — author fills from wandb at blog-writing time. Target
> length: ~2500–3500 words. Publish on X with link back to
> https://github.com/GuanxingLu/OpenPARL.

## 1. Why PARL?

_TODO: sequential-agent latency vs. parallel decomposition. K2.5's bet._

## 2. Architecture primer

_TODO: decoupled trainable Orchestrator + frozen Subagent, credit-assignment
rationale. Embed `docs/architecture.md` figure._

## 3. Reproducing on Qwen3-4B + Qwen3-0.6B-frozen

_TODO: scale choices, hardware (1×H200), cost estimate, training wall-clock._

## 4. The 193 LOC of framework hooks

_TODO: walk through the 6 commits on `GuanxingLu/miles@openparl-v1`.
Per-token advantages, critical-step loss, entropy flag, frozen-subagent
weight sync, multi-agent metrics, inference hooks. Link to commit URLs._

## 5. Observations

_TODO (author fills from wandb):_
- 5.1 Training dynamics (reward curve, critical-step curve, avg parallelism — does it match paper Fig 4?)
- 5.2 Emergent subagent specialization (cluster `create_subagent` calls by system prompt — do we see `Biography Researcher` / `Verification Specialist` etc.?)
- 5.3 Single vs. swarm vs. orchestrator_only on WideSearch item-F1
- 5.4 Serial-collapse ablation — what happens without the critical-step budget?
- 5.5 False-tool-call rate over training

## 6. What I'd change next

_TODO: curriculum (small → large subagent), r_parallel ablation, BrowseComp, In-house Swarm Bench._

## 7. Code + setup

_TODO: link to `GuanxingLu/OpenPARL`, bibtex, thanks._
```

- [ ] **Step 2: Commit**

```bash
cd /Users/a74739/code/OpenPARL
git add BLOG.md
git commit -m "docs: add BLOG.md skeleton (author fills observations)"
```

---

## Phase C — Verification + publish (Tasks 22–25)

### Task 22: Install OpenPARL in a fresh venv and run tests

- [ ] **Step 1: Create a fresh venv**

```bash
python3 -m venv /tmp/openparl-venv
source /tmp/openparl-venv/bin/activate
pip install --upgrade pip
```

- [ ] **Step 2: Run install.sh**

```bash
cd /Users/a74739/code/OpenPARL
bash install.sh
```

Expected: miles installs from `git+https://...miles.git@v0.1-openparl`, then `pip install -e .` succeeds. No import errors.

> **Gate:** if `miles@v0.1-openparl` is not yet pushed (Task 8 was skipped or failed), this step will fail on the first line. Do Task 8 first.

- [ ] **Step 3: Run OpenPARL tests**

```bash
cd /Users/a74739/code/OpenPARL
pytest tests/ -v
```

Expected: all PASS. The two test files (`test_reward.py`, `test_reward_utils.py`) have ~85 + ~443 LOC of assertions from the source repo; all should pass after the import rewrite in Task 15.

- [ ] **Step 4: Sanity-check launcher dry-run**

```bash
python -c "from openparl import run; print(run.__doc__[:200])"
python -c "from openparl.widesearch import reward, reward_utils; print('ok')"
python -c "from openparl.widesearch.orchestrator_tools import dispatch; print(dispatch.__doc__[:120] if dispatch.__doc__ else 'ok')"
```

Expected: all three print without ImportError.

- [ ] **Step 5: If any test or import fails — iterate**

Common issues:
- Missing `__init__.py` files → add.
- A path that wasn't rewritten by sed (e.g. `"examples.parl_v2"` embedded in a triple-quoted docstring example) → hand-edit.
- Paths in docstrings that still reference `examples.parl_v2.*` are acceptable (reader-facing comments) but also fine to rewrite for consistency.

Iterate, then commit the fixes:

```bash
git add -A
git commit -m "fix: iron out import / packaging issues from smoke test"
```

- [ ] **Step 6: Deactivate venv**

```bash
deactivate
rm -rf /tmp/openparl-venv
```

### Task 23: CI workflow

**Files:** `.github/workflows/fast-tests.yml`

- [ ] **Step 1: Write workflow**

File: `/Users/a74739/code/OpenPARL/.github/workflows/fast-tests.yml`
```yaml
name: fast-tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Install
        run: |
          pip install --upgrade pip
          # Do NOT pip-install miles here — the reward tests don't import it.
          # pyproject dependencies (numpy, typer, pytest) are enough.
          pip install -e .
      - name: Run fast tests
        run: pytest tests/ -v
```

- [ ] **Step 2: Verify `tests/` doesn't transitively import miles**

```bash
grep -R "^from miles\|^import miles" /Users/a74739/code/OpenPARL/tests/ || echo "tests are miles-free"
```

Expected: `tests are miles-free`. If miles imports leak in (e.g. via a helper in `reward_utils.py`), the CI install needs to pull miles — update the workflow's install step to run `bash install.sh` instead of `pip install -e .`.

- [ ] **Step 3: Commit**

```bash
cd /Users/a74739/code/OpenPARL
git add .github/
git commit -m "ci: add GitHub Actions fast-tests workflow"
```

### Task 24: Create GitHub repo + push

- [ ] **Step 1: Create remote repo**

```bash
gh repo create GuanxingLu/OpenPARL --public --description "Reproduction of Kimi K2.5 PARL Agent Swarm on WideSearch" --source=/Users/a74739/code/OpenPARL --remote=origin
```

- [ ] **Step 2: Push**

```bash
cd /Users/a74739/code/OpenPARL
git branch -M main
git push -u origin main
```

- [ ] **Step 3: Verify the repo on github.com**

Open <https://github.com/GuanxingLu/OpenPARL> and confirm:
- README renders
- `src/openparl/`, `scripts/`, `docs/` visible
- CI workflow triggers (Actions tab, `fast-tests` run starts)

- [ ] **Step 4: STOP — wait for CI to go green before announcing**

If CI fails, iterate on `tests/` or workflow and push fixes. Do not publish the X blog until CI passes on `main`.

### Task 25: Final release checks

- [ ] **Step 1: Verify both repos render compare views correctly**

- `https://github.com/GuanxingLu/miles/compare/5d11fe2f0...openparl-v1` — should show 6 clean commits, ~193/+ −27 lines across 12 files.
- `https://github.com/GuanxingLu/OpenPARL` — README renders, install.sh visible, tests pass.

- [ ] **Step 2: Announce readiness**

Post in this channel:
- Miles fork: `https://github.com/GuanxingLu/miles/tree/openparl-v1` (tag `v0.1-openparl`).
- OpenPARL: `https://github.com/GuanxingLu/OpenPARL`.
- Remaining author work before publishing X post: (a) fill `BLOG.md` §5 from wandb, (b) fill `docs/reproducibility.md` seeds + wall-clock, (c) generate architecture / training-curve figures.

---

## Self-review checklist (run before handoff)

- [ ] Each spec requirement has a matching task (spec §"Resolved decisions" 1–5 → Tasks 12, 13/19, 13, 21, 1)
- [ ] No "TBD" / "TODO" in plan itself (only in the BLOG.md / reproducibility.md skeletons, which is by design — author fills those)
- [ ] Type consistency: `openparl.widesearch.*` used everywhere post-rewrite; launcher names `parl` / `single` / `orchestrator_only` used consistently
- [ ] File paths match the spec's `File Structure` section
- [ ] Commit titles match the spec's 6-commit schema
- [ ] Base SHA (`5d11fe2f0`) matches the spec

