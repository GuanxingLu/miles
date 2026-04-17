# PARL v2 Frozen Subagent Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Spin up a separate, frozen SGLang engine pool for `assign_task` subagent inference in `examples/parl_math_v2/`, so subagent runs on hard-frozen SFT weights while only the orchestrator (main agent) is updated by RL — aligning with K2.5 PARL Agent Swarm.

**Architecture:** Use miles' existing multi-model `--sglang-config` infrastructure to declare two ServerGroups under one router cluster: `actor` (live, `update_weights: true`) and `subagent` (frozen, `update_weights: false`). Both colocate with the training actor on the same GPU pool — total GPU count unchanged. `generate.py` calls `miles.rollout.sglang_rollout.get_model_url(args, "subagent")` to route subagent requests; this auto-falls-back to the live router when the yaml is absent (ablation mode). Zero changes to miles core.

**Tech Stack:** Python 3.12, miles (Megatron + SGLang), Ray, YAML, bash. Test execution is end-to-end smoke (no fast-test surface for parl_math_v2).

**Reference spec:** `docs/superpowers/specs/2026-04-17-parl-v2-frozen-subagent-engine-design.md`

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `examples/parl_math_v2/sglang_config_4B.yaml` | Declare 4B colocate-split topology (actor=6, subagent=2, TP=2) | **create** |
| `examples/parl_math_v2/sglang_config_0.6B.yaml` | Declare 0.6B colocate-split topology (actor=3, subagent=1, TP=1) | **create** |
| `examples/parl_math_v2/tool.py` | Take `router_url` as kwarg; drop env-based `_router_url` helper | **modify** |
| `examples/parl_math_v2/generate.py` | Compute subagent URL via `get_model_url`; thread via closure; first-call log | **modify** |
| `examples/parl_math_v2/rollout_log.py` | Add `parl/subagent_mode` + `parl/subagent_endpoint_distinct` W&B summary keys | **modify** |
| `examples/parl_math_v2/run-qwen3-4B-parl-v2.sh` | Add `SUBAGENT_MODE` env switch (default `frozen`) | **modify** |
| `examples/parl_math_v2/run-qwen3-0.6B-parl-v2.sh` | Add `SUBAGENT_MODE` env switch (default `frozen`) | **modify** |
| `examples/parl_math_v2/run_parl_math.py` | (no change) | — |

Splitting rationale: each file's responsibility stays narrow. yaml = topology declaration. `tool.py` = stateless tool implementations. `generate.py` = rollout orchestration glue. `rollout_log.py` = metric emission. Launch scripts = run-time configuration. None of these grow uncomfortably.

---

## Task 1: Create yaml configs for both model sizes

**Files:**
- Create: `examples/parl_math_v2/sglang_config_4B.yaml`
- Create: `examples/parl_math_v2/sglang_config_0.6B.yaml`

This is a pure-config change. Verification is "yaml parses + miles' `from_yaml` accepts it"; deferred to the smoke test in Task 6.

- [ ] **Step 1: Create 4B yaml**

Write `examples/parl_math_v2/sglang_config_4B.yaml`:

```yaml
# PARL v2 frozen-subagent topology for Qwen3-4B (8 GPUs, TP=2).
#
# Splits the colocate rollout pool into two SGLang models sharing the
# same router cluster:
#   - actor:    live policy, receives RL weight updates
#   - subagent: frozen at args.hf_checkpoint (true hard-frozen),
#               called by examples.parl_math_v2.tool.assign_task
#
# Total num_gpus across both groups MUST equal --rollout-num-gpus
# (= --actor-num-gpus under --colocate). Each group's num_gpus must
# be divisible by --rollout-num-gpus-per-engine (=2 for 4B).
sglang:
  - name: actor
    update_weights: true
    server_groups:
      - worker_type: regular
        num_gpus: 6        # 3 engines × TP=2
  - name: subagent
    # model_path empty -> falls back to args.hf_checkpoint (hard-frozen).
    # For heterogeneous-size ablation, set an explicit path here
    # (e.g. /workspace/miles/MODEL/Qwen3-0.6B).
    update_weights: false
    server_groups:
      - worker_type: regular
        num_gpus: 2        # 1 engine × TP=2
```

- [ ] **Step 2: Create 0.6B yaml**

Write `examples/parl_math_v2/sglang_config_0.6B.yaml`:

```yaml
# PARL v2 frozen-subagent topology for Qwen3-0.6B (4 GPUs, TP=1).
# See sglang_config_4B.yaml for design notes.
sglang:
  - name: actor
    update_weights: true
    server_groups:
      - worker_type: regular
        num_gpus: 3        # 3 engines × TP=1
  - name: subagent
    update_weights: false
    server_groups:
      - worker_type: regular
        num_gpus: 1        # 1 engine × TP=1
```

- [ ] **Step 3: Verify yamls parse via miles `SglangConfig.from_yaml`**

Run:

```bash
cd /ssd0/guanxing/miles && python3 -c "
from miles.backends.sglang_utils.sglang_config import SglangConfig
for path in ['examples/parl_math_v2/sglang_config_4B.yaml',
             'examples/parl_math_v2/sglang_config_0.6B.yaml']:
    cfg = SglangConfig.from_yaml(path)
    names = [m.name for m in cfg.models]
    update_flags = {m.name: m.update_weights for m in cfg.models}
    totals = {m.name: m.total_num_gpus for m in cfg.models}
    print(f'{path}: models={names} update_flags={update_flags} totals={totals}')
"
```

Expected output (exact totals):
```
examples/parl_math_v2/sglang_config_4B.yaml: models=['actor', 'subagent'] update_flags={'actor': None, 'subagent': False} totals={'actor': 6, 'subagent': 2}
examples/parl_math_v2/sglang_config_0.6B.yaml: models=['actor', 'subagent'] update_flags={'actor': None, 'subagent': False} totals={'actor': 3, 'subagent': 1}
```

Note: `actor.update_weights` shows `None` because `from_yaml` doesn't run `resolve()` (which fills it in as `True` based on hf_checkpoint at training time). That's expected — `resolve()` is invoked later inside `_resolve_sglang_config`.

- [ ] **Step 4: Commit**

```bash
git add examples/parl_math_v2/sglang_config_4B.yaml \
        examples/parl_math_v2/sglang_config_0.6B.yaml
git commit -m "$(cat <<'EOF'
[parl] add frozen-subagent sglang configs for 4B and 0.6B

Two-model colocate split: actor (live, RL-updated) + subagent (frozen at
hf_checkpoint). Used by run-qwen3-{4B,0.6B}-parl-v2.sh in default
SUBAGENT_MODE=frozen path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Refactor `tool.py` to take `router_url` as kwarg

**Files:**
- Modify: `examples/parl_math_v2/tool.py`

Goal: drop the env-based `_router_url` helper; `_assign_task_call` now takes `router_url` as a required kwarg, injected by `generate.py`'s closure binding. Keeps `tool.py` stateless w.r.t. routing.

- [ ] **Step 1: Read current state of tool.py**

Read `examples/parl_math_v2/tool.py` to confirm current line numbers. Specifically locate:
- import of `os` (line 20)
- `_router_url()` function (lines 86–94)
- `_assign_task_call(...)` signature (line 120) and call site of `_router_url()` (line 147)

- [ ] **Step 2: Remove `_router_url` and `os` import**

Edit `examples/parl_math_v2/tool.py`:

- Delete the entire `_router_url()` function (lines 86–94 inclusive, including the docstring/blank lines around it).
- Delete `import os` if no other usage remains in the file (verify by `grep "os\." examples/parl_math_v2/tool.py` — there shouldn't be any other uses after removing `_router_url`).

- [ ] **Step 3: Update `_assign_task_call` signature and body**

Replace the function definition. The current signature is:

```python
async def _assign_task_call(params: dict, *, registry: dict[str, str], tokenizer) -> tuple[str, bool]:
```

Change it to:

```python
async def _assign_task_call(
    params: dict, *, registry: dict[str, str], tokenizer, router_url: str
) -> tuple[str, bool]:
```

Inside the function body, replace `output = await post(_router_url(), payload)` with:

```python
            output = await post(router_url, payload)
```

(All other lines inside the function — validation, payload building, semaphore, error handling, `is_valid` computation — remain unchanged.)

- [ ] **Step 4: Update the module docstring**

The existing module docstring says routing is via env vars. Update the relevant paragraph to reflect closure injection. Replace this part of the docstring:

```
Parallelism: the orchestrator is free to emit multiple ``assign_task`` calls
in one turn. ``generate.py``'s custom parallel wrapper runs them via
``asyncio.gather``. Registry state and the tokenizer are injected as
keyword-only args by the wrapper via closure binding — ``tool.py`` itself is
stateless.
```

with:

```
Parallelism: the orchestrator is free to emit multiple ``assign_task`` calls
in one turn. ``generate.py``'s custom parallel wrapper runs them via
``asyncio.gather``. Registry state, tokenizer, and the subagent SGLang
router URL are injected as keyword-only args by the wrapper via closure
binding — ``tool.py`` itself is stateless. The router URL points at the
"subagent" model when --sglang-config declares it (frozen mode); otherwise
it falls back to the live router (ablation / shared mode).
```

- [ ] **Step 5: Verify the file imports cleanly**

```bash
cd /ssd0/guanxing/miles && python3 -c "
from examples.parl_math_v2 import tool
import inspect
sig = inspect.signature(tool._assign_task_call)
print('signature:', sig)
assert 'router_url' in sig.parameters, f'router_url missing: {sig.parameters}'
assert sig.parameters['router_url'].kind.name == 'KEYWORD_ONLY', sig
print('os module imported:', 'os' in dir(tool))
"
```

Expected output:
```
signature: (params: dict, *, registry: dict[str, str], tokenizer, router_url: str) -> tuple[str, bool]
os module imported: False
```

- [ ] **Step 6: Commit**

```bash
git add examples/parl_math_v2/tool.py
git commit -m "$(cat <<'EOF'
[parl] tool: take router_url as kwarg, drop env-based discovery

_assign_task_call now requires router_url, supplied by generate.py's
closure binding. Removes env-var indirection in preparation for routing
subagent calls to a frozen SGLang model.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Wire subagent router URL through `generate.py`

**Files:**
- Modify: `examples/parl_math_v2/generate.py`

Goal: at the top of each `generate()` call, compute the subagent router URL via `miles.rollout.sglang_rollout.get_model_url(args, "subagent")` (auto-fallback to live router built into the helper). Thread it through `_execute_tool_calls_parallel` to `_assign_task_call`. Add a one-time module-level startup log so we can eyeball "frozen vs shared" at run start.

- [ ] **Step 1: Read current state of generate.py**

Read `examples/parl_math_v2/generate.py`. Confirm:
- Imports start at lines 19–40 (look for the existing `from miles.rollout.*` imports)
- `_execute_tool_calls_parallel` definition at line 74, signature includes `*, registry, tokenizer`
- `_assign_task_call` is invoked at line 97 with `registry=registry, tokenizer=tokenizer`
- `_execute_tool_calls_parallel` is called inside `generate()` at line 177

- [ ] **Step 2: Add the import + module-level log cache**

In `examples/parl_math_v2/generate.py`, add to the top-level (after the existing `from miles.rollout.*` imports, right before the constant `MAX_CONCURRENT_ASSIGN = 8`):

```python
import logging

from miles.rollout.sglang_rollout import get_model_url

logger = logging.getLogger(__name__)
_logged_endpoint = False
```

Verify there's not already a `logger` defined; if there is, just reuse it and skip the re-definition.

- [ ] **Step 3: Update `_execute_tool_calls_parallel` signature and call site**

Change the function definition from:

```python
async def _execute_tool_calls_parallel(
    tool_calls,
    *,
    registry: dict[str, str],
    tokenizer,
) -> tuple[list[dict], dict]:
```

to:

```python
async def _execute_tool_calls_parallel(
    tool_calls,
    *,
    registry: dict[str, str],
    tokenizer,
    router_url: str,
) -> tuple[list[dict], dict]:
```

Inside the function, update the inner `run_assign` closure to pass `router_url` to `_assign_task_call`. Change:

```python
    async def run_assign(i: int):
        _, params, _ = normalized[i]
        text, is_valid = await _assign_task_call(params, registry=registry, tokenizer=tokenizer)
        return i, text, is_valid
```

to:

```python
    async def run_assign(i: int):
        _, params, _ = normalized[i]
        text, is_valid = await _assign_task_call(
            params, registry=registry, tokenizer=tokenizer, router_url=router_url
        )
        return i, text, is_valid
```

- [ ] **Step 4: Compute router URL inside `generate()` and pass it down**

Inside `async def generate(input: GenerateFnInput)`, after the existing line:

```python
    tokenizer = input.state.tokenizer
```

(around line 134), insert:

```python
    subagent_router_url = get_model_url(args, "subagent")
    live_router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    global _logged_endpoint
    if not _logged_endpoint:
        mode = "frozen" if subagent_router_url != live_router_url else "shared (ablation)"
        logger.info(f"[parl_v2] subagent mode: {mode}")
        logger.info(f"[parl_v2] subagent router: {subagent_router_url}")
        logger.info(f"[parl_v2] live router:     {live_router_url}")
        _logged_endpoint = True
```

Then change the existing `_execute_tool_calls_parallel` call (around line 177) from:

```python
        tool_messages, stats = await _execute_tool_calls_parallel(
            tool_calls, registry=registry, tokenizer=tokenizer
        )
```

to:

```python
        tool_messages, stats = await _execute_tool_calls_parallel(
            tool_calls,
            registry=registry,
            tokenizer=tokenizer,
            router_url=subagent_router_url,
        )
```

- [ ] **Step 5: Update the module docstring**

The existing top docstring describes what the module does. Add one more bullet to the bullet list. Find the bullet list that ends with `- structured per-turn stats for reward attribution:` and the lines following. After all existing bullets but before the `Design:` line, add:

```
- subagent SGLang router URL discovered via miles.get_model_url("subagent")
  with auto-fallback to the live router when --sglang-config does not
  declare a "subagent" model (= shared/ablation mode)
```

- [ ] **Step 6: Smoke-import and verify signatures**

```bash
cd /ssd0/guanxing/miles && python3 -c "
import inspect
from examples.parl_math_v2 import generate as g
sig_g = inspect.signature(g.generate)
sig_e = inspect.signature(g._execute_tool_calls_parallel)
print('generate:', sig_g)
print('_execute_tool_calls_parallel:', sig_e)
assert 'router_url' in sig_e.parameters, sig_e
print('get_model_url imported:', g.get_model_url is not None)
print('_logged_endpoint initial:', g._logged_endpoint)
"
```

Expected:
```
generate: (input: miles.rollout.base_types.GenerateFnInput) -> miles.rollout.base_types.GenerateFnOutput
_execute_tool_calls_parallel: (tool_calls, *, registry: dict[str, str], tokenizer, router_url: str) -> tuple[list[dict], dict]
get_model_url imported: True
_logged_endpoint initial: False
```

- [ ] **Step 7: Commit**

```bash
git add examples/parl_math_v2/generate.py
git commit -m "$(cat <<'EOF'
[parl] generate: route assign_task to frozen subagent router

Discover subagent SGLang URL via miles.rollout.sglang_rollout.get_model_url,
which falls back to the live router when --sglang-config doesn't declare
a 'subagent' model (= shared/ablation mode). Thread the URL through
_execute_tool_calls_parallel into _assign_task_call.

Adds a one-time INFO log at first generate() call summarizing the mode and
both endpoints — easy way to confirm topology in run output.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add observability metrics in `rollout_log.py`

**Files:**
- Modify: `examples/parl_math_v2/rollout_log.py`

Goal: emit two W&B summary keys (constant per run): `parl/subagent_mode` (string) and `parl/subagent_endpoint_distinct` (0 or 1). Logged every rollout step but they're constants — main consumer is W&B sweep filtering.

- [ ] **Step 1: Read current state of rollout_log.py**

Read `examples/parl_math_v2/rollout_log.py`. Confirm:
- `log_rollout_data(rollout_id, args, samples, ...)` lives at line 166
- The function uses `tracking_utils.log(args, log_dict, step_key="rollout/step")` to emit metrics
- Existing imports include `from miles.utils import tracking_utils`

- [ ] **Step 2: Add `get_model_url` import**

Add an import near the existing miles imports (right after `from miles.utils.metric_utils import compute_rollout_step`):

```python
from miles.rollout.sglang_rollout import get_model_url
```

- [ ] **Step 3: Add the two metrics to `log_rollout_data`**

Inside `log_rollout_data`, after the `log_dict |= _compute_multi_turn_metrics(args, samples)` line (around line 169), insert:

```python
    sub_url = get_model_url(args, "subagent")
    live_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    log_dict["parl/subagent_endpoint_distinct"] = int(sub_url != live_url)
```

(The `parl/subagent_mode` is intentionally **not** logged as a numeric — wandb scalars don't carry strings well as time-series. The boolean `parl/subagent_endpoint_distinct` is the durable signal: 1=frozen, 0=shared. The startup log in `generate.py` covers the human-readable mode label.)

- [ ] **Step 4: Smoke-import and check the function still parses**

```bash
cd /ssd0/guanxing/miles && python3 -c "
import inspect
from examples.parl_math_v2 import rollout_log
src = inspect.getsource(rollout_log.log_rollout_data)
assert 'parl/subagent_endpoint_distinct' in src, 'metric not added'
assert 'get_model_url' in src, 'helper not imported into the function call'
print('OK: parl/subagent_endpoint_distinct emitted in log_rollout_data')
"
```

Expected: `OK: parl/subagent_endpoint_distinct emitted in log_rollout_data`

- [ ] **Step 5: Commit**

```bash
git add examples/parl_math_v2/rollout_log.py
git commit -m "$(cat <<'EOF'
[parl] rollout_log: emit parl/subagent_endpoint_distinct each step

Boolean (0/1): 1 when --sglang-config declares a separate 'subagent'
SGLang model (frozen mode), 0 when subagent calls fall back to the live
router (shared/ablation mode). Lets W&B sweeps filter frozen vs shared
runs without grepping launch logs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update launch scripts to switch yaml on `SUBAGENT_MODE`

**Files:**
- Modify: `examples/parl_math_v2/run-qwen3-4B-parl-v2.sh`
- Modify: `examples/parl_math_v2/run-qwen3-0.6B-parl-v2.sh`

Goal: each script defaults to frozen (passes `--sglang-config <yaml>`), but `SUBAGENT_MODE=shared bash <script>` skips the yaml so miles falls back to single-model single-pool (= current behavior, used for ablation comparison).

- [ ] **Step 1: Patch the 4B script**

In `examples/parl_math_v2/run-qwen3-4B-parl-v2.sh`, find this block at the end:

```bash
python examples/parl_math_v2/run_parl_math.py \
   ${MODEL_ARGS[@]} \
   ${RUN_ARGS[@]} \
   ${PARALLEL_ARGS[@]} \
   ${DATA_ARGS[@]} \
   ${GENERATE_ARGS[@]}
```

Replace it with:

```bash
# SUBAGENT_MODE selects how examples.parl_math_v2.tool.assign_task is routed:
#   frozen (default): --sglang-config carves a separate 'subagent' SGLang
#                     model from the colocate rollout pool, frozen at the
#                     SFT hf_checkpoint and excluded from RL weight updates.
#   shared           : skip --sglang-config; subagent shares the live
#                     policy router (= pre-frozen-engine baseline, used
#                     as ablation control).
SUBAGENT_MODE=${SUBAGENT_MODE:-frozen}
if [ "$SUBAGENT_MODE" = "frozen" ]; then
   SGLANG_EXTRA_ARGS=(--sglang-config examples/parl_math_v2/sglang_config_4B.yaml)
elif [ "$SUBAGENT_MODE" = "shared" ]; then
   SGLANG_EXTRA_ARGS=()
else
   echo "ERROR: SUBAGENT_MODE must be 'frozen' or 'shared', got '$SUBAGENT_MODE'" >&2
   exit 1
fi

python examples/parl_math_v2/run_parl_math.py \
   ${MODEL_ARGS[@]} \
   ${RUN_ARGS[@]} \
   ${PARALLEL_ARGS[@]} \
   ${DATA_ARGS[@]} \
   ${GENERATE_ARGS[@]} \
   "${SGLANG_EXTRA_ARGS[@]}"
```

- [ ] **Step 2: Patch the 0.6B script**

In `examples/parl_math_v2/run-qwen3-0.6B-parl-v2.sh`, find this block at the end:

```bash
python examples/parl_math_v2/run_parl_math.py \
   ${MODEL_ARGS[@]} \
   ${RUN_ARGS[@]} \
   ${PARALLEL_ARGS[@]} \
   ${DATA_ARGS[@]} \
   ${GENERATE_ARGS[@]} \
   --sglang-router-ip ${SGLANG_ROUTER_IP} \
   --sglang-router-port ${SGLANG_ROUTER_PORT}
```

Replace it with:

```bash
# See run-qwen3-4B-parl-v2.sh for SUBAGENT_MODE semantics.
SUBAGENT_MODE=${SUBAGENT_MODE:-frozen}
if [ "$SUBAGENT_MODE" = "frozen" ]; then
   SGLANG_EXTRA_ARGS=(--sglang-config examples/parl_math_v2/sglang_config_0.6B.yaml)
elif [ "$SUBAGENT_MODE" = "shared" ]; then
   SGLANG_EXTRA_ARGS=()
else
   echo "ERROR: SUBAGENT_MODE must be 'frozen' or 'shared', got '$SUBAGENT_MODE'" >&2
   exit 1
fi

python examples/parl_math_v2/run_parl_math.py \
   ${MODEL_ARGS[@]} \
   ${RUN_ARGS[@]} \
   ${PARALLEL_ARGS[@]} \
   ${DATA_ARGS[@]} \
   ${GENERATE_ARGS[@]} \
   --sglang-router-ip ${SGLANG_ROUTER_IP} \
   --sglang-router-port ${SGLANG_ROUTER_PORT} \
   "${SGLANG_EXTRA_ARGS[@]}"
```

- [ ] **Step 3: Bash-syntax-check both scripts**

```bash
bash -n /ssd0/guanxing/miles/examples/parl_math_v2/run-qwen3-4B-parl-v2.sh
bash -n /ssd0/guanxing/miles/examples/parl_math_v2/run-qwen3-0.6B-parl-v2.sh
echo "exit=$?"
```

Expected: no output from `bash -n` and `exit=0`. Any syntax error must be fixed before commit.

- [ ] **Step 4: Commit**

```bash
git add examples/parl_math_v2/run-qwen3-4B-parl-v2.sh \
        examples/parl_math_v2/run-qwen3-0.6B-parl-v2.sh
git commit -m "$(cat <<'EOF'
[parl] launch scripts: SUBAGENT_MODE switch (default frozen)

Both run-qwen3-{4B,0.6B}-parl-v2.sh now default to the frozen-subagent
topology (passing --sglang-config <model_size>.yaml). Setting
SUBAGENT_MODE=shared skips the config so miles falls back to the
single-model single-pool default — used as the ablation baseline.

Invalid mode strings hard-fail rather than silently picking a path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: End-to-end smoke verification

**Files:**
- (no code change; this task validates Tasks 1–5)

Goal: exercise both `frozen` and `shared` modes on the 0.6B debug box and confirm topology + observability are correct. We do NOT verify training convergence in this task — only that the system boots, routes correctly, and the metrics light up as designed.

This task assumes you have access to a 4-GPU box with Qwen3-0.6B already converted (per the existing `run-qwen3-0.6B-parl-v2.sh` requirements — `MODEL/Qwen3-0.6B` and `MODEL/Qwen3-0.6B_torch_dist` should exist under `DEV_REPO_DIR`).

- [ ] **Step 1: Run fast tests to confirm refactor didn't break unrelated paths**

```bash
cd /ssd0/guanxing/miles && pytest tests/fast -x -q
```

Expected: all tests pass. If anything fails, fix the regression before proceeding (the tool/generate refactor must not change behavior under the experimental rollout path).

- [ ] **Step 2: Frozen mode smoke (0.6B, debug_minimal)**

In a fresh terminal on the 4-GPU debug box (where the existing 0.6B run works):

```bash
cd /ssd0/guanxing/miles
MODE=debug_minimal SUBAGENT_MODE=frozen bash examples/parl_math_v2/run-qwen3-0.6B-parl-v2.sh 2>&1 | tee /tmp/parl_v2_frozen_smoke.log
```

Let it run until you see at least 3 successful rollout iterations (look for `rollout_id=2` or higher in the log) — usually 5–10 minutes after Ray cluster comes up. Then `Ctrl-C`.

- [ ] **Step 3: Verify frozen mode log signals**

While the run is up (or against the saved log):

```bash
grep -E "subagent mode|subagent router|live router" /tmp/parl_v2_frozen_smoke.log | head -10
```

Expected output (IPs/ports may differ; key is two distinct ports and `subagent mode: frozen`):
```
[parl_v2] subagent mode: frozen
[parl_v2] subagent router: http://<ip>:<port_A>/generate
[parl_v2] live router:     http://<ip>:<port_B>/generate
```
where `port_A != port_B`.

Also confirm two SGLang router processes:
```bash
pgrep -af sglang_router | wc -l
```
Expected: `2`.

If only 1 router exists, frozen mode failed silently — investigate `_resolve_sglang_config` and the SglangConfig YAML before continuing.

- [ ] **Step 4: Verify W&B `parl/subagent_endpoint_distinct == 1` in frozen mode**

If wandb is online (`WANDB_MODE=online`), check the run's W&B page for the `parl/subagent_endpoint_distinct` metric — it should be `1`.

If offline (default for the 0.6B debug script), inspect the local wandb dir:
```bash
RUN_ID=$(ls -1t /workspace/miles/saves/Qwen3-0.6B-parl-v2 | head -1)
find /workspace/miles -path "*wandb/offline-run*" -type d 2>/dev/null | head -1
```
Open the matching `wandb-summary.json` (under that offline-run dir) and confirm `parl/subagent_endpoint_distinct: 1`.

- [ ] **Step 5: Stop the frozen run cleanly**

```bash
pkill -9 sglang
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
```

- [ ] **Step 6: Shared (ablation) mode smoke (0.6B, debug_minimal)**

```bash
cd /ssd0/guanxing/miles
MODE=debug_minimal SUBAGENT_MODE=shared bash examples/parl_math_v2/run-qwen3-0.6B-parl-v2.sh 2>&1 | tee /tmp/parl_v2_shared_smoke.log
```

Again, wait for ≥3 successful rollout iterations, then `Ctrl-C`.

- [ ] **Step 7: Verify shared mode log signals**

```bash
grep -E "subagent mode|subagent router|live router" /tmp/parl_v2_shared_smoke.log | head -10
```

Expected (the two URLs should be **identical**):
```
[parl_v2] subagent mode: shared (ablation)
[parl_v2] subagent router: http://<ip>:<port>/generate
[parl_v2] live router:     http://<ip>:<port>/generate
```

Confirm only one router process:
```bash
pgrep -af sglang_router | wc -l
```
Expected: `1`.

- [ ] **Step 8: Verify W&B `parl/subagent_endpoint_distinct == 0` in shared mode**

Same procedure as Step 4, but expect `parl/subagent_endpoint_distinct: 0`.

- [ ] **Step 9: Stop the shared run and document outcomes**

```bash
pkill -9 sglang
ray stop --force
pkill -9 ray
pkill -9 python
```

Verification PASS criteria:
- fast tests pass
- frozen run shows 2 routers + `subagent mode: frozen` + distinct ports + `endpoint_distinct == 1`
- shared run shows 1 router + `subagent mode: shared (ablation)` + same port + `endpoint_distinct == 0`
- both runs survive ≥3 rollout iterations without unhandled exceptions in the log

If any criterion fails, **do not** declare the implementation complete — investigate, fix, and re-run the smoke from the failing step. No commit is needed for this task (it's verification only).

---

## Self-review checklist (do not execute as task — done before sharing plan)

- Spec section 1 (yaml configs) → Task 1 ✓
- Spec section 2 (launch scripts) → Task 5 ✓
- Spec section 3 (`generate.py` routing + log) → Task 3 ✓
- Spec section 4 (`tool.py` kwarg) → Task 2 ✓
- Spec section 5 (observability) → Task 4 ✓
- Spec section 6 (edge cases): no new code needed; covered by `_resolve_sglang_config:1103` assert (Task 1 step 3) and existing miles guardrails ✓
- Spec section 7 (file change summary) → matches Tasks 1–5 ✓
- Spec section 8 (verification) → Task 6 ✓
- Spec section 9 (paper alignment table) → narrative; no task ✓

No placeholders, no "TODO", no "similar to Task N". Each step shows the exact code or command. Function signatures are consistent (`router_url` everywhere it appears).
