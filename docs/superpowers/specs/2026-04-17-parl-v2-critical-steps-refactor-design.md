# PARL v2 Critical Steps & Reward Alignment — Design

**Date:** 2026-04-17
**Target:** `examples/parl_math_v2/`
**Reference:** `.claude/reference/kimi_k2.5_parl_agent_swarm.md` (K2.5 PARL paper summary, arXiv:2602.02276)

## 背景

当前 `examples/parl_math_v2/` 的实现与 K2.5 PARL paper 有两处偏离：

1. **Critical Steps 形态错了**：paper 把 critical steps 作为 episode-length **预算约束**（第 175 行："训练时用 critical steps 而非 total steps 限制 episode 长度"），当前代码把它当成 reward 里的负项 `-λ₃·critical_steps_ratio`。
2. **r_parallel / r_finish 语义互换**：paper 的 r_parallel 是"实例化奖励，防 serial collapse"，r_finish 是"subagent 完成率"。当前代码把 `valid/total`（完成率）叫 r_parallel，把 `has_boxed`（orchestrator 交卷）叫 r_finish，角色错配。

此外 miles 原 rollout 里的 `response_length` budget 同时计 orchestrator 和 solver 文本，这让并行 spawn N 个 solver 消耗的预算 ≈ Σ solver tokens，**反向惩罚并行**，与 paper 初衷相反。

## 目标

对齐 paper；critical_steps 作预算不作 reward；r_parallel / r_finish 换到 paper 语义；保留 `has_boxed` 作为独立的 format/交卷信号。

---

## Reward 公式（终稿）

```
score = r_perf + λ₁·r_parallel + λ₂·r_finish + λ_box·r_box
```

| 项 | 定义 |
|---|---|
| `r_perf` | `1.0 if math_dapo.score > 0 else 0.0` |
| `r_parallel` | `min(Σ num_parallel_i, PARALLEL_CAP) / PARALLEL_CAP`（P2） |
| `r_finish` | `Σ valid_i / Σ total_i` if Σ total_i > 0 else 0.0 |
| `r_box` | `1.0 if has_boxed else 0.0` |

**删掉**：
- `COST_PER_CALL` 项（critical_steps 预算已隐式惩罚多 call）
- `λ₃·critical_steps_ratio` reward 惩罚（改用预算约束）

**超参默认值**（reference 未给具体数值，按推荐）：

```python
LAMBDA1_INIT = 0.3   # r_parallel
LAMBDA2_INIT = 0.2   # r_finish
LAMBDA_BOX   = 0.1   # r_box（原 LAMBDA3 槽位复用、重命名）
PARALLEL_CAP = 16
ANNEAL_FRAC  = 100.0 # 保留故意不退火
```

## Per-turn 分解

```
r_final        = r_perf + λ_box · r_box
per_call_r[i]  = λ₁ · (num_parallel_i / PARALLEL_CAP)
               + λ₂ · (valid_i / total_i)
```

**credit 归属**：
- r_perf、r_box 属于 orchestrator 的最终 turn 决策 → final
- r_parallel、r_finish 是 per-call 量 → 归于触发 call 的 spawn turn
- `num_parallel_i` = 第 i 次 `consult_solvers` 的 `total`（footer 已有）

Group-norm（finals 一组、spawn-calls 一池）保持不变。

## Critical Steps 预算（方案 C1：双阈值）

**新增 arg**：`--rollout-max-critical-steps`（paper 的 episode-length 预算；默认等于 `--rollout-max-response-len`，向后兼容）
**保留 arg**：`--rollout-max-response-len`（语义改为 context-window cap）

**续跑条件**：`critical_steps < max_cs AND response_length < max_rl`，任一破即 `Sample.Status.TRUNCATED`。

**累加规则**（每轮 tool call 完成后）：
```
critical_steps += orch_turn_tokens + max_i(solver_tokens_i)
```
最后一轮无 spawn 的答题 turn：
```
critical_steps += final_orch_turn_tokens
```

**实现定位（方案 2a）**：不动 miles 通用 rollout。
- 新建 `examples/parl_math_v2/rollout.py`（拷 `miles/rollout/generate_hub/multi_turn.py` 并在续跑检查处加 critical_steps 逻辑）
- `tool.py` 的 `execute_tool` 把 `max(solver_tokens)` 通过 side-channel（stash 到调用方可读位置，比如 coroutine-local / 放进 tool 响应 metadata）传给 rollout
- `run_parl_math.py` 加 `--rollout-function-path` 指向 parl 自己的 rollout

## 文件改动

| 文件 | 改动 |
|---|---|
| `examples/parl_math_v2/rollout.py` | **新建**：拷 miles `multi_turn.py` + critical_steps 累加 + 双阈值检查 |
| `examples/parl_math_v2/reward.py` | 重写 reward；删 `_compute_critical_steps`、`COST_PER_CALL`、`LAMBDA3`、`critical_steps_ratio`；`LAMBDA3_INIT` → `LAMBDA_BOX`；改 per-turn 分解 |
| `examples/parl_math_v2/tool.py` | 保留 solver_tokens 在 footer（供 rollout 读）；增 side-channel 传 `max_solver_tokens` |
| `examples/parl_math_v2/rollout_log.py` | 新增 `r_box`、`critical_steps` 日志键；删 `critical_steps_ratio`、`lambda3` |
| `examples/parl_math_v2/run_parl_math.py` | 加 `--rollout-max-critical-steps`、`--rollout-function-path` |

## 不在本次范围

- `--use-tis` / `icepop_function`（已在暂存 diff 中，独立变更，保留现状）
- 动 miles 通用 rollout（明确走 2a 路线不污染 `miles/rollout/`）
- Subagent 冻结 / curriculum / MuonClip / token-level 梯度置零（paper 其他部分，超出本次对齐范围）

## 验证

- `pytest tests/fast` 仍通过（不应被 examples 修改影响）
- 运行一次 `scripts/` 里 parl_math 相关小脚本做烟测（可选，看用户是否有对应 launcher）
- 检查 W&B 指标里 `r_parallel`、`r_finish`、`r_box`、`critical_steps` 出现且数值合理
