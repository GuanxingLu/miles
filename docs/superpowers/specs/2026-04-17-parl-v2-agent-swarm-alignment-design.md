# PARL v2 Agent Swarm Alignment — Design

**Date:** 2026-04-17
**Target:** `examples/parl_v2/`
**Reference:** `.claude/reference/kimi_k2.5_parl_agent_swarm.md` (K2.5 PARL paper summary, arXiv:2602.02276)
**Prior design:** `docs/superpowers/specs/2026-04-17-parl-v2-critical-steps-refactor-design.md` (critical_steps as budget + reward re-labeling; already merged)

## 背景

上一轮对齐把 `critical_steps` 从 reward 负项改成 episode 预算、把 `r_parallel` / `r_finish` 语义调回 paper。本轮继续对齐两项偏离：

1. **Critical steps 单位错了**：paper L93–118 定义 step = tool invocation / environment interaction 计数（per-benchmark 预算 "BrowseComp Orch=15 / Sub=100" 即 turn/call 单位）。当前 `generate.py` 用的是 **token** 累加（`orch_new_tokens + max_solver_tokens`）。
2. **Tool schema 错了**：paper 用 `create_subagent(name, system_prompt)` + `assign_task(agent, prompt)` 两工具支持**异构** specialized subagent 的命名复用；当前只有一个 `consult_solvers(problem, num_parallel)`，fixed solver prompt，本质是"parallel voting"而非 agent swarm。

**本次明确不改** 的 paper 偏离（保持解耦）：
- Subagent 冻结（live policy 依然，需另外一个 frozen engine）
- Token-level gradient masking（PPO clip 不改）
- MuonClip 优化器（Adam 保留）
- Curriculum / Wide-Deep Search 合成 prompt

**本次明确保留** 的故意偏离：
- Reward 退火保持**禁用**（ANNEAL_FRAC = 100.0），为了显式鼓励并行。

---

## 范围选择：Option B（minimal-change 异构化）

Subagent 保持 **single-shot**（不上多轮 subagent；不给 subagent 自己的工具）。Subagent 的 system_prompt 由 orchestrator 通过 `create_subagent` 动态定义，获得 paper 的"涌现式专家化"能力，但不承担 multi-turn subagent 的工程复杂度（frozen engine 调度、subagent max-steps 预算、subagent 独立 tool chain 全部跳过）。

直接代价：`S_sub,i` 恒等于 1，critical_steps 里 `max_i S_sub,i` 失去宽度信号——"一轮 spawn 1 个" 和 "一轮 spawn 8 个" 的 CS cost 都是 2。宽度信号全部由 `r_parallel` 承担。这是对齐目标的自觉取舍。

---

## 1. Tool Schema

```jsonc
// tool_specs (两条)
{
  "type": "function",
  "function": {
    "name": "create_subagent",
    "description": "Create a custom subagent with a specific system prompt and name for later reuse.",
    "parameters": {
      "type": "object",
      "properties": {
        "name": {"type": "string", "description": "Unique name for this agent configuration."},
        "system_prompt": {"type": "string", "description": "System prompt defining the agent's role/boundaries."}
      },
      "required": ["name", "system_prompt"]
    }
  }
},
{
  "type": "function",
  "function": {
    "name": "assign_task",
    "description": "Launch a subagent to produce a single solution. You can launch multiple agents concurrently by emitting multiple assign_task calls in one turn.",
    "parameters": {
      "type": "object",
      "properties": {
        "agent": {"type": "string", "description": "Which created agent to use."},
        "prompt": {"type": "string", "description": "The task for the agent to perform."}
      },
      "required": ["agent", "prompt"]
    }
  }
}
```

**删除：** 旧 `consult_solvers` schema、`STATS_FOOTER_PREFIX` 及其所有 footer 生成 / 解析代码。

## 2. Registry 语义（per-rollout）

- **载体：** `dict[str, str]`（`name -> system_prompt`），在 `generate.py` 每个 rollout 开始处 new 一个空 dict，连同 `tokenizer` 一起 closure-bind 到 `execute_tool`（通过 `functools.partial` 或 lambda）。生命周期 = 一次 rollout。
- **签名变化：** `execute_tool(name, params, *, registry, tokenizer)` —— 两个 kwarg 都由 closure 注入，上层 `_execute_tool_calls_parallel` 只看到 `(name, params) -> str` 的统一接口。
- **`create_subagent`：**
  - 校验 name 非空、system_prompt 非空
  - 重复 name → **直接覆盖**，allow orchestrator iteratively refine
  - 上限 **8** 个 unique name；超限（已有 8 个且新 name 不在其中）→ 返回 `"Error: subagent registry full (max 8). Reuse an existing agent or avoid creating more."`
  - 成功返回 `"Registered subagent '{name}'."`
- **`assign_task`：**
  - 若 `agent` 不在 registry → 返回 `"Error: agent '{agent}' not found. Call create_subagent first."`
  - 若在 → 用 registry[agent] 当 system_prompt + params["prompt"] 当 user message，apply chat template，POST SGLang router，返回 subagent 文本输出

## 3. Turn 内执行顺序（自定义 parallel wrapper）

在 `generate.py` 内部实现 `_execute_tool_calls_parallel(tool_calls, execute_one)`，**取代** miles 默认的串行 `execute_tool_calls`（不改 miles 核心）：

1. 按 name 拆分为 `create_calls` 和 `assign_calls`
2. `create_calls` **串行** 执行（全部是 dict 写入，瞬时）
3. `assign_calls` 按 `max_concurrent_assign_per_turn = 8` 截断多余
4. 截断后的 `assign_calls` 用 `asyncio.gather` **并行** 执行
5. 按原 tool_call 顺序重新排列 tool_messages 返回（保持 response 文本里 tool_call 和 tool_response 的对应关系）

**未知 tool name** → 和当前一致，返回错误 string。

## 4. Critical Steps（turn-based）

```
phase_cost(turn) = 1   if no assign_task in turn (final answer / create-only / length-truncated orch)
                 = 2   if ≥1 assign_task in turn
critical_steps   = Σ phase_cost
```

注意：
- `create_subagent` 无论多少个，本身不产生 subagent 运行 → 只算 orchestrator 自己那 1 step
- Orchestrator 本轮 `finish_reason ∈ {"abort", "length"}` 且未 emit tool_call（或 emit 后被截断）→ 按 final 处理，cost=1，跳出主循环

**默认预算：** `rollout_max_critical_steps = 2 * generate_max_turns`（`run_parl_v2.py` 的 `__post_init__` 填充）。`generate_max_turns=6` → 默认 `rollout_max_critical_steps=12`。

**终止：** `generate.py` 主循环保持和当前一致的"critical_steps ≥ max_cs → TRUNCATED"。

**向后兼容：** CLI 参数 `--rollout-max-critical-steps` 保留，语义从 token 单位改成 turn 单位。

## 5. Reward（structured metadata，无正则）

### 数据流

`generate.py` 在每个 orchestrator turn 结束时，往 `sample.metadata["turns"]` 追加：

```python
{
  "n_create": int,   # 本轮 create_subagent 调用数
  "n_assign": int,   # 本轮 assign_task 调用数（已经过 max_concurrent 截断）
  "n_valid": int,    # 本轮 assign_task 中 subagent 输出非空、非 error 的数量
  "final": bool,     # 是否是最后答题 turn（无 tool_call 或 finish_reason=length）
}
```

**完全删除** 旧的 footer 机制（`_STATS_FOOTER_RE`, `_SOLVER_TOKENS_RE`, `_max_solver_tokens_from`, `_read_per_call_stats`, `_count_tool_calls`, `tool.py::_format_candidates::footer`）。

### Per-turn advantage

```
per_turn_r[t] = λ₁ · (n_assign_t / PARALLEL_CAP)
              + λ₂ · (n_valid_t / max(1, n_assign_t))
```

- 纯 create turn（`n_assign_t == 0` 且非 final）→ per_turn_r = 0
- final turn credit = `r_perf + λ_box · r_box`（和现行逻辑一致）

### 全局 score（每个 sample 的 scalar）

```
r_parallel = min(Σ n_assign_t, PARALLEL_CAP) / PARALLEL_CAP
r_finish   = Σ n_valid_t / Σ n_assign_t      (0 if Σ n_assign_t == 0)
r_box      = 1 if '\boxed{' in response else 0
r_perf     = 1 if math_dapo_compute_score(response, label) > 0 else 0

score = r_perf + λ₁ · r_parallel + λ₂ · r_finish + λ_box · r_box
```

### 常量

```python
LAMBDA1_INIT = 0.3
LAMBDA2_INIT = 0.2
LAMBDA_BOX   = 0.1
PARALLEL_CAP = 16
ANNEAL_FRAC  = 100.0   # annealing 保持禁用，故意鼓励并行
GRPO_STD_EPS = 1e-6
```

### Group-norm

和现行一致：finals 一组、per-turn spawn rewards 一池；`--group-rm` 模式下写 `sample.per_token_advantages`（loss_mask 的每个 span 对应一个 turn，span 里的 token 统一填该 turn 的 advantage）。

## 6. Subagent 调用细节（`tool.py::_assign_task_call`）

```python
# 伪码
chat = [
  {"role": "system", "content": registry[agent]},
  {"role": "user",   "content": prompt},
]
text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
payload = {"text": text, "sampling_params": {"max_new_tokens": 1024, "temperature": 1.0, "top_p": 1.0}}
async with semaphore:
    output = await post(router_url, payload)
```

- `SOLVER_MAX_NEW_TOKENS = 1024`（保留）
- `SOLVER_TEMPERATURE = 1.0`（保留）
- `SOLVER_CONCURRENCY = 16`（保留；全局信号量限 SGLang 并发）
- **Tokenizer：** 和 orchestrator 共用——由 `generate.py` 的 `input.state.tokenizer` closure-bind 进 `execute_tool`，`_assign_task_call` 从 kwarg 拿；不在 tool.py 里重复加载。
- `__SOLVER_ERROR__:` prefix + 空字符串视为 invalid（`n_valid` 不计）

## 7. Orchestrator System Prompt

重写 `prompts.py::ORCHESTRATOR_SYSTEM_PROMPT`，描述新工具和异构专家模式。遵循 paper L225–246 的"不直接指令 parallelize，通过任务分布 shape"原则——**不写 "please parallelize"**，只描述工具能力和示例专家角色：

```
You are solving a math problem. You have two tools:

- create_subagent(name, system_prompt): Register a specialized math-solver agent
  with a custom system prompt (e.g., an algebraic manipulation specialist, a
  number-theory expert, a computational checker). The name is reusable.
- assign_task(agent, prompt): Dispatch a task to a previously-created subagent.
  Each subagent produces one candidate solution. You can emit multiple
  assign_task calls in the same turn; they run in parallel.

Think step by step. When multiple independent angles on the problem would help,
create the appropriate specialists and assign them tasks. Reconcile their
outputs, then end with the final answer in \boxed{...}.
```

## 8. rollout_log.py 更新

**`_REWARD_KEYS` 改动：**
- 删 `n_spawn`（旧名）
- 加 `n_assign`, `n_create`, `registry_size`

**新增 metrics：**
- `multi_turn/assign_per_turn/*`（从 `sample.metadata["turns"]` 的 `n_assign_t` 分布统计）
- `multi_turn/n_unique_agents_used/*`（从 metadata 统计 orchestrator 实际 assign 到的不同 agent 名字数）

**保留：** `spawn_rate`（改为 `assign_rate`：至少一个 turn 有 n_assign>0 的 rollout 占比）、`turns_per_rollout`、`effective_response_ratio`、spawn-by-difficulty（语义现为 n_assign-by-difficulty）、group-std 检查（key 从 `n_spawn` 改 `n_assign`）。

## 9. `run_parl_v2.py` 改动

- `ScriptArgs.__post_init__` 里把 `self.rollout_max_critical_steps = self.rollout_max_critical_steps or self.rollout_max_response_len` 改为 `or (2 * self.generate_max_turns)`
- 无新 CLI arg
- `generate_max_turns` 保持 6（shell 默认）

---

## 10. 文件改动汇总

| 文件 | 改动 |
|---|---|
| `examples/parl_v2/tool.py` | 重写：`tool_specs` 改双工具；新增 `_assign_task_call`；`execute_tool(name, params, registry)` 加 registry；删除 `_solver_call`, `_format_candidates`, `STATS_FOOTER_*`, `_is_valid_solver_output`（保留 `_router_url`, `_get_semaphore`, 常量） |
| `examples/parl_v2/generate.py` | 每 rollout init registry + closure-bind execute_tool；自写 `_execute_tool_calls_parallel`（create 串行、assign gather、concurrent cap）；每轮记录 `sample.metadata["turns"]`；critical_steps 改 turn 累加；删除 `_SOLVER_TOKENS_RE`, `_max_solver_tokens_from` |
| `examples/parl_v2/reward.py` | 删 `_STATS_FOOTER_RE`, `_read_per_call_stats`, `_count_tool_calls`；新增 metadata 读取 + per-turn 聚合；pure-create turn = 0 credit；`_fill_per_token_advantages` 按 metadata["turns"] 对齐 loss_mask span |
| `examples/parl_v2/rollout_log.py` | `_REWARD_KEYS` 更新；新增 `assign_per_turn`, `n_unique_agents_used` 指标；`n_spawn`→`n_assign`（全部引用） |
| `examples/parl_v2/run_parl_v2.py` | `rollout_max_critical_steps` 默认值从 `rollout_max_response_len` 改成 `2 * generate_max_turns` |
| `examples/parl_v2/prompts.py` | 重写 `ORCHESTRATOR_SYSTEM_PROMPT`；删除 `SOLVER_PROMPT_TEMPLATE`（不再用固定模板） |

## 11. 非本次范围

- Subagent frozen engine（需独立 SGLang 实例 + checkpoint 调度，另立设计）
- Token-level gradient masking / MuonClip（loss kernel / optimizer 改动，另立设计）
- Wide/Deep Search 合成数据（数据集工程，另立设计）
- Reward annealing（故意保留禁用）

## 12. 验证

- `pytest tests/fast`：examples 改动不应影响 fast 测试，预期仍通过
- 人工烟测 `run-qwen3-0.6B-parl-v2.sh`（debug_minimal 模式）：观察 W&B 的 `reward/n_assign`, `reward/n_create`, `reward/critical_steps`, `multi_turn/n_unique_agents_used` 出现且数值合理
- 检查 rollout log 里的样本：orchestrator 是否产生了多种 agent name；`assign_task` 未知 name error 是否可被 orchestrator 自然避免
