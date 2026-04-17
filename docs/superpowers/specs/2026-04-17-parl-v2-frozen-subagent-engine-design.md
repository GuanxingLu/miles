# PARL v2 Frozen Subagent Engine — Design

**Date:** 2026-04-17
**Target:** `examples/parl_v2/`
**Reference:** `.claude/reference/kimi_k2.5_parl_agent_swarm.md` (K2.5 PARL paper, arXiv:2602.02276)
**Prior designs:**
- `docs/superpowers/specs/2026-04-17-parl-v2-critical-steps-refactor-design.md`
- `docs/superpowers/specs/2026-04-17-parl-v2-agent-swarm-alignment-design.md`

## 背景

K2.5 PARL Agent Swarm 的核心解耦：Orchestrator **可训练**、Subagent **冻结**。冻结避免端到端联合优化的 credit assignment ambiguity 与 training instability，让 Subagent 输出作为环境观测而非可微分决策点。

当前 `parl_v2`（agent-swarm-alignment 设计落地后）已经做对了 context isolation —— subagent 跑独立 chat（system_prompt + 用户 prompt 单 shot），orchestrator 只看到 subagent 的最终 tool message。但 subagent 调的是**同一个 SGLang router**，权重和 orchestrator 共用 → subagent 实际上跟着 RL 一起更新，违反"frozen subagent"。

本设计实现"独立起一个冻结的 subagent 实例"。

## 范围与不做项

**做（v1）：**
- Hard frozen at SFT — subagent 在训练全程使用 `args.hf_checkpoint` 的初始权重（true hard-frozen-A）
- 拓扑：colocate + split — actor / live engine / frozen subagent engine 三者全部落在同一组 GPU 上，rollout 池内部按 yaml 切分
- 纯实例化在 `examples/parl_v2/`，不动 miles 核心
- 保留 ablation 模式（subagent 走 live router，跟 live policy 一起更新）做对照实验

**不做（留作 follow-up）：**
- Periodic refresh / curriculum（K2.5 paper L267–269 的"逐步迁移到更大 subagent"）
- 异构尺寸 subagent（yaml 写一个不同的 `model_path` 即可，不需要代码改动；但 v1 默认 `model_path` 为空 → fallback 到 orchestrator 的 hf_checkpoint）
- Subagent 也具备 tool-use / 多轮（subagent 仍是 single-shot 纯文本生成）
- Separated 模式（actor 与 rollout 不共享 GPU）— 当前所有实验都跑 colocate，不需要这个分支
- Token-level gradient masking / MuonClip / Wide-Deep Search 合成 prompt（独立设计）

## 关键事实（来自 miles 代码）

设计依赖以下 4 个已经存在但未被 parl_v2 用到的事实：

1. **`--sglang-config` 已支持多 model 多 ServerGroup**（`miles/backends/sglang_utils/sglang_config.py:46–167`）。每个 model 拿到独立 router；group 之间可以不同 `num_gpus_per_engine` 和 `model_path`。

2. **`update_weights: false` 已是 model-level 字段**（`sglang_config.py:49–54, 83–92`）。带这个标记的 model 自动从 weight update 路径剔除，由 `_get_updatable_server`（`miles/ray/rollout.py:421–425`）做过滤。

3. **Per-group offload 已就位**（`rollout.py:1050`）：`needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus`。colocate 范围内的所有 group 都会 offload，包括 frozen group。

4. **IPC weight update 在 TP all-gather 之后再 ship**（`miles/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py:106`）。每个训练 rank 在 IPC ship 前已经持有完整 HF 模型权重，所以 IPC gather group 不需要覆盖所有训练 rank —— frozen subagent 占的 rank 被 IPC 跳过没问题，跟现有 `placeholder` worker_type 同构处理。

→ 这 4 点意味着"colocate + split + frozen subagent"在 miles 已有基础设施上**不需要任何核心改动**。

## 拓扑

colocate 模式、总 GPU 不变：
```
total = actor_num_gpus = rollout_num_gpus = R

actor:    [0, R)        训练时活，rollout 时 offload
live:     [0, R-F)      colocate，IPC 接收 weight update
subagent: [R-F, R)      colocate，update_weights:false，IPC 跳过
```

**关键运行时设置**：frozen engine 在每次训练 step 都会跟 live engine 一起被 `release_memory_occupation` 释放 GPU，但**没有 IPC 路径把权重推回来**。为了让 frozen engine 在 resume 时仍能拿到原始权重，必须在其 yaml 条目里显式开启 SGLang 的 `enable_weights_cpu_backup: true`（见 §1），它让 SGLang 在 release 时把权重 pin 到 CPU、resume 时从 CPU 还原。这与 miles 为 LoRA engines 默认打开的机制（`miles/backends/sglang_utils/sglang_engine.py:657-658`）是同一套。

每 step 额外开销：frozen engine 的 weights copy 到 CPU pinned mem 和拷回 GPU，~几百 ms 量级；+1 份 pinned host RAM（4B ≈ 8GB，按 TP rank 分）。

### 为什么必须 CPU backup（实测）

2026-04-17 4B 跑到 step 21 时 curl 两个 endpoint 对打：

| Endpoint | 输出 | `weight_version` |
|---|---|---|
| live (18765) | `" Paris. The capital of Germany is Berlin."` | `"21"` |
| subagent (4077) | `"!!!!!!!!!"` (token ids 全是 0) | `"default"` |

subagent 从未收到 IPC 更新（`"default"`），且输出是退化的 token-0——SGLang 默认 `enable_weights_cpu_backup=False`，release 时 weights 被真 free，resume 时 GPU 上是未初始化内存，推理落入 argmax 塌到低位 token id。加 `enable_weights_cpu_backup: true` 修复此问题。

## 1. SGLang 配置 yaml

`examples/parl_v2/sglang_config_4B.yaml` —— 8 卡（colocate 下 actor_num_gpus=rollout_num_gpus=8），actor TP=2 / subagent TP=2，6+2 拆分两边都能整除：
```yaml
sglang:
  - name: actor
    update_weights: true
    server_groups:
      - worker_type: regular
        num_gpus: 6                   # 3 个 engine × TP=2
  - name: subagent
    # model_path 留空 → fallback args.hf_checkpoint（hard-frozen at SFT）
    # 异构尺寸 ablation：填一个具体 path（例如 /path/to/Qwen3-0.6B）
    update_weights: false
    server_groups:
      - worker_type: regular
        num_gpus: 2                   # 1 个 engine × TP=2
        overrides:
          enable_weights_cpu_backup: true   # 见"拓扑"节说明
```

`examples/parl_v2/sglang_config_0.6B.yaml` —— 4 卡，TP=1：
```yaml
sglang:
  - name: actor
    update_weights: true
    server_groups:
      - worker_type: regular
        num_gpus: 3                   # 3 个 engine × TP=1
  - name: subagent
    update_weights: false
    server_groups:
      - worker_type: regular
        num_gpus: 1                   # 1 个 engine × TP=1
        overrides:
          enable_weights_cpu_backup: true
```

- 总和（actor + subagent）= `--rollout-num-gpus`，否则 `_resolve_sglang_config:1103` assert 失败
- 每个 group 的 `num_gpus` 必须能被 `rollout_num_gpus_per_engine`（4B=2, 0.6B=1）整除
- subagent group **必须**带 `overrides.enable_weights_cpu_backup: true`，否则 frozen engine 在第 1 次 train step 之后退化（见"拓扑"节实测结果）

## 2. Launch 脚本

两个脚本默认 frozen，shell env 切 ablation。`run-qwen3-4B-parl-v2.sh` 末尾改：
```bash
SUBAGENT_MODE=${SUBAGENT_MODE:-frozen}
if [ "$SUBAGENT_MODE" = "frozen" ]; then
   SGLANG_EXTRA_ARGS=(--sglang-config examples/parl_v2/sglang_config_4B.yaml)
else
   SGLANG_EXTRA_ARGS=()                           # ablation: 默认 single-model
fi

python examples/parl_v2/run_parl_v2.py \
   ... \
   "${SGLANG_EXTRA_ARGS[@]}"
```
`run-qwen3-0.6B-parl-v2.sh` 同模式（yaml 路径换成 `_0.6B.yaml`）。

ablation 跑法：`SUBAGENT_MODE=shared bash run-qwen3-4B-parl-v2.sh`。

## 3. `generate.py` 改动（关键路由发现）

**时序约束：** `args.sglang_model_routers` 在 `start_rollout_servers` 内部填充（`miles/ray/rollout.py:1092`），这发生在 `train.py` 主流程中，**晚于** launcher 注入 `extra_env_vars`。所以不能走 env var 路径。

miles 提供了公共 helper `get_model_url(args, model_name)`（`miles/rollout/sglang_rollout.py:41–57`），专门处理这种多 model 路由场景，自动 fallback 到 live router。直接用即可：

```python
from miles.rollout.sglang_rollout import get_model_url

async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    subagent_router_url = get_model_url(args, "subagent")  # 自动 fallback
    # ...
    # 把 subagent_router_url 通过 closure 传给 _execute_tool_calls_parallel → _assign_task_call
    tool_messages, stats = await _execute_tool_calls_parallel(
        tool_calls,
        registry=registry,
        tokenizer=tokenizer,
        router_url=subagent_router_url,
    )
```

模式由 yaml 拓扑自动推断（`get_model_url` 内置）：
- yaml 里有 `subagent` 条目 → frozen 模式 → 用 subagent router URL
- 没有（即 ablation） → fallback 到 live router URL

**首次调用打一行 log**（module-level cache，避免每个 rollout 都打）：
```python
_logged_endpoint = False

async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    global _logged_endpoint
    args = input.args
    subagent_router_url = get_model_url(args, "subagent")
    live_router_url     = get_model_url(args, "actor")  # 或读 args.sglang_router_ip/port
    if not _logged_endpoint:
        mode = "frozen" if subagent_router_url != live_router_url else "shared (ablation)"
        logger.info(f"[parl_v2] subagent mode: {mode}")
        logger.info(f"[parl_v2] subagent router: {subagent_router_url}")
        logger.info(f"[parl_v2] live router:     {live_router_url}")
        _logged_endpoint = True
```

## 4. `tool.py` 改动

`_assign_task_call` 增加 `router_url` 必填 kwarg；删除 `_router_url` 函数和 env var 兜底逻辑：

```python
async def _assign_task_call(
    params: dict, *, registry: dict[str, str], tokenizer, router_url: str
) -> tuple[str, bool]:
    # ... 现有校验逻辑 ...
    payload = { ... }
    async with _get_semaphore():
        try:
            output = await post(router_url, payload)
        except Exception as e:
            return f"__SOLVER_ERROR__: {e}", False
    # ... 现有 is_valid 逻辑 ...
```

`tool.py` 不再读任何环境变量，全靠 `generate.py` 把 URL 注入下来。`run_parl_v2.py` 也无需任何改动。

## 5. Observability

人类可读的"frozen / shared"标签由节 3 的 startup log 承担（一次性打到 stdout）。这里只补一条**机器可过滤**的数值指标 `parl/subagent_endpoint_distinct`，让 W&B sweep 一眼区分两组 run：

```python
# rollout_log.py::log_rollout_data
sub_url  = get_model_url(args, "subagent")
live_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
log_dict["parl/subagent_endpoint_distinct"] = int(sub_url != live_url)
```

frozen 模式恒为 `1`，shared 恒为 `0`。走现有 `tracking_utils.log` 通道（数值 step-level），不引入 `wandb.run.summary` 直调依赖，离线 wandb 也能工作。

## 6. 边界条件

- `total_num_gpus == rollout_num_gpus`：由 `_resolve_sglang_config:1103` assert 兜底
- `model_path` 默认推断（`sglang_config.py:83–92`）：留空时 = `args.hf_checkpoint`，`update_weights` 自动按是否等于 hf_checkpoint 推断
- `--colocate`：`run_parl_v2.py:214` 已无条件传，无需额外 assert
- `train_async.py`：parl_v2 入口走 `train.py`，不会触发 async；无需 assert
- LoRA：weight sync 路径已限定 colocated（`update_weight_from_tensor.py:262`），不冲突

## 7. 文件改动清单

全部改动局限在 `examples/parl_v2/`，miles 核心 0 改动。

| 文件 | 改动 | 行数估计 |
|---|---|---|
| `examples/parl_v2/sglang_config_4B.yaml`     | **新增** | ~12 行 yaml |
| `examples/parl_v2/sglang_config_0.6B.yaml`   | **新增** | ~12 行 yaml |
| `examples/parl_v2/run-qwen3-4B-parl-v2.sh`   | 加 `SUBAGENT_MODE` 切换 + `--sglang-config` | ~5 行 |
| `examples/parl_v2/run-qwen3-0.6B-parl-v2.sh` | 同上（yaml 路径换 `_0.6B.yaml`） | ~5 行 |
| `examples/parl_v2/generate.py`               | 用 `get_model_url(args, "subagent")` 拿 URL；闭包传给 `_execute_tool_calls_parallel`；首次打一行 log | ~10 行 |
| `examples/parl_v2/tool.py`                   | `_assign_task_call` 加 `router_url` kwarg；删 `_router_url` + env 兜底 | ~5 行（净 -5） |
| `examples/parl_v2/rollout_log.py`            | `parl/subagent_mode` + `parl/subagent_endpoint_distinct` | ~6 行 |
| `examples/parl_v2/run_parl_v2.py`          | `ScriptArgs.sglang_config` 字段 + `execute()` 里 forward 到 `sglang_args` train string（typer 不接受未知 CLI arg） | ~6 行 |

总计约 30 行净增（不含 yaml 文件本体）。

## 8. 验证

按顺序，每步独立可断点：

1. **静态校验**：`pytest tests/fast`——examples 改动应不影响 fast 测试，预期全过。
2. **0.6B frozen 烟测** `MODE=debug_minimal SUBAGENT_MODE=frozen bash run-qwen3-0.6B-parl-v2.sh`：
   - 启动日志出现 `[parl_v2] subagent mode: frozen` + 两个不同 router endpoint
   - `ps` 看到 2 个 SGLang router 进程（端口不同）
   - W&B `parl/subagent_endpoint_distinct == 1`
   - 跑 ≥3 个 rollout step 不崩
3. **0.6B shared (ablation) 烟测** `SUBAGENT_MODE=shared bash run-qwen3-0.6B-parl-v2.sh`：
   - 启动日志 `[parl_v2] subagent mode: shared (ablation)` + 同一 endpoint
   - 只有 1 个 router
   - W&B `parl/subagent_endpoint_distinct == 0`
   - 跑 ≥3 个 rollout step 不崩
4. **frozen vs shared 短期对比**（各 10–50 step）：
   - reward 曲线**不应完全重合**（frozen 模式 subagent reward 不随训练漂移；shared 模式跟 live policy 一起涨）
   - 定性 sanity，不是严格 acceptance 标准

## 9. 与 K2.5 paper 对齐情况

| K2.5 设计要素 | v1 实现 | 状态 |
|---|---|---|
| Subagent frozen | 独立 SGLang model + `update_weights: false` | ✅ 完成 |
| Context sharding | subagent fresh chat、orchestrator 只看 tool message | ✅ 已在前序设计完成 |
| Critical steps 预算 | turn-based `1 / 2` cost | ✅ 已在前序设计完成 |
| `create_subagent` + `assign_task` 双工具 | per-rollout registry + parallel assign | ✅ 已在前序设计完成 |
| Reward 三段式 | `r_perf + λ₁ r_parallel + λ₂ r_finish` | ✅ 已在前序设计完成 |
| 涌现式专家化 | `system_prompt` 由 orchestrator 动态生成 | ✅ 已在前序设计完成 |
| Subagent 自带 tool-use / 多轮 | single-shot 纯文本 | ❌ 故意不做 |
| Curriculum / refresh subagent | hard-frozen-A | ❌ v1 不做 |
| Wide-Deep Search 合成 prompt | 当前用 dapo-math-17k | ❌ 数据工程，独立设计 |
| Token-level gradient masking | 用 PPO clip | ❌ 独立设计 |
| MuonClip 优化器 | 用 Adam | ❌ 独立设计 |

## 10. 考虑过但未采用的方案：actor < rollout 拓扑

**未做。** 留在这里是为了防止未来踩同一个坑。

**方案描述**：让 frozen engine 住在 actor GPU 范围**之外**（`[actor_num_gpus, rollout_num_gpus)`），利用 `miles/ray/rollout.py:1050` 的 per-group offload 判断 `needs_offload = group_abs_start < megatron_num_gpus` —— 这样 frozen group 永远不被 offload，权重从不被 release，不需要 CPU backup。

**需要改动**：
- `miles/utils/arguments.py:1990–1995`：原本 colocate 下硬把 `rollout_num_gpus` override 成等于 `actor_num_gpus`，要改成允许 `rollout > actor`
- `miles/ray/placement_group.py:95–100`：colocate 下 PG 大小从 `actor` 改成 `max(actor, rollout)`
- `run_parl_v2.py`：加 `actor_num_gpus_per_node` / `rollout_num_gpus` 字段并 forward
- Launch script：frozen 分支里 pin actor 到 R-F 张卡

**为什么不做**：
- 需要 2 处 miles core 改动，污染面更大
- Actor 训练吞吐 -25%（4B 从 8 卡缩到 6 卡），需要永久付这个税
- `enable_weights_cpu_backup: true` 方案只改 yaml 1 行，actor 保持 8 卡满吞吐，且用的是 miles 为 LoRA 已经验证过的同一套 SGLang 机制，胜率更高

**什么时候可能反过来选这个**：
- Qwen3-4B bf16 在 CPU pinned mem 占 ~8GB；如果未来做更大模型（比如 70B MoE），CPU backup 成本高到装不下
- 或 frozen engine 需要 per-step 毫秒级 resume 延迟的场景（CPU→GPU 拷贝不可忽略）

此时再切 actor<rollout 方案。实现细节已经被验证可行（相关 commit `2d7068963` / `73ba0176d` 已 revert，代码曾经在 branch 上跑通过 parse 和 AST 校验），未来启用时可作为参考。
