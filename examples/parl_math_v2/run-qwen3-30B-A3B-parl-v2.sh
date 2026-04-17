#!/bin/bash

# PARL v2 prod run on Qwen3-30B-A3B (H200x16, 2 nodes).
# Thin wrapper around examples/parl_math_v2/run_parl_math.py.

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONBUFFERED=16
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export WANDB_API_KEY=${WANDB_API_KEY:-local-82cbbacfe8e3c0c527da528160bd76a1e85c9fea}
export WANDB_BASE_URL=${WANDB_BASE_URL:-http://33.180.4.104}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

DEV_REPO_DIR=${DEV_REPO_DIR:-${REPO_DIR}}
DATA_ROOT=${DATA_ROOT:-${DEV_REPO_DIR}/DATA}
MODEL_ROOT=${MODEL_ROOT:-${DEV_REPO_DIR}/MODEL}
MODE=${MODE:-normal}  # debug_minimal | normal
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
RUN_ID=${RUN_ID:-"run_$(date +%Y%m%d_%H%M%S)"}

MODEL_ARGS=(
   --model qwen3-30B-A3B
   --hf-checkpoint "${MODEL_ROOT}/Qwen3-30B-A3B"
   --ref-load "${MODEL_ROOT}/Qwen3-30B-A3B_torch_dist"
   --tensor-model-parallel-size 4
   --rollout-num-gpus-per-engine 4
)

RUN_ARGS=(
   --mode "${MODE}"
   --run-id "${RUN_ID}"
   --dev-repo-dir "${DEV_REPO_DIR}"
   --save-path "${DEV_REPO_DIR}/saves/Qwen3-30B-A3B-parl-v2/${RUN_ID}"
   --rollout-batch-size 32
   --global-batch-size 256
   --rollout-max-response-len 8192
)

PARALLEL_ARGS=(
   --num-gpus-per-node "${NUM_GPUS}"
)

DATA_ARGS=(
   --prompt-data "${DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl"
   # --eval-prompt-data "${DATA_ROOT}/aime-2024/aime-2024.jsonl"
)

GENERATE_ARGS=(
   --generate-max-turns 6
)

# MoE-specific perf args passed through to train.py via --extra-args.
# These mirror the multi-agent 30B-A3B launcher.
EXTRA_ARGS=(
   --sequence-parallel
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

cd "${REPO_DIR}"

RAY_GCS_PORT=${RAY_GCS_PORT:-26379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-28265}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# Bind miles' host-IP detection to MASTER_ADDR. Otherwise get_host_info()
# can fall back to 127.0.0.1 in egress-restricted envs, making the SGLang
# router unreachable from worker nodes.
export MILES_HOST_IP=${MILES_HOST_IP:-${MASTER_ADDR}}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --port ${RAY_GCS_PORT} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=${RAY_DASHBOARD_PORT}
export RAY_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}"
export MILES_SCRIPT_EXTERNAL_RAY=1

# SUBAGENT_MODE selects how examples.parl_math_v2.tool.assign_task is routed:
#   frozen (default): --sglang-config carves a separate 'subagent' SGLang
#                     model from the colocate rollout pool, frozen at the
#                     SFT hf_checkpoint and excluded from RL weight updates.
#   shared           : skip --sglang-config; subagent shares the live
#                     policy router (= pre-frozen-engine baseline, used
#                     as ablation control).
SUBAGENT_MODE=${SUBAGENT_MODE:-frozen}
if [ "$SUBAGENT_MODE" = "frozen" ]; then
   SGLANG_EXTRA_ARGS=(--sglang-config examples/parl_math_v2/sglang_config_30B_A3B_2node.yaml)
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
   "${SGLANG_EXTRA_ARGS[@]}" \
   --extra-args "${EXTRA_ARGS[*]}"
