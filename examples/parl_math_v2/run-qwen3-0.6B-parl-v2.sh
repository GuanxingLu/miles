#!/bin/bash

# PARL v2 debug run on Qwen3-0.6B (4x small GPUs).
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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export WANDB_API_KEY=${WANDB_API_KEY:-local-82cbbacfe8e3c0c527da528160bd76a1e85c9fea}
export WANDB_BASE_URL=${WANDB_BASE_URL:-http://33.180.4.104}
# Default to offline wandb so the debug host (which can't reach the remote
# wandb URL above) doesn't block on wandb.init(). Set WANDB_MODE=online to
# re-enable upload when running on the prod box.
export WANDB_MODE=${WANDB_MODE:-offline}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

DATA_ROOT=${DATA_ROOT:-/workspace/miles/DATA}
MODEL_ROOT=${MODEL_ROOT:-/workspace/miles/MODEL}
MODE=${MODE:-normal}  # debug_minimal | normal
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
RUN_ID=${RUN_ID:-"run_$(date +%Y%m%d_%H%M%S)"}

MODEL_ARGS=(
   --model qwen3-0.6B
   --hf-checkpoint "${MODEL_ROOT}/Qwen3-0.6B"
   --ref-load "${MODEL_ROOT}/Qwen3-0.6B_torch_dist"
)

RUN_ARGS=(
   --mode "${MODE}"
   --run-id "${RUN_ID}"
   --save-path "${REPO_DIR}/saves/Qwen3-0.6B-parl-v2/${RUN_ID}"
)

PARALLEL_ARGS=(
   --num-gpus-per-node "${NUM_GPUS}"
   --tensor-model-parallel-size 1
   --rollout-num-gpus-per-engine 1
)

DATA_ARGS=(
   --prompt-data "${DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl"
   --eval-prompt-data "${DATA_ROOT}/aime-2024/aime-2024.jsonl"
)

GENERATE_ARGS=(
   --generate-max-turns 6
)

cd "${REPO_DIR}"

# Host machine occupies the default Ray ports (6379 redis, 8265). Bring up
# Ray ourselves on alternate ports and let run_parl_math.py reuse it via
# MILES_SCRIPT_EXTERNAL_RAY + RAY_ADDRESS.
RAY_GCS_PORT=${RAY_GCS_PORT:-26379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-28265}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --port ${RAY_GCS_PORT} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=${RAY_DASHBOARD_PORT}
export RAY_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}"
export MILES_SCRIPT_EXTERNAL_RAY=1

python examples/parl_math_v2/run_parl_math.py \
   ${MODEL_ARGS[@]} \
   ${RUN_ARGS[@]} \
   ${PARALLEL_ARGS[@]} \
   ${DATA_ARGS[@]} \
   ${GENERATE_ARGS[@]}
