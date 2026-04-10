#!/bin/bash

# Small Qwen3-4B run for the retool_v2 multi-turn tool-calling example.
# Thin wrapper around examples/retool_v2/run_retool_multi_turn.py with
# overrides for the local /workspace/miles layout.

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python




set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2}
NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export WANDB_API_KEY=${WANDB_API_KEY:-local-82cbbacfe8e3c0c527da528160bd76a1e85c9fea}
export WANDB_BASE_URL=${WANDB_BASE_URL:-http://110.76.27.132}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

DATA_ROOT=${DATA_ROOT:-/workspace/miles/DATA}
MODEL_ROOT=${MODEL_ROOT:-/workspace/miles/MODEL}
MODE=${MODE:-debug_minimal}  # debug_minimal | normal
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
RUN_ID=${RUN_ID:-"run_$(date +%Y%m%d_%H%M%S)"}

MODEL_ARGS=(
   --model qwen3-4B
   --hf-checkpoint "${MODEL_ROOT}/qwen3-4b-sft-SGLang-RL"
   --ref-load "${MODEL_ROOT}/qwen3-4b-sft-SGLang-RL_torch_dist"
)

RUN_ARGS=(
   --mode "${MODE}"
   --run-id "${RUN_ID}"
   --save-path "${REPO_DIR}/saves/qwen3-4b-sft-SGLang-RL_retool_v2_multi_turn/${RUN_ID}"
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
   --generate-max-turns 8
)

cd "${REPO_DIR}"

python examples/retool_v2/run_retool_multi_turn.py \
   ${MODEL_ARGS[@]} \
   ${RUN_ARGS[@]} \
   ${PARALLEL_ARGS[@]} \
   ${DATA_ARGS[@]} \
   ${GENERATE_ARGS[@]}
