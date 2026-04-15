#!/bin/bash

# PARL run on Qwen3-4B (math task).
# Orchestrator (trainable) + N frozen solvers, composite annealed reward.

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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# Training talks only to local services; drop all proxies to avoid httpx[socks] issues and localhost routing through clash.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export WANDB_API_KEY=${WANDB_API_KEY:-local-82cbbacfe8e3c0c527da528160bd76a1e85c9fea}
export WANDB_BASE_URL=${WANDB_BASE_URL:-http://33.180.4.104}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

DATA_ROOT=${DATA_ROOT:-/workspace/miles/DATA}
MODEL_ROOT=${MODEL_ROOT:-/workspace/miles/MODEL}
MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM/}
RUN_ID=${RUN_ID:-"run_$(date +%Y%m%d_%H%M%S)"}
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')

source "${REPO_DIR}/scripts/models/qwen3-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3-4B
   --ref-load ${MODEL_ROOT}/Qwen3-4B_torch_dist
   --load ${REPO_DIR}/saves/Qwen3-4B-parl/${RUN_ID}
   --save ${REPO_DIR}/saves/Qwen3-4B-parl/${RUN_ID}
   --save-interval 50
)

ROLLOUT_ARGS=(
   --custom-generate-function-path examples.parl_math.rollout_with_parl.generate_with_parl
   --prompt-data ${DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   # --rm-type dapo
   --reward-key score
   --num-rollout 500
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-context-len 16384
   --rollout-max-response-len 4096
   --rollout-temperature 1
   --global-batch-size 128
   --balance-data
)

EVAL_ARGS=(
   # eval disabled for smoke; re-enable to measure AIME pass-rate.
   # --eval-interval 20
   # --eval-prompt-data aime ${DATA_ROOT}/aime-2024/aime-2024.jsonl
   # --n-samples-per-eval-prompt 8
   # --eval-max-response-len 8192
   # --eval-top-p 1
   # --log-passrate
)

PARALLEL_ARGS=(
   --num-gpus-per-node ${NUM_GPUS}
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-dev-multi-agent
   --wandb-group qwen3-4B-parl-math
   --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --port 26379 \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=28265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"http_proxy\": \"\",
    \"https_proxy\": \"\",
    \"all_proxy\": \"\",
    \"HTTP_PROXY\": \"\",
    \"HTTPS_PROXY\": \"\",
    \"ALL_PROXY\": \"\"
  }
}"

cd "${REPO_DIR}"

ray job submit --address="http://127.0.0.1:28265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PARALLEL_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
