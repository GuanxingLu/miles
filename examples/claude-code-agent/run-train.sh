#!/bin/bash

# Claude Code Agent Training Script
# This script trains an agent to perform software engineering tasks using RL

# Cleanup from previous runs
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# Environment setup
export PYTHONBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Script directory
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Run ID for tracking
RUN_ID=${RUN_ID:-"claude_code_$(date +%Y%m%d_%H%M%S)"}

# IMPORTANT: Set Claude Code Gym URL
# This should point to your Claude Code Gym server
export CLAUDE_CODE_GYM_URL=${CLAUDE_CODE_GYM_URL:-"http://localhost:12000"}
echo "Using Claude Code Gym at: $CLAUDE_CODE_GYM_URL"

# Verify Gym is reachable
if ! curl -s -f "${CLAUDE_CODE_GYM_URL}/health" > /dev/null 2>&1; then
    echo "WARNING: Cannot reach Claude Code Gym at ${CLAUDE_CODE_GYM_URL}"
    echo "Please start the Gym server first: python examples/claude-code-agent/gym_server.py"
    echo "Or set CLAUDE_CODE_GYM_URL to the correct address"
    # Uncomment to exit on error:
    # exit 1
fi

# Model checkpoint configuration
CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B-Instruct  # Change to your model path
   --load /root/Qwen3-4B-Instruct
   --ref-load /root/Qwen3-4B-Instruct
   --save /root/shared_data/${RUN_ID}/checkpoints
)

# LoRA configuration (optional, for efficient fine-tuning)
# Uncomment to enable LoRA training
# LORA_ARGS=(
#    --lora-rank 32
#    --lora-alpha 32
#    --target-modules all-linear
#    --lora-sync-from-tensor
# )

# Rollout configuration - Claude Code Agent specific
ROLLOUT_ARGS=(
   # Data settings
   --prompt-data /root/claude-code-tasks.jsonl  # Your task dataset
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   # Rollout batch settings
   --num-rollout 50                # Number of rollouts per epoch
   --rollout-batch-size 16         # Batch size for rollout generation
   --n-samples-per-prompt 4        # Number of trajectories per task (for GRPO)
   --global-batch-size 64          # Global training batch size

   # Generation settings
   --rollout-max-response-len 8192 # Max tokens for agent trajectory
   --rollout-temperature 1.0       # Sampling temperature

   # Claude Code Agent specific settings
   --rollout-type generate         # Use custom generate function
   --rollout-max-turns 16          # Max conversation turns
   --rollout-max-tool-calls 20     # Max tool executions per task

   # Custom module - IMPORTANT!
   --custom-generate-module examples.claude-code-agent.generate_with_claude_code
   --rollout-global-dataset        # Required for custom generate function
)

# GRPO (Group Relative Policy Optimization) configuration
GRPO_ARGS=(
   --use-kl-loss
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# Optimizer settings
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6                       # Lower learning rate for stability
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# Weights & Biases logging (optional)
if [ -z "${WANDB_API_KEY}" ]; then
   WANDB_ARGS=()
else
   WANDB_ARGS=(
      --use-wandb
      --wandb-project claude-code-agent
      --wandb-group ${RUN_ID}
      --wandb-key "${WANDB_API_KEY}"
   )
fi

# SGLang rollout engine configuration
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --sglang-decode-log-interval 1000
   --sglang-chunked-prefill-size 4096
   --sglang-attention-backend fa3
)

# Training backend configuration
TRAIN_BACKEND_ARGS=(
   --train-backend fsdp
   --update-weight-buffer-size 536870912
   --gradient-checkpointing
   --attn-implementation flash_attention_3
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)

# Performance optimization
PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

# Miscellaneous settings
MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node ${NUM_GPUS}
   --colocate                      # Colocate actor and rollout on same GPUs
   --use-fault-tolerance           # Enable fault tolerance
   --dump-details /root/shared_data/${RUN_ID}/dump_details
   --num-epochs 3                  # Number of training epochs
)

# Launch Ray cluster
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats

# Runtime environment for Ray
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}/../..\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"CLAUDE_CODE_GYM_URL\": \"${CLAUDE_CODE_GYM_URL}\"
  }
}"

# Launch training
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   "${CKPT_ARGS[@]}" \
   "${LORA_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${TRAIN_BACKEND_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"
