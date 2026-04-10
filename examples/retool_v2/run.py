import os
from dataclasses import dataclass, field
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

WANDB_PROJECT = "miles-dev-retool-v2"
WANDB_GROUP = "baseline"


_MODEL_DEFAULTS = {
    "qwen3-4B-sft": {
        "hf_checkpoint": "/root/font-info/qwen3-4b-sft",
        "ref_load": "/root/font-info/qwen3-4b-sft_torch_dist",
        "megatron_model_type": "qwen3-4B",
        "tensor_model_parallel_size": 2,
        "rotary_base": 5000000,
    },
    "qwen3-4B": {
        "hf_checkpoint": "/root/models/Qwen3-4B",
        "ref_load": "/root/models/Qwen3-4B_torch_dist",
        "megatron_model_type": "qwen3-4B",
        "tensor_model_parallel_size": 2,
        "rotary_base": None,
    },
    "qwen3-0.6B": {
        "hf_checkpoint": "/workspace/miles/MODEL/Qwen3-0.6B",
        "ref_load": "/workspace/miles/MODEL/Qwen3-0.6B_torch_dist",
        "megatron_model_type": "qwen3-0.6B",
        "tensor_model_parallel_size": 1,
        "rotary_base": None,
    },
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = field(default_factory=U.create_run_id)
    hardware: Literal["H100", "GB200", "GB300"] = "H100"
    num_gpus_per_node: int | None = None
    model: Literal["qwen3-4B-sft", "qwen3-4B", "qwen3-0.6B"] = "qwen3-4B-sft"
    save_path: str = "/root/Qwen3-4B_miles/baseline"
    prompt_data: str = "/root/dapo-math-17k/dapo-math-17k.jsonl"
    eval_prompt_data: str = "/root/aime-2024/aime-2024.jsonl"
    rollout_num_gpus_per_engine: int = 2
    # empty string means "use default for the selected model"
    hf_checkpoint: str = ""
    ref_load: str = ""
    megatron_model_type: str = ""
    tensor_model_parallel_size: int = 0
    extra_args: str = ""

    def __post_init__(self):
        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]
        defaults = _MODEL_DEFAULTS[self.model]
        self.hf_checkpoint = self.hf_checkpoint or defaults["hf_checkpoint"]
        self.ref_load = self.ref_load or defaults["ref_load"]
        self.megatron_model_type = self.megatron_model_type or defaults["megatron_model_type"]
        self.tensor_model_parallel_size = self.tensor_model_parallel_size or defaults["tensor_model_parallel_size"]
        self._rotary_base = defaults["rotary_base"]


def _get_wandb_args() -> str:
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    return (
        "--use-wandb "
        f"--wandb-project {WANDB_PROJECT} "
        f"--wandb-group {WANDB_GROUP} "
        f"--wandb-key {WANDB_API_KEY} "
    )


def prepare(args: ScriptArgs):
    hf_dir = os.path.dirname(args.hf_checkpoint)
    U.convert_checkpoint(
        model_name=os.path.basename(args.hf_checkpoint),
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        hf_checkpoint=args.hf_checkpoint,
        dir_dst=hf_dir,
    )


def execute(args: ScriptArgs):
    megatron_model_type = args.megatron_model_type

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        f"--save {args.save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 1000} "
        f"{f'--rotary-base {args._rotary_base} ' if args._rotary_base else ''}"
    )

    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type dapo "
        "--reward-key score "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
        "--balance-data "
    )

    eval_args = ""
    if args.mode != "debug_minimal":
        eval_args = (
            "--eval-interval 20 "
            f"--eval-prompt-data aime {args.eval_prompt_data} "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
        )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} " "--sglang-mem-fraction-static 0.7 "
    )

    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--log-passrate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{_get_wandb_args()} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=megatron_model_type,
        extra_env_vars={
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
            "PYTHONPATH": "/root/Megatron-LM/:/root/miles",
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
