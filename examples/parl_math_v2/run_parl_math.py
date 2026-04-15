"""PARL v2 launcher (retool_v2 style).

Wraps train.py with the PARL v2 args (orchestrator with `consult_solvers`
spawn-as-tool). Mirrors examples/retool_v2/run_retool_multi_turn.py — the
.sh launchers in this folder are thin shells that invoke this script.

NOTE on PYTHONPATH: parl_math_v2 only lives in the dev tree
(/workspace/miles/examples/...), not in the baked-in /root/miles. To make
`import examples.parl_math_v2.*` resolve correctly we pass an absolute
train_script under the dev tree — `python3 <abs>/train.py` puts that
directory at sys.path[0], shadowing both /root/miles and Megatron's own
`examples` package.
"""
import os
from dataclasses import dataclass, field
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

WANDB_PROJECT = "miles-dev-multi-agent"

DEV_REPO_DIR = "/workspace/miles"

_MODEL_DEFAULTS = {
    "qwen3-4B": {
        "hf_checkpoint": f"{DEV_REPO_DIR}/MODEL/Qwen3-4B",
        "ref_load": f"{DEV_REPO_DIR}/MODEL/Qwen3-4B_torch_dist",
        "megatron_model_type": "qwen3-4B",
        "tensor_model_parallel_size": 2,
        "rollout_num_gpus_per_engine": 2,
    },
    "qwen3-0.6B": {
        "hf_checkpoint": f"{DEV_REPO_DIR}/MODEL/Qwen3-0.6B",
        "ref_load": f"{DEV_REPO_DIR}/MODEL/Qwen3-0.6B_torch_dist",
        "megatron_model_type": "qwen3-0.6B",
        "tensor_model_parallel_size": 1,
        "rollout_num_gpus_per_engine": 1,
    },
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = field(default_factory=U.create_run_id)
    hardware: Literal["H100", "GB200", "GB300"] = "H100"
    num_gpus_per_node: int | None = None
    model: Literal["qwen3-4B", "qwen3-0.6B"] = "qwen3-0.6B"
    save_path: str = ""
    prompt_data: str = f"{DEV_REPO_DIR}/DATA/dapo-math-17k/dapo-math-17k.jsonl"
    eval_prompt_data: str = f"{DEV_REPO_DIR}/DATA/aime-2024/aime-2024.jsonl"
    generate_max_turns: int = 6
    rollout_max_context_len: int = 32768
    rollout_max_response_len: int = 4096
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 64
    num_rollout: int = 500
    entropy_coef: float = 0.001
    sglang_router_ip: str = "127.0.0.1"
    sglang_router_port: int = 18765
    # empty string means "use default for the selected model"
    hf_checkpoint: str = ""
    ref_load: str = ""
    megatron_model_type: str = ""
    tensor_model_parallel_size: int = 0
    rollout_num_gpus_per_engine: int = 0
    extra_args: str = ""

    def __post_init__(self):
        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]
        defaults = _MODEL_DEFAULTS[self.model]
        self.hf_checkpoint = self.hf_checkpoint or defaults["hf_checkpoint"]
        self.ref_load = self.ref_load or defaults["ref_load"]
        self.megatron_model_type = self.megatron_model_type or defaults["megatron_model_type"]
        self.tensor_model_parallel_size = self.tensor_model_parallel_size or defaults["tensor_model_parallel_size"]
        self.rollout_num_gpus_per_engine = (
            self.rollout_num_gpus_per_engine or defaults["rollout_num_gpus_per_engine"]
        )
        if not self.save_path:
            self.save_path = f"{DEV_REPO_DIR}/saves/{os.path.basename(self.hf_checkpoint)}-parl-v2/{self.run_id}"


def _get_wandb_args(args: ScriptArgs) -> str:
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
    return (
        "--use-wandb "
        f"--wandb-project {WANDB_PROJECT} "
        f"--wandb-group {args.model}-parl-v2-math "
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
        f"--load {args.save_path} "
        f"--save {args.save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 50} "
    )

    custom_args = (
        "--custom-generate-function-path examples.parl_math_v2.generate.generate "
        "--generate-tool-specs-path examples.parl_math_v2.tool.tool_specs "
        "--generate-execute-tool-function-path examples.parl_math_v2.tool.execute_tool "
        "--generate-tool-call-parser qwen25 "
        f"--generate-max-turns {args.generate_max_turns} "
        "--log-multi-turn "
        "--custom-rm-path examples.parl_math_v2.reward.reward_func "
    )

    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--label-key label "
        "--rollout-shuffle "
        "--reward-key score "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-context-len {args.rollout_max_context_len} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
        f"--sglang-router-ip {args.sglang_router_ip} "
        f"--sglang-router-port {args.sglang_router_port} "
    )

    eval_args = ""
    if args.mode != "debug_minimal":
        eval_args = (
            "--eval-interval 20 "
            f"--eval-prompt-data aime {args.eval_prompt_data} "
            "--n-samples-per-eval-prompt 4 "
            f"--eval-max-response-len {args.rollout_max_response_len} "
            f"--eval-max-context-len {args.rollout_max_context_len} "
            "--eval-top-p 1 "
            "--log-passrate "
        )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        f"--entropy-coef {args.entropy_coef} "
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
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        "--sglang-mem-fraction-static 0.7 "
    )

    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
    )

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{_get_wandb_args(args)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{custom_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=megatron_model_type,
        # Absolute dev-tree train.py: ensures sys.path[0]=/workspace/miles so
        # `examples.parl_math_v2.*` imports resolve to the dev copy and not
        # the baked-in /root/miles or Megatron's own examples package.
        train_script=f"{DEV_REPO_DIR}/train.py",
        extra_env_vars={
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
            "MILES_SGLANG_ROUTER_IP": args.sglang_router_ip,
            "MILES_SGLANG_ROUTER_PORT": str(args.sglang_router_port),
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
