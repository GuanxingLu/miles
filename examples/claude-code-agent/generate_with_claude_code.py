"""
Custom generation function for Claude Code Agent training.

This module implements the rollout generation logic that integrates with
an external Claude Code Gym environment (similar to SWE-Agent architecture).
"""

import json
import logging
import os
from argparse import Namespace
from collections.abc import Callable
from typing import Any

from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.sglang_rollout import GenerateState, eval_rollout
from miles.utils.async_utils import run
from miles.utils.http_utils import post
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def build_tokens_and_mask_from_messages(
    messages: list[dict],
    tokenizer,
) -> tuple[list[int], list[int], str, int]:
    """
    Build tokens and loss mask from conversation messages.

    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: The tokenizer to use

    Returns:
        Tuple of (all_tokens, loss_mask, response_text, response_length)
    """
    if not messages or len(messages) < 2:
        return [], [], "", 0

    # First 2 messages are prompt (system + user)
    prompt_msgs = messages[:2]
    # Rest are responses (assistant + tool results)
    response_msgs = messages[2:]

    # Tokenize prompt
    prompt_tokens = []
    for msg in prompt_msgs:
        content = msg.get("content", "")
        if content:
            prompt_tokens.extend(tokenizer(content, add_special_tokens=False)["input_ids"])

    # Tokenize responses with loss mask
    response_tokens = []
    loss_mask = []
    response_text_parts = []

    for msg in response_msgs:
        content = msg.get("content", "")
        if not content:
            continue

        tokens = tokenizer(content, add_special_tokens=False)["input_ids"]
        token_len = len(tokens)

        response_tokens.extend(tokens)
        response_text_parts.append(content)

        # Only compute loss on assistant responses, not tool results
        mask_val = 1 if msg.get("role") == "assistant" else 0
        loss_mask.extend([mask_val] * token_len)

    all_tokens = prompt_tokens + response_tokens
    response_text = "".join(response_text_parts)
    response_length = len(response_tokens)

    return all_tokens, loss_mask, response_text, response_length


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """
    Custom generation function for Claude Code Agent integration.

    Orchestrates the interaction with the external Claude Code Gym environment:
    1. Sends task prompt/metadata to Gym.
    2. Gym runs the agent with the model, executing tools and collecting trajectory.
    3. Receives execution trace (messages), metrics, and rewards.
    4. Formats data for Miles training format.

    Note: Performs in-place modification of `sample` for memory efficiency.

    Args:
        args: Training arguments
        sample: The sample to generate for
        sampling_params: Sampling parameters for the model

    Returns:
        The modified sample with generated response and reward
    """
    # Prepare request for Claude Code Gym /run endpoint
    request = {
        "task": {
            "prompt": sample.prompt,
            "task_type": sample.metadata.get("task_type", "general"),
            "workspace_setup": sample.metadata.get("workspace_setup", {}),
            "success_criteria": sample.metadata.get("success_criteria", {}),
        },
        "sampling_params": sampling_params,
        "sglang_url": f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1",
        "max_turns": getattr(args, "rollout_max_turns", 16),
        "max_tool_calls": getattr(args, "rollout_max_tool_calls", 20),
    }

    # Get Gym URL from environment
    gym_url = os.getenv("CLAUDE_CODE_GYM_URL", "http://localhost:12000")

    logger.debug(f"Sending request to Claude Code Gym: {gym_url}/run")

    # Call Gym environment
    response = await post(f"{gym_url}/run", request)

    # Extract results
    task_status = response.get("task_status", "unknown")
    success = response.get("success", False)
    messages = response.get("messages", [])
    agent_metrics = response.get("agent_metrics", {})

    logger.debug(
        f"Task status: {task_status}, success: {success}, "
        f"turns: {agent_metrics.get('turns', 0)}, "
        f"tool_calls: {agent_metrics.get('tool_calls', 0)}"
    )

    # Extract prompt from first 2 messages if available
    if len(messages) >= 2:
        sample.prompt = messages[:2]

    # Build tokens and loss mask from conversation
    state = GenerateState(args)
    tokens, loss_mask, response_text, response_length = build_tokens_and_mask_from_messages(
        messages=messages,
        tokenizer=state.tokenizer,
    )

    # Update sample
    sample.rollout_log_probs = None  # TODO: collect from SGLang if needed
    sample.tokens = tokens
    sample.loss_mask = loss_mask
    sample.response = response_text
    sample.response_length = response_length

    # Store metadata for reward calculation
    sample.metadata["success"] = success
    sample.metadata["task_status"] = task_status
    sample.metadata["agent_metrics"] = agent_metrics
    sample.metadata["messages"] = messages
    sample.metadata["evaluation_details"] = response.get("evaluation_details", {})

    # Set sample status based on task completion
    if task_status == "completed" and success:
        sample.status = Sample.Status.COMPLETED
    elif task_status in ("truncated", "timeout"):
        sample.status = Sample.Status.TRUNCATED
    elif task_status in ("error", "failed"):
        sample.status = Sample.Status.ABORTED
    else:
        # Partial success or unknown status
        sample.status = Sample.Status.COMPLETED if success else Sample.Status.TRUNCATED

    return sample


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """
    Reward function for Claude Code Agent.

    Reward structure:
    - Task success: +1.0 (primary signal)
    - Partial progress: +0.0 to +0.5 (based on evaluation metrics)
    - Tool usage efficiency: -0.01 per tool call (capped at -0.3)
    - Error penalty: -0.5 for errors/failures

    Args:
        args: Training arguments
        sample: The sample with metadata containing evaluation results

    Returns:
        Reward value between -0.5 and 1.0
    """
    reward = 0.0

    # Main reward: task success
    success = sample.metadata.get("success", False)
    if success:
        reward += 1.0
    else:
        # Partial credit based on progress
        eval_details = sample.metadata.get("evaluation_details", {})
        progress = eval_details.get("progress", 0.0)  # 0.0 to 1.0
        reward += progress * 0.5

    # Tool usage efficiency penalty (encourage concise solutions)
    agent_metrics = sample.metadata.get("agent_metrics", {})
    tool_calls = agent_metrics.get("tool_calls", 0)
    tool_penalty = -0.01 * tool_calls
    reward += max(tool_penalty, -0.3)  # Cap at -0.3

    # Error/failure penalty
    task_status = sample.metadata.get("task_status", "unknown")
    if task_status in ("error", "failed"):
        reward -= 0.5

    # Ensure reward is in reasonable range
    reward = max(-0.5, min(1.0, reward))

    logger.debug(
        f"Reward calculation: success={success}, progress={eval_details.get('progress', 0.0)}, "
        f"tool_calls={tool_calls}, status={task_status}, final_reward={reward:.3f}"
    )

    return reward


def dynamic_filter(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """
    Filter out groups with aborted samples from training.

    This prevents training on failed/error trajectories which might
    teach the model bad behaviors.
    """
    has_aborted = any(sample.status == Sample.Status.ABORTED for sample in samples)
    if has_aborted:
        return DynamicFilterOutput(keep=False, reason="group_has_aborted")
    return DynamicFilterOutput(keep=True)


def aggregate_agent_metrics(samples: list[Sample]) -> dict:
    """
    Aggregate agent metrics across samples for logging.

    Tracks:
    - Turn counts and tool call counts
    - Time statistics (model query, env execution, total)
    - Success rate

    Args:
        samples: List of samples with agent_metrics in metadata

    Returns:
        Dictionary of aggregated metrics
    """
    metrics = {}

    all_metrics = []
    for sample in samples:
        if hasattr(sample, "metadata") and sample.metadata:
            agent_metrics = sample.metadata.get("agent_metrics", {})
            if agent_metrics:
                all_metrics.append(agent_metrics)

    if not all_metrics:
        return {}

    # Count metrics - mean and sum
    for key in ["turns", "tool_calls"]:
        values = [m.get(key, 0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)
            metrics[f"agent/{key}_sum"] = sum(values)

    # Time metrics - mean across rollouts
    for key in ["model_query_time_sum", "env_execution_time_sum", "total_time"]:
        values = [m.get(key, 0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)

    # Success rate
    successes = [sample.metadata.get("success", False) for sample in samples]
    metrics["agent/success_rate"] = sum(successes) / len(successes) if successes else 0.0

    return metrics


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """
    Custom rollout function that wraps sglang_rollout.generate_rollout_async
    and adds Claude Code agent metrics aggregation.
    """
    from miles.rollout.sglang_rollout import generate_rollout_async as base_generate_rollout_async

    rollout_output, aborted_samples = await base_generate_rollout_async(args, rollout_id, data_source)

    # Aggregate all samples for metrics
    all_samples = []
    for group in rollout_output.samples:
        if isinstance(group[0], list):
            for sample_list in group:
                all_samples.extend(sample_list)
        else:
            all_samples.extend(group)

    # Compute agent metrics
    agent_metrics = aggregate_agent_metrics(all_samples)

    # Merge with existing metrics
    metrics = rollout_output.metrics or {}
    metrics.update(agent_metrics)

    logger.info(f"Aggregated agent metrics for rollout {rollout_id}: {agent_metrics}")

    return RolloutFnTrainOutput(samples=rollout_output.samples, metrics=metrics), aborted_samples


def generate_rollout(
    args: Namespace, rollout_id: int, data_buffer: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """
    Main entry point for rollout generation.

    Args:
        args: Training arguments
        rollout_id: Rollout ID for deterministic data generation
        data_buffer: Data buffer to store generated samples
        evaluation: Whether this is evaluation or training rollout

    Returns:
        RolloutFnTrainOutput or RolloutFnEvalOutput with generated samples
    """
    output, aborted_samples = generate_abortable_samples(
        args, rollout_id, data_buffer.get_samples, evaluation=evaluation
    )
    data_buffer.add_samples(aborted_samples)
    return output


def generate_abortable_samples(
    args: Namespace,
    rollout_id: int,
    data_source: Callable[[int], list[list[Sample]]],
    evaluation: bool = False,
) -> tuple[Any, list[list[Sample]]]:
    """
    Generate samples with proper handling of aborted samples.
    """
    assert args.rollout_global_dataset
    if evaluation:
        return run(eval_rollout(args, rollout_id))
    return run(generate_rollout_async(args, rollout_id, data_source))
