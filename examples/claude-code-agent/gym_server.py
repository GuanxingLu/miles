"""
Claude Code Gym Server

A FastAPI server that provides a Gym environment for training Claude Code agents.
This server receives task requests from Miles training, runs the agent with tool
execution, and returns trajectories with rewards.

Architecture:
    Miles Training → HTTP Request → Gym Server → Agent Controller → Tool Executor
                   ← HTTP Response ← Task Result ←
"""

import argparse
import asyncio
import json
import logging
import os
import tempfile
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Import tool executor
from tool_executor import ToolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Claude Code Gym", version="1.0.0")


class AgentController:
    """Controls the agent's interaction loop with tools"""

    def __init__(
        self,
        sglang_url: str,
        max_turns: int = 16,
        max_tool_calls: int = 20,
        timeout: int = 600,
    ):
        self.sglang_url = sglang_url
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.timeout = timeout

    async def run_task(
        self,
        task: dict[str, Any],
        sampling_params: dict[str, Any],
        workspace_dir: str,
    ) -> dict[str, Any]:
        """
        Run a task with the agent in the given workspace.

        Args:
            task: Task specification with prompt and success criteria
            sampling_params: Model sampling parameters
            workspace_dir: Working directory for file operations

        Returns:
            Dictionary with:
                - messages: Conversation history
                - success: Whether task was completed successfully
                - task_status: 'completed', 'truncated', 'error', 'timeout'
                - agent_metrics: Performance metrics
                - evaluation_details: Detailed evaluation results
        """
        start_time = time.time()

        # Initialize tool executor
        tool_executor = ToolExecutor(workspace_dir=workspace_dir)

        # Setup workspace from task specification
        await self._setup_workspace(task, workspace_dir)

        # Build initial messages
        messages = [
            {"role": "system", "content": self._build_system_prompt(tool_executor)},
            {"role": "user", "content": task["prompt"]},
        ]

        # Agent metrics
        metrics = {
            "turns": 0,
            "tool_calls": 0,
            "model_query_time_sum": 0.0,
            "env_execution_time_sum": 0.0,
        }

        task_status = "running"
        final_answer = None

        try:
            # Main interaction loop
            for turn in range(self.max_turns):
                metrics["turns"] = turn + 1

                # Query model
                model_start = time.time()
                response = await self._query_model(messages, sampling_params)
                metrics["model_query_time_sum"] += time.time() - model_start

                if response is None:
                    task_status = "error"
                    break

                # Add assistant response to messages
                messages.append({"role": "assistant", "content": response})

                # Parse tool calls from response
                tool_calls = self._parse_tool_calls(response)

                if not tool_calls:
                    # No tool calls, check for final answer
                    if self._is_final_answer(response):
                        final_answer = response
                        task_status = "completed"
                        break
                    else:
                        # Agent didn't call tools or provide answer, continue
                        continue

                # Execute tool calls
                tool_results = []
                for tool_call in tool_calls:
                    if metrics["tool_calls"] >= self.max_tool_calls:
                        task_status = "truncated"
                        break

                    env_start = time.time()
                    result = await tool_executor.execute_tool(
                        tool_call["name"], tool_call["arguments"]
                    )
                    metrics["env_execution_time_sum"] += time.time() - env_start
                    metrics["tool_calls"] += 1

                    tool_results.append(
                        {"tool": tool_call["name"], "result": result}
                    )

                if task_status == "truncated":
                    break

                # Add tool results to messages
                tool_result_text = self._format_tool_results(tool_results)
                messages.append({"role": "tool", "content": tool_result_text})

            # If we exited loop without completion
            if task_status == "running":
                task_status = "truncated"

        except asyncio.TimeoutError:
            task_status = "timeout"
            logger.warning(f"Task timed out after {self.timeout}s")
        except Exception as e:
            task_status = "error"
            logger.error(f"Error during task execution: {e}", exc_info=True)

        # Calculate total time
        total_time = time.time() - start_time
        metrics["total_time"] = total_time

        # Evaluate task success
        success, eval_details = await self._evaluate_task(
            task, workspace_dir, messages, final_answer
        )

        # Cleanup
        tool_executor.cleanup()

        return {
            "messages": messages,
            "success": success,
            "task_status": task_status,
            "agent_metrics": metrics,
            "evaluation_details": eval_details,
        }

    def _build_system_prompt(self, tool_executor: ToolExecutor) -> str:
        """Build system prompt with tool descriptions"""
        tools = tool_executor.get_tool_specs()

        tool_descriptions = []
        for tool in tools:
            func = tool["function"]
            tool_descriptions.append(
                f"- {func['name']}: {func['description']}"
            )

        prompt = f"""You are Claude Code, an AI assistant that helps with software engineering tasks.

You have access to the following tools:
{chr(10).join(tool_descriptions)}

To use a tool, format your response as:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}
</tool_call>

You can call multiple tools in sequence. Tool results will be provided in <tool_result> tags.

When you have completed the task, provide your final answer without any tool calls.
"""
        return prompt

    async def _setup_workspace(self, task: dict, workspace_dir: str):
        """Setup workspace with initial files from task specification"""
        workspace_setup = task.get("workspace_setup", {})

        # Create initial files
        files = workspace_setup.get("files", {})
        for file_path, content in files.items():
            full_path = os.path.join(workspace_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)

        # Clone repository if specified
        repo_url = workspace_setup.get("repo")
        if repo_url:
            # TODO: Implement git clone
            pass

    async def _query_model(
        self, messages: list[dict], sampling_params: dict
    ) -> str | None:
        """Query the language model via SGLang"""
        try:
            import aiohttp

            # Convert messages to text prompt
            prompt = self._messages_to_prompt(messages)

            # Call SGLang /generate endpoint
            async with aiohttp.ClientSession() as session:
                payload = {
                    "text": prompt,
                    "sampling_params": sampling_params,
                }

                async with session.post(
                    f"{self.sglang_url}/generate", json=payload, timeout=120
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"SGLang error: {resp.status}")
                        return None

                    data = await resp.json()
                    return data.get("text", "")

        except Exception as e:
            logger.error(f"Error querying model: {e}")
            return None

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """Convert messages to text prompt"""
        # Simple implementation - customize based on your model's chat template
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            elif role == "tool":
                prompt_parts.append(f"<|im_start|>tool\n{content}<|im_end|>")

        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)

    def _parse_tool_calls(self, response: str) -> list[dict]:
        """Parse tool calls from model response"""
        import re

        tool_calls = []

        # Find all <tool_call>...</tool_call> blocks
        pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            try:
                # Parse JSON
                tool_call = json.loads(match)
                if "name" in tool_call:
                    tool_calls.append(
                        {
                            "name": tool_call["name"],
                            "arguments": tool_call.get("arguments", {}),
                        }
                    )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call: {match}")

        return tool_calls

    def _is_final_answer(self, response: str) -> bool:
        """Check if response contains a final answer"""
        # Simple heuristic: if no tool calls and has substantial content
        if "<tool_call>" in response:
            return False

        # Check for common completion indicators
        completion_indicators = [
            "task completed",
            "done",
            "finished",
            "successfully",
        ]

        response_lower = response.lower()
        return any(indicator in response_lower for indicator in completion_indicators)

    def _format_tool_results(self, tool_results: list[dict]) -> str:
        """Format tool results for inclusion in messages"""
        formatted = []
        for result in tool_results:
            formatted.append(
                f"<tool_result tool=\"{result['tool']}\">\n"
                f"{result['result']}\n"
                f"</tool_result>"
            )
        return "\n".join(formatted)

    async def _evaluate_task(
        self,
        task: dict,
        workspace_dir: str,
        messages: list[dict],
        final_answer: str | None,
    ) -> tuple[bool, dict]:
        """
        Evaluate whether the task was completed successfully.

        Returns:
            Tuple of (success: bool, details: dict)
        """
        success_criteria = task.get("success_criteria", {})

        # TODO: Implement sophisticated evaluation logic
        # For now, simple heuristics:

        details = {"criteria_met": {}, "progress": 0.0}

        # Check if tests pass (if specified)
        if success_criteria.get("tests_pass"):
            # Run tests in workspace
            # TODO: Implement test running
            pass

        # Check if specific files were modified
        if "files_modified" in success_criteria:
            # TODO: Check file modifications
            pass

        # Simple heuristic: if agent provided final answer, partial success
        if final_answer:
            success = True
            details["progress"] = 1.0
        else:
            success = False
            details["progress"] = 0.3  # Partial credit for attempting

        return success, details


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "claude-code-gym"}


@app.post("/run")
async def run_task(request: dict):
    """
    Main endpoint to run a task with the agent.

    Request body:
        {
            "task": {
                "prompt": "Task description",
                "task_type": "bug_fix|feature|exploration|...",
                "workspace_setup": {...},
                "success_criteria": {...}
            },
            "sampling_params": {...},
            "sglang_url": "http://...",
            "max_turns": 16,
            "max_tool_calls": 20
        }

    Response:
        {
            "messages": [...],
            "success": true/false,
            "task_status": "completed|truncated|error|timeout",
            "agent_metrics": {...},
            "evaluation_details": {...}
        }
    """
    try:
        # Extract request parameters
        task = request.get("task")
        sampling_params = request.get("sampling_params", {})
        sglang_url = request.get("sglang_url")
        max_turns = request.get("max_turns", 16)
        max_tool_calls = request.get("max_tool_calls", 20)

        if not task or not sglang_url:
            raise HTTPException(
                status_code=400, detail="Missing required fields: task, sglang_url"
            )

        # Create temporary workspace
        workspace_dir = tempfile.mkdtemp(prefix="claude_code_workspace_")

        try:
            # Initialize agent controller
            controller = AgentController(
                sglang_url=sglang_url,
                max_turns=max_turns,
                max_tool_calls=max_tool_calls,
            )

            # Run the task
            result = await controller.run_task(task, sampling_params, workspace_dir)

            return JSONResponse(content=result)

        finally:
            # Cleanup workspace
            import shutil

            if os.path.exists(workspace_dir):
                shutil.rmtree(workspace_dir, ignore_errors=True)

    except Exception as e:
        logger.error(f"Error in /run endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description="Claude Code Gym Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=12000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    logger.info(f"Starting Claude Code Gym server on {args.host}:{args.port}")

    uvicorn.run(
        "gym_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
