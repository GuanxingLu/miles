"""
Tool executor for Claude Code Agent.

This module implements the execution environment for Claude Code tools,
mimicking the CLI interface that Claude Code uses.
"""

import asyncio
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class ToolExecutor:
    """Executes Claude Code tools in a sandboxed environment"""

    def __init__(self, workspace_dir: str = None, timeout: int = 120):
        """
        Initialize tool executor

        Args:
            workspace_dir: Working directory for file operations
            timeout: Timeout for command execution (seconds)
        """
        self.workspace_dir = workspace_dir or tempfile.mkdtemp(prefix="claude_code_workspace_")
        self.timeout = timeout
        self.max_output_length = 30000  # Match Claude Code's output limit

        # Ensure workspace exists
        Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)

    def get_tool_specs(self) -> list[dict[str, Any]]:
        """Return Claude Code tool specifications"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute a bash command in the workspace directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "The bash command to execute"}
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to the file to read"}
                        },
                        "required": ["file_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "description": "Write content to a file (creates or overwrites)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to the file to write"},
                            "content": {"type": "string", "description": "Content to write to the file"},
                        },
                        "required": ["file_path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit",
                    "description": "Edit a file by replacing old_string with new_string",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to the file to edit"},
                            "old_string": {"type": "string", "description": "String to replace"},
                            "new_string": {"type": "string", "description": "Replacement string"},
                        },
                        "required": ["file_path", "old_string", "new_string"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "grep",
                    "description": "Search for pattern in files (uses ripgrep)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "The regex pattern to search for"},
                            "path": {
                                "type": "string",
                                "description": "Directory or file to search in (default: workspace root)",
                            },
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "glob",
                    "description": "Find files matching a glob pattern",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.py')"},
                            "path": {
                                "type": "string",
                                "description": "Directory to search in (default: workspace root)",
                            },
                        },
                        "required": ["pattern"],
                    },
                },
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a tool and return the result

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result as string
        """
        try:
            if tool_name == "bash":
                return await self._execute_bash(arguments)
            elif tool_name == "read":
                return await self._execute_read(arguments)
            elif tool_name == "write":
                return await self._execute_write(arguments)
            elif tool_name == "edit":
                return await self._execute_edit(arguments)
            elif tool_name == "grep":
                return await self._execute_grep(arguments)
            elif tool_name == "glob":
                return await self._execute_glob(arguments)
            else:
                return f"Error: Unknown tool '{tool_name}'"
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    async def _execute_bash(self, args: dict) -> str:
        """Execute bash command"""
        command = args.get("command", "")
        if not command:
            return "Error: No command provided"

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_dir,
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
                output = stdout.decode("utf-8", errors="replace")
                error = stderr.decode("utf-8", errors="replace")

                result = ""
                if output:
                    result += output
                if error:
                    result += f"\nSTDERR:\n{error}"

                # Truncate if too long
                if len(result) > self.max_output_length:
                    result = result[: self.max_output_length] + "\n... (output truncated)"

                return result if result else "(command completed with no output)"

            except asyncio.TimeoutError:
                process.kill()
                return f"Error: Command timed out after {self.timeout} seconds"

        except Exception as e:
            return f"Error: {str(e)}"

    async def _execute_read(self, args: dict) -> str:
        """Read file contents"""
        file_path = args.get("file_path", "")
        if not file_path:
            return "Error: No file_path provided"

        full_path = os.path.join(self.workspace_dir, file_path)

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Add line numbers (like cat -n)
            lines = content.split("\n")
            numbered_lines = [f"{i+1:6d}\t{line}" for i, line in enumerate(lines)]
            result = "\n".join(numbered_lines)

            # Truncate if too long
            if len(result) > self.max_output_length:
                result = result[: self.max_output_length] + "\n... (output truncated)"

            return result

        except FileNotFoundError:
            return f"Error: File not found: {file_path}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _execute_write(self, args: dict) -> str:
        """Write content to file"""
        file_path = args.get("file_path", "")
        content = args.get("content", "")

        if not file_path:
            return "Error: No file_path provided"

        full_path = os.path.join(self.workspace_dir, file_path)

        try:
            # Create parent directories if needed
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

            return f"Successfully wrote {len(content)} characters to {file_path}"

        except Exception as e:
            return f"Error: {str(e)}"

    async def _execute_edit(self, args: dict) -> str:
        """Edit file by replacing old_string with new_string"""
        file_path = args.get("file_path", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")

        if not file_path:
            return "Error: No file_path provided"
        if not old_string:
            return "Error: No old_string provided"

        full_path = os.path.join(self.workspace_dir, file_path)

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check if old_string exists
            if old_string not in content:
                return f"Error: old_string not found in {file_path}"

            # Check if old_string appears multiple times
            count = content.count(old_string)
            if count > 1:
                return f"Error: old_string appears {count} times in {file_path}. Please provide a more specific string."

            # Replace
            new_content = content.replace(old_string, new_string)

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return f"Successfully replaced text in {file_path}"

        except FileNotFoundError:
            return f"Error: File not found: {file_path}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _execute_grep(self, args: dict) -> str:
        """Search for pattern in files"""
        pattern = args.get("pattern", "")
        search_path = args.get("path", ".")

        if not pattern:
            return "Error: No pattern provided"

        full_path = os.path.join(self.workspace_dir, search_path)

        try:
            # Use ripgrep if available, otherwise use grep
            cmd = f"rg -n '{pattern}' {full_path} 2>/dev/null || grep -rn '{pattern}' {full_path}"
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_dir,
            )

            stdout, stderr = await process.communicate()
            output = stdout.decode("utf-8", errors="replace")

            if not output:
                return f"No matches found for pattern: {pattern}"

            # Truncate if too long
            if len(output) > self.max_output_length:
                output = output[: self.max_output_length] + "\n... (output truncated)"

            return output

        except Exception as e:
            return f"Error: {str(e)}"

    async def _execute_glob(self, args: dict) -> str:
        """Find files matching glob pattern"""
        pattern = args.get("pattern", "")
        search_path = args.get("path", ".")

        if not pattern:
            return "Error: No pattern provided"

        full_path = os.path.join(self.workspace_dir, search_path)

        try:
            # Use find with glob pattern
            cmd = f"find {full_path} -name '{pattern}' -type f"
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_dir,
            )

            stdout, stderr = await process.communicate()
            output = stdout.decode("utf-8", errors="replace")

            if not output.strip():
                return f"No files found matching pattern: {pattern}"

            # Make paths relative to workspace
            lines = output.strip().split("\n")
            rel_paths = [os.path.relpath(line, self.workspace_dir) for line in lines if line]
            result = "\n".join(sorted(rel_paths))

            # Truncate if too long
            if len(result) > self.max_output_length:
                result = result[: self.max_output_length] + "\n... (output truncated)"

            return result

        except Exception as e:
            return f"Error: {str(e)}"

    def cleanup(self):
        """Clean up workspace directory"""
        import shutil

        if self.workspace_dir and os.path.exists(self.workspace_dir):
            shutil.rmtree(self.workspace_dir, ignore_errors=True)
