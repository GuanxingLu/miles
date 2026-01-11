"""
Tool executor for Claude Code Agent.

This module implements the execution environment for Claude Code tools,
mimicking the CLI interface that Claude Code uses.
"""

import asyncio
import json
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
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information. Returns search results with titles, URLs, and snippets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query"},
                            "num_results": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": "Fetch content from a URL and convert to markdown. Useful for reading documentation, blog posts, or web pages.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "The URL to fetch"},
                            "extract_main_content": {
                                "type": "boolean",
                                "description": "Try to extract main content only (default: true)",
                            },
                        },
                        "required": ["url"],
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
            elif tool_name == "web_search":
                return await self._execute_web_search(arguments)
            elif tool_name == "web_fetch":
                return await self._execute_web_fetch(arguments)
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

    async def _execute_web_search(self, args: dict) -> str:
        """Search the web using SerpAPI"""
        query = args.get("query", "")
        num_results = args.get("num_results", 5)

        if not query:
            return "Error: No query provided"

        api_key = os.getenv("SEARCH_API_KEY")
        if not api_key:
            return "Error: SEARCH_API_KEY environment variable not set. Get API key from https://serpapi.com/"

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                params = {"q": query, "num": num_results, "api_key": api_key}
                async with session.get("https://serpapi.com/search", params=params, timeout=30) as resp:
                    if resp.status != 200:
                        return f"Error: Search API returned status {resp.status}"

                    data = await resp.json()
                    results = []
                    for item in data.get("organic_results", [])[:num_results]:
                        results.append(
                            f"**{item.get('title', 'No title')}**\n"
                            f"URL: {item.get('link', '')}\n"
                            f"{item.get('snippet', 'No description')}\n"
                        )
                    return "\n".join(results) if results else "No results found"

        except Exception as e:
            return f"Error performing web search: {str(e)}"

    async def _execute_web_fetch(self, args: dict) -> str:
        """Fetch content from a URL and convert to markdown"""
        url = args.get("url", "")
        extract_main = args.get("extract_main_content", True)

        if not url:
            return "Error: No URL provided"

        try:
            import aiohttp

            # Fetch the URL
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as resp:
                    if resp.status != 200:
                        return f"Error: HTTP {resp.status} - {resp.reason}"

                    content_type = resp.headers.get("Content-Type", "")

                    # Handle different content types
                    if "text/html" in content_type:
                        html_content = await resp.text()
                        markdown_content = await self._html_to_markdown(html_content, extract_main)
                    elif "text/plain" in content_type:
                        markdown_content = await resp.text()
                    elif "application/json" in content_type:
                        json_data = await resp.json()
                        markdown_content = f"```json\n{json.dumps(json_data, indent=2)}\n```"
                    else:
                        markdown_content = f"Content type {content_type} - binary content not displayed"

                    # Truncate if too long
                    if len(markdown_content) > self.max_output_length:
                        markdown_content = (
                            markdown_content[: self.max_output_length] + "\n... (content truncated)"
                        )

                    return f"Content from {url}:\n\n{markdown_content}"

        except asyncio.TimeoutError:
            return f"Error: Request to {url} timed out after 30 seconds"
        except Exception as e:
            return f"Error fetching URL: {str(e)}"

    async def _html_to_markdown(self, html: str, extract_main: bool = True) -> str:
        """Convert HTML to markdown using html2text"""
        try:
            import html2text
            from readability import Document

            # Extract main content if requested
            if extract_main:
                doc = Document(html)
                html = doc.summary()

            # Convert to markdown
            converter = html2text.HTML2Text()
            converter.ignore_links = False
            converter.ignore_images = False
            converter.ignore_emphasis = False
            converter.body_width = 0  # Don't wrap lines

            return converter.handle(html).strip()

        except ImportError as e:
            return f"Error: Missing required library. Install with: pip install html2text readability-lxml\nDetails: {e}"
        except Exception as e:
            return f"Error converting HTML to markdown: {str(e)}"

    def cleanup(self):
        """Clean up workspace directory"""
        import shutil

        if self.workspace_dir and os.path.exists(self.workspace_dir):
            shutil.rmtree(self.workspace_dir, ignore_errors=True)
