# Claude Code Agent Training

## Introduction

This example demonstrates how to train a **Claude Code Agent** using Miles RL framework. The agent learns to perform software engineering tasks by interacting with a codebase using tools like `bash`, `read`, `write`, `edit`, `grep`, `glob`, `web_search`, and `web_fetch`.

**Architecture:** This implementation follows the **SWE-Agent** pattern with a decoupled Gym environment:

```
┌─────────────────┐         HTTP API        ┌──────────────────────────┐
│                 │◄──────────────────────► │  Claude Code Gym         │
│  Miles Training │                         │  ┌────────────────────┐  │
│  (RL Loop)      │  Request: task prompt   │  │ Agent Controller   │  │
│                 │  Response: trajectory   │  │                    │  │
│                 │           + reward      │  │ ┌──────────────┐   │  │
└─────────────────┘                         │  │ │ Tool Executor│   │  │
                                            │  │ │ - bash       │   │  │
                                            │  │ │ - read/write │   │  │
                                            │  │ │ - edit       │   │  │
                                            │  │ │ - grep/glob  │   │  │
                                            │  │ │ - web_search │   │  │
                                            │  │ │ - web_fetch  │   │  │
                                            │  │ └──────────────┘   │  │
                                            │  └────────────────────┘  │
                                            └──────────────────────────┘
```

## Project Structure

```
examples/claude-code-agent/
├── README.md                          # This file
├── tool_executor.py                   # Tool execution (8 tools: bash, read, write, edit, grep, glob, web_search, web_fetch)
├── generate_with_claude_code.py       # Miles integration (generate & reward functions)
├── gym_server.py                      # Claude Code Gym environment server
├── run-train.sh                       # Training launch script
├── sample_tasks.jsonl                 # Example task dataset (10 tasks)
├── requirements.txt                   # Python dependencies
└── test_tools.py                      # Tool testing script
```

## Setup

### 1. Environment Setup

You'll need **two separate environments**:

#### A. Miles Training Environment (this container)
Already set up with Miles installation.

#### B. Claude Code Gym Environment (separate container/process)

```bash
# Create a separate directory for the Gym environment
mkdir -p ~/claude-code-gym
cd ~/claude-code-gym

# Install dependencies (or use requirements.txt)
pip install -r /data/guanxinglu/miles/examples/claude-code-agent/requirements.txt

# Copy the gym server implementation
cp /data/guanxinglu/miles/examples/claude-code-agent/gym_server.py .
cp /data/guanxinglu/miles/examples/claude-code-agent/tool_executor.py .

# Test tools (optional but recommended)
python /data/guanxinglu/miles/examples/claude-code-agent/test_tools.py
```

### 2. Network Configuration

If running in Docker, ensure both containers can communicate:

```bash
# Create a shared network
docker network create miles-net

# Attach both containers to the network
docker network connect miles-net <miles_container_name>
docker network connect miles-net <gym_container_name>
```

### 3. Task Dataset Preparation

Create a task dataset in JSONL format:

```jsonl
{"prompt": "Fix the bug in the authentication module where users can't log in", "metadata": {"task_type": "bug_fix", "workspace_setup": {"repo": "https://github.com/example/repo.git", "branch": "main"}, "success_criteria": {"tests_pass": true, "files_modified": ["auth.py"]}}, "label": "success"}
{"prompt": "Add a new API endpoint for user profile updates", "metadata": {"task_type": "feature", "workspace_setup": {"repo": "https://github.com/example/api.git"}, "success_criteria": {"endpoint_created": "/api/profile", "tests_added": true}}, "label": "success"}
```

See `sample_tasks.jsonl` for more examples.

## Running Training

### Step 1: Start the Claude Code Gym Server

In the **Gym environment container**:

```bash
cd ~/claude-code-gym
python gym_server.py --port 12000 --host 0.0.0.0
```

This starts the Gym server that will:
- Receive task requests from Miles
- Execute the agent with tools
- Evaluate task completion
- Return trajectories and rewards

### Step 2: Configure and Launch Training

In the **Miles training container**:

```bash
# Set Gym URL environment variable
export CLAUDE_CODE_GYM_URL="http://<gym_container_name>:12000"

# Launch training
bash examples/claude-code-agent/run-train.sh
```

The training script will:
1. Load task dataset
2. For each task, send it to the Gym environment
3. Gym runs the agent, collecting tool calls and responses
4. Miles receives the trajectory and reward
5. Performs GRPO update on the policy

## Configuration

### Key Arguments

Edit `run-train.sh` to customize:

```bash
# Model settings
--model_name_or_path=/path/to/qwen3-4b-instruct

# Rollout settings
--rollout_max_turns=16              # Max conversation turns
--rollout_max_tool_calls=20         # Max tool executions
--rollout_batch_size=64             # Rollout batch size

# Training settings
--num_epochs=3                      # Training epochs
--learning_rate=1e-6                # Learning rate
--grpo_beta=0.1                     # GRPO KL penalty

# Data settings
--train_data_path=/path/to/tasks.jsonl
```

### Reward Function

The reward function in `generate_with_claude_code.py:152-192` uses:

- **Task success**: +1.0 (primary signal)
- **Partial progress**: +0.0 to +0.5 (based on evaluation)
- **Tool efficiency**: -0.01 per tool call (capped at -0.3)
- **Error penalty**: -0.5 for failures

You can customize the reward function to match your training objectives.

## Available Tools

The agent has access to the following tools (matching real Claude Code capabilities):

### File Operations
- **bash**: Execute bash commands in the workspace
  - Usage: `{"name": "bash", "arguments": {"command": "ls -la"}}`

- **read**: Read file contents with line numbers
  - Usage: `{"name": "read", "arguments": {"file_path": "main.py"}}`

- **write**: Create or overwrite files
  - Usage: `{"name": "write", "arguments": {"file_path": "config.py", "content": "..."}}`

- **edit**: Replace text in files (with uniqueness check)
  - Usage: `{"name": "edit", "arguments": {"file_path": "app.py", "old_string": "...", "new_string": "..."}}`

### Search & Discovery
- **grep**: Search for patterns in files (uses ripgrep)
  - Usage: `{"name": "grep", "arguments": {"pattern": "TODO", "path": "."}}`

- **glob**: Find files matching patterns
  - Usage: `{"name": "glob", "arguments": {"pattern": "**/*.py"}}`

### Web Access (NEW!)
- **web_search**: Search the web for information
  - Usage: `{"name": "web_search", "arguments": {"query": "Python asyncio tutorial", "num_results": 5}}`
  - Requires: Set `SEARCH_API_KEY` environment variable for SerpAPI/Google Custom Search
  - Fallback: Returns mock results if no API key configured

- **web_fetch**: Fetch and convert web pages to markdown
  - Usage: `{"name": "web_fetch", "arguments": {"url": "https://docs.python.org/3/library/asyncio.html"}}`
  - Features: Automatic HTML→Markdown conversion, main content extraction
  - Dependencies: `html2text`, `readability-lxml` (optional, graceful fallback)

### Web Search API Configuration

To enable real web search (otherwise uses mock results):

```bash
# Option 1: SerpAPI (recommended)
export SEARCH_API_KEY="your-serpapi-key"  # Get from https://serpapi.com/

# Option 2: Google Custom Search
# Modify _execute_web_search() in tool_executor.py to use Google API
export GOOGLE_API_KEY="your-google-api-key"
export GOOGLE_CSE_ID="your-custom-search-engine-id"
```

## Task Types

The system supports various task types:

1. **Bug Fix**: Fix failing tests or broken functionality
   - Success: All tests pass after changes

2. **Feature Addition**: Implement new functionality
   - Success: New feature works, tests added and passing

3. **Refactoring**: Improve code structure without changing behavior
   - Success: Tests still pass, code quality metrics improved

4. **Code Review**: Identify and fix code quality issues
   - Success: Issues identified and resolved

5. **General**: Open-ended software engineering tasks
   - Success: Based on custom evaluation criteria

## Metrics

The system logs the following metrics:

```python
agent/turns_mean              # Average conversation turns
agent/tool_calls_mean         # Average tool calls per task
agent/success_rate            # Task completion rate
agent/model_query_time_mean   # Avg model inference time
agent/env_execution_time_mean # Avg tool execution time
agent/total_time_mean         # Avg total task time
```

## Troubleshooting

### 1. Connection Issues

**Error:** `Failed to connect to Gym URL`

**Solution:**
- Check `CLAUDE_CODE_GYM_URL` is set correctly
- Verify Gym server is running: `curl http://<gym_url>/health`
- Check network connectivity between containers

### 2. Gym Server Timeouts

**Error:** `Gym request timed out`

**Solution:**
- Increase timeout in `gym_server.py`
- Reduce `rollout_max_turns` or `rollout_max_tool_calls`
- Check for infinite loops in agent behavior

### 3. Low Success Rate

**Issue:** Agent not completing tasks successfully

**Solution:**
- Check task dataset quality (are tasks clear and achievable?)
- Review reward function (is it providing good signal?)
- Inspect agent trajectories (what mistakes is it making?)
- Consider starting with simpler tasks

## Next Steps

### 1. Implement the Gym Server

The `gym_server.py` file needs to be implemented with:
- FastAPI server handling `/run` endpoint
- Agent controller that orchestrates tool calls
- Task evaluation logic
- Workspace management (git clone, cleanup, etc.)

### 2. Create Task Dataset

Build a diverse dataset of software engineering tasks:
- Start with simple tasks (read file, list files)
- Progress to complex tasks (fix bugs, add features)
- Include variety of repositories and programming languages

### 3. Experiment with Reward Shaping

Try different reward structures:
- Dense rewards (reward per tool call)
- Sparse rewards (only final success/failure)
- Curriculum learning (start easy, increase difficulty)
- Multi-objective rewards (success + efficiency + code quality)

### 4. Scale Up

Once the basic setup works:
- Use larger models (Qwen3-8B, 30B)
- Increase dataset size and diversity
- Add more tools (git, pytest, linting, etc.)
- Implement multi-step verification

## References

- **SWE-Agent Example**: `examples/swe-agent/` - Similar architecture for software engineering tasks
- **Retool Example**: `examples/retool/` - Tool execution and sandbox implementation
- **Miles Documentation**: See main README for GRPO algorithm and training details

## Contributing

To extend this example:
1. Add new tools in `tool_executor.py`
2. Create new task types in dataset
3. Customize reward function for your use case
4. Share your results and improvements!
