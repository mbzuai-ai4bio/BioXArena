# Configuration Reference

## Overview

EvoMaster uses YAML configuration files with support for environment variable substitution (`${VAR_NAME}`), `.env` file loading, and hierarchical multi-agent configuration.

## Configuration File Location

```
configs/{agent_name}/
├── config.yaml                # Default config (loaded when no --config is specified)
├── deepseek-v3.2-example.yaml  # Alternative configs for different LLM backends
├── gpt-5-example.yaml
└── mcp_config.json            # MCP server configuration (optional)
```

## Environment Variables

Create a `.env` file at the project root:

```bash
# LLM API Keys
OPENAI_API_KEY=sk-...
GPT_BASE_URL=https://api.openai.com/v1
GPT_CHAT_MODEL=gpt-4o

ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_BASE_URL=https://api.anthropic.com

# Optional: Bohrium
BOHRIUM_ACCESS_KEY=...
BOHRIUM_PROJECT_ID=...
```

EvoMaster searches for `.env` from the config directory upward to the project root.

## Complete Configuration Structure

```yaml
# ============================================
# LLM Configuration
# ============================================
llm:
  # Define named LLM profiles
  openai:
    provider: "openai"          # Provider: "openai" or "anthropic"
    model: "${GPT_CHAT_MODEL}"  # Model name (supports env vars)
    api_key: "${OPENAI_API_KEY}"
    base_url: "${GPT_BASE_URL}"
    temperature: 0.7            # Sampling temperature (0.0-2.0)
    max_tokens: 16384           # Max output tokens per response
    timeout: 60                 # API call timeout in seconds
    max_retries: 3              # Number of retries on failure
    retry_delay: 1.0            # Delay between retries in seconds

  anthropic:
    provider: "anthropic"
    model: "claude-haiku-4-5-20251001"
    api_key: "${ANTHROPIC_API_KEY}"
    base_url: "${ANTHROPIC_BASE_URL}"
    temperature: 0.7
    max_tokens: 16384
    timeout: 60
    max_retries: 3
    retry_delay: 1.0

  # For local models (e.g., via SGLang/vLLM)
  local_sglang:
    provider: "openai"          # Uses OpenAI-compatible API
    model: "deepseek-v3"
    api_key: "EMPTY"
    base_url: "http://localhost:30000/v1"
    temperature: 0.7
    max_tokens: 16384
    timeout: 300                # Local models may need longer timeout

  default: "openai"             # Which profile to use when agent doesn't specify

# ============================================
# Agent Configuration
# ============================================
agents:
  # Each key defines a named agent
  general:                      # Agent name (used to create agent slot: general_agent)
    llm: "openai"               # Reference to an LLM profile above
    max_turns: 50               # Maximum ReAct loop iterations
    finish_on_text_response: false  # If true, text-only responses end the turn

    tools:
      builtin: ["*"]            # Built-in tools: ["*"], [], or specific names
      # builtin: ["execute_bash", "finish"]
      mcp: ""                   # MCP config file path (relative to config dir), "" = disabled
      # mcp: "mcp_config.json"
      # Custom tools: key=type, value=filename
      # my_search: "google_search"

    skills: []                  # Skills to expose: ["*"], [], or specific names
    # skills: ["rag", "pdf"]
    skill_dir: "./evomaster/skills"  # Skills root directory

    context:
      max_tokens: 128000        # Context window size
      truncation_strategy: "latest_half"  # Options below
      preserve_system_messages: true
      preserve_recent_turns: 5  # Number of recent turns to keep during truncation

    # Prompt file paths (relative to playground directory)
    system_prompt_file: "prompts/system_prompt.txt"
    user_prompt_file: "prompts/user_prompt.txt"

    # Template variables for prompts
    prompt_format_kwargs: {}

# ============================================
# Session Configuration
# ============================================
session:
  type: "local"                 # "local" or "docker"

  local:
    working_dir: "./playground/minimal/workspace"
    timeout: 60                 # Command execution timeout
    gpu_devices: "0"            # GPU assignment: null, "0", ["0","1"], "all"
    cpu_devices: "0-15"         # CPU assignment: null, "0-15", [0,1,2,3]

    # Symlink data into workspace
    symlinks:
      # "/path/to/data": "input"    # Source → target (relative to workspace)

    # Parallel execution config
    parallel:
      enabled: false
      max_parallel: 3
      split_workspace_for_exp: false  # Independent workspace per parallel exp

  docker:
    image: "evomaster/base:latest"
    container_name: null        # null = auto-generated
    use_existing_container: null # Reuse a specific container
    working_dir: "/workspace"
    memory_limit: "64g"
    cpu_limit: 16.0
    gpu_devices: "0"
    network_mode: "host"        # "bridge", "host", "none"
    volumes:
      "./playground/minimal/workspace": "/workspace"
    env_vars: {}
    auto_remove: false          # true = remove container on exit
    timeout: 300

# ============================================
# Output & Logging
# ============================================
llm_output:
  show_in_console: true         # Stream LLM output to terminal
  log_to_file: true             # Log LLM output to file

logging:
  level: "INFO"                 # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: null                    # Log file path (null = no file)
  console: true                 # Output to console
```

## Context Truncation Strategies

| Strategy | Behavior |
|---|---|
| `none` | No truncation; fails if context overflows |
| `latest_half` | Keeps system messages + the latest half of conversation turns |
| `sliding_window` | Keeps the N most recent turns (N = `preserve_recent_turns`) |
| `summary` | Uses an LLM to summarize older turns before removing them |

## Running with Different Configs

```bash
# Default config
python run.py --agent minimal --task "your task"

# Specific config file
python run.py --agent minimal --config configs/minimal/deepseek-v3.2-example.yaml --task "your task"

# Custom run directory
python run.py --agent minimal --task "your task" --run-dir runs/experiment_01

# Batch tasks from JSON file
python run.py --agent minimal --task-file tasks.json

# Parallel batch tasks
python run.py --agent minimal --task-file tasks.json --parallel
```

## Task File Format

For batch tasks, provide a JSON file:

```json
[
  {"id": "task_1", "description": "First task description"},
  {"id": "task_2", "description": "Second task description"},
  "Simple task description (auto-assigned ID: task_2)"
]
```

## Run Directory Structure

```
runs/{agent}_{timestamp}/
├── config.yaml              # Copy of the config used
├── logs/
│   ├── evomaster.log        # Single task mode
│   └── {task_id}.log        # Batch task mode
├── trajectories/
│   ├── trajectory.json      # Single task mode
│   └── {task_id}/           # Batch task mode
│       └── trajectory.json
├── workspace/               # Single task mode
└── workspaces/              # Batch task mode
    ├── task_1/
    └── task_2/
```
