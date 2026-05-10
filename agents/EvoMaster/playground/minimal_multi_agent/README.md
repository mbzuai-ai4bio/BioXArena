# Multi-Agent Playground

A multi-agent playground demonstrating Planning Agent and Coding Agent collaboration.

## Overview

Multi-Agent Playground showcases how multiple agents can work together to complete complex tasks:

- **Planning Agent**: Analyzes the task and creates an execution plan
- **Coding Agent**: Executes code tasks based on the plan

This pattern is useful for tasks that benefit from separation of planning and execution.

## Workflow

```
┌─────────────────┐
│  Task Input     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Planning Agent  │  Analyzes task, creates plan
└────────┬────────┘
         │ Plan
         ▼
┌─────────────────┐
│  Coding Agent   │  Executes code based on plan
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Results      │
└─────────────────┘
```

## Quick Start

### 1. Configure

Edit `configs/minimal_multi_agent/deepseek-v3.2-example.yaml`:

```yaml
  local_sglang:
    provider: "deepseek"
    model: "deepseek-v3.2"
    api_key: "dummy"  # Use placeholder for local deployment
    base_url: "http://192.168.2.110:18889/v1"
    temperature: 0.7
    max_tokens: 16384
    timeout: 300  
    max_retries: 3
    retry_delay: 1.0

  # ... If using OpenAI API format, also modify each agent's LLM configuration, e.g.
  agents:
    draft:
      llm: "local_sglang" # Change to openai if needed
```

### 2. Run

```bash
# Run with task description
python run.py --agent minimal_multi_agent --config configs/minimal_multi_agent/deepseek-v3.2-example.yaml --task "Write a Python program that implements the following features: Read a text file (create a sample file if it doesn't exist). Count the occurrences of each word in the file. Sort the results by frequency in descending order. Save the results to a new file named word_count.txt. Output the top 10 most common words to the terminal."
```

### 3. View Results

Results are saved in:

```
runs/minimal_multi_agent_{timestamp}/
├── trajectories/       # Agent execution trajectories
├── logs/              # Execution logs
└── workspace/         # Generated files
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `agents.planning.max_turns` | Max planning turns | `10` |
| `agents.planning.enable_tools` | Enable planning tools | `false` |
| `agents.coding.max_turns` | Max coding turns | `50` |
| `agents.coding.enable_tools` | Enable coding tools | `true` |
| `skills.enabled` | Enable skill system | `false` |

## Usage Examples

### Write Code
```bash
python run.py --agent minimal_multi_agent --config configs/minimal_multi_agent/deepseek-v3.2-example.yaml --task "Write a Python program that implements the following features: Read a text file (create a sample file if it doesn't exist). Count the occurrences of each word in the file. Sort the results by frequency in descending order. Save the results to a new file named word_count.txt. Output the top 10 most common words to the terminal."
```

## Directory Structure

```
playground/minimal_multi_agent/
├── core/
│   ├── __init__.py
│   ├── playground.py    # Main playground
│   └── exp.py           # Multi-agent experiment
├── prompts/
│   ├── planning_system_prompt.txt
│   ├── planning_user_prompt.txt
│   ├── coding_system_prompt.txt
│   └── coding_user_prompt.txt
```

## Customization

### Adding More Agents

To add more agents, update the config:

```yaml
agents:
  planning:
    # ...
  coding:
    # ...
  review:  # New agent
    llm: "openai"
    max_turns: 10
    enable_tools: false
    system_prompt_file: "prompts/review_system_prompt.txt"
```

Then modify `playground.py` and `exp.py` to include the new agent in the workflow.

## Related

- [EvoMaster Main README](../../README.md)
- [Minimal Playground](../minimal/README.md)
- [Configuration Examples](../../configs/)
