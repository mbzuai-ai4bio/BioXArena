# Minimal Playground

The simplest playground implementation, demonstrating how to use EvoMaster's basic features.

## Overview

Minimal Playground is a single-agent playground that shows how to quickly set up and run an EvoMaster agent. It's ideal for:

- Getting started with EvoMaster
- Simple task execution
- Learning the basic playground structure

## Quick Start

### 1. Configure

Edit `configs/minimal/deepseek-v3.2-example.yaml`:

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
# Using the unified entry point
python run.py --agent minimal --config configs/minimal/deepseek-v3.2-example.yaml --task "Discover a pattern: Given sequence 1, 4, 9, 16, 25... find the formula"
```

### 3. View Results

After execution, results are saved in the `runs/` directory:

```
runs/minimal_{timestamp}/
├── trajectories/       # Agent execution trajectories
├── logs/              # Execution logs
└── workspace/         # Agent working files
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `llm.default` | Default LLM provider | `"openai"` |
| `agents.general.max_turns` | Maximum conversation turns | `50` |
| `agents.general.enable_tools` | Enable tool calling | `true` |
| `session.type` | Session type (local/docker) | `"local"` |
| `mcp.enabled` | Enable MCP tools | `true` |

## Example Tasks

### Finding Mathematical Patterns
```bash
python run.py --agent minimal --config configs/minimal/deepseek-v3.2-example.yaml --task "Discover a pattern: Given sequence 1, 4, 9, 16, 25... find the formula"
```

## Directory Structure

```
playground/minimal/
├── core/
│   ├── __init__.py
│   └── playground.py    # Main playground implementation
├── prompts/
│   ├── system_prompt.txt
│   └── user_prompt.txt
```

## Customization

To customize the agent's behavior, edit the prompt files in `configs/minimal/prompts/`:

- `system_prompt.txt` - Agent's role and capabilities
- `user_prompt.txt` - Task formatting template

## Related Documentation

- [EvoMaster Main README](../../README.md)
- [Configuration Examples](../../configs/)
