# Minimal: Single-Agent ReAct Pattern

## Design Pattern

The `minimal` playground demonstrates the simplest EvoMaster setup: a single autonomous agent that uses the **ReAct** (Reasoning + Acting) pattern to iteratively think, execute tools, observe results, and refine its approach until a task is complete.

### Architecture

```
User Task
    │
    ▼
┌─────────────────────────┐
│  Single Agent (ReAct)   │
│  ┌───────────────────┐  │
│  │ Think → Act →     │  │
│  │ Observe → Think   │  │
│  │ → Act → ...       │  │
│  └───────────────────┘  │
│  Tools: bash, editor,   │
│         think, finish   │
└─────────────────────────┘
    │
    ▼
  Result
```

### Application Scenarios

- Autonomous scientific discovery and data analysis
- Code generation and debugging
- File manipulation and automation
- Any single-objective task that benefits from iterative tool use

## Core Code Logic

### Playground Class

The `MinimalPlayground` inherits from `BasePlayground` and uses all default behaviors — no custom experiment logic is needed:

```python
from evomaster.core import BasePlayground, register_playground

@register_playground("minimal")
class MinimalPlayground(BasePlayground):
    def __init__(self, config_dir=None, config_path=None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "agent" / "minimal"
        super().__init__(config_dir=config_dir, config_path=config_path)
```

Key points:
- `@register_playground("minimal")` registers this class so `run.py` can discover it by name.
- Inheriting `BasePlayground` provides the full lifecycle: config loading → session setup → agent creation → experiment execution → cleanup.
- The default `_create_exp()` creates a `BaseExp` that runs a single agent loop.

### Execution Flow

1. `BasePlayground.run()` calls `setup()` → `_setup_session()` + `_setup_agents()`
2. `_create_exp()` creates a `BaseExp` with the configured agent
3. `BaseExp.run()` creates a `TaskInstance` and calls `agent.run(task)`
4. The agent enters its ReAct loop (`BaseAgent._step()`):
   - Prepares context (system prompt + user prompt + conversation history)
   - Queries the LLM
   - Processes tool calls or text responses
   - Repeats until `finish` is called or `max_turns` is reached

### Configuration Structure

```yaml
# configs/minimal/config.yaml

llm:
  openai:
    provider: "openai"
    model: "${GPT_CHAT_MODEL}"
    api_key: "${OPENAI_API_KEY}"
    base_url: "${GPT_BASE_URL}"
    temperature: 0.7
    max_tokens: 16384
  default: "openai"

agents:
  general:
    llm: "openai"
    max_turns: 50
    tools:
      builtin: ["*"]          # All built-in tools enabled
    context:
      max_tokens: 128000
      truncation_strategy: "latest_half"
      preserve_system_messages: true
      preserve_recent_turns: 5
    system_prompt_file: "prompts/system_prompt.txt"
    user_prompt_file: "prompts/user_prompt.txt"

session:
  type: "local"
  local:
    working_dir: "./playground/minimal/workspace"
    timeout: 60
    gpu_devices: "0"
```

### Prompt Design

The system prompt instructs the agent to:
1. Understand the task
2. Formulate an experiment plan
3. Write code to files (not inline), then execute
4. Analyze results, iterate if needed
5. Call `finish` when the discovery is complete

The user prompt is a template with placeholders `{task_id}`, `{task_type}`, `{description}`, and `{input_data}` that are filled from the `TaskInstance`.

## How to Run

```bash
# Basic usage
python run.py --agent minimal --task "Analyze the correlation between temperature and ice cream sales"

# With a custom config
python run.py --agent minimal --config configs/minimal/config.yaml --task "your task"

# With a task file
python run.py --agent minimal --task path/to/task.txt

# With images (multimodal)
python run.py --agent minimal --task "Describe this image" --images photo.png

# Specify a run directory
python run.py --agent minimal --task "your task" --run-dir runs/my_experiment
```

### Output Structure

```
runs/minimal_20250101_120000/
├── config.yaml          # Copy of the config used
├── logs/
│   └── evomaster.log    # Execution log
├── trajectories/
│   └── trajectory.json  # Full agent conversation history
└── workspace/           # Agent's working directory (files created by the agent)
```
