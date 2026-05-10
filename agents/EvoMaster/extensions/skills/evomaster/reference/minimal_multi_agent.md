# Minimal Multi-Agent: Sequential Collaboration

## Design Pattern

The `minimal_multi_agent` playground demonstrates how two agents with different roles collaborate **sequentially** on a task. A Planning Agent analyzes the task and creates a plan, then a Coding Agent executes the plan using tools.

### Architecture

```
User Task
    │
    ▼
┌───────────────────────┐
│   Planning Agent      │  (text-only, no tools)
│   Analyzes task,      │
│   formulates plan     │
└───────────┬───────────┘
            │ plan output
            ▼
┌───────────────────────┐
│   Coding Agent        │  (ReAct with all tools)
│   Implements plan,    │
│   writes & runs code  │
└───────────────────────┘
            │
            ▼
        Result
```

### Application Scenarios

- Tasks that benefit from separating planning and execution
- Complex coding tasks requiring architectural decisions first
- Workflows where a lightweight reasoning model plans and a capable model executes

## Core Code Logic

### Playground Class

```python
@register_playground("minimal_multi_agent")
class MultiAgentPlayground(BasePlayground):
    def __init__(self, config_dir=None, config_path=None):
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.agents.declare("planning_agent", "coding_agent")
```

Key elements:
- `agents.declare()` pre-declares agent slots for IDE auto-completion and validation
- Agent names in `declare()` must match the config `agents:` keys with `_agent` suffix (e.g., config key `planning` → slot `planning_agent`)

### Custom Experiment Class

The `MultiAgentExp` class orchestrates the two-phase workflow:

```python
class MultiAgentExp(BaseExp):
    def __init__(self, planning_agent, coding_agent, config):
        super().__init__(planning_agent, config)
        self.planning_agent = planning_agent
        self.coding_agent = coding_agent

    def run(self, task_description, task_id="exp_001"):
        # Phase 1: Planning
        planning_task = TaskInstance(
            task_id=f"{task_id}_planning",
            task_type="planning",
            description=task_description,
        )
        planning_trajectory = self.planning_agent.run(planning_task)
        planning_result = self._extract_agent_response(planning_trajectory)

        # Phase 2: Coding (receives the plan via prompt_format_kwargs)
        self.coding_agent._prompt_format_kwargs.update({
            'planning_result': planning_result
        })
        coding_task = TaskInstance(
            task_id=f"{task_id}_coding",
            task_type="coding",
            description=task_description,
        )
        coding_trajectory = self.coding_agent.run(coding_task)
        return results
```

### Inter-Agent Communication

The planning agent's output is passed to the coding agent through **prompt template variables**:

1. The planning agent runs and produces a text response
2. `_extract_agent_response()` extracts the final text from the trajectory
3. The result is injected into the coding agent's user prompt via `_prompt_format_kwargs`
4. The coding agent's prompt template uses `{planning_result}` placeholder

### Overriding `_create_exp()`

The playground overrides `_create_exp()` to create a custom experiment:

```python
def _create_exp(self):
    return MultiAgentExp(
        planning_agent=self.agents.planning_agent,
        coding_agent=self.agents.coding_agent,
        config=self.config
    )
```

### Configuration

```yaml
agents:
  planning:
    llm: "openai"
    max_turns: 10
    tools:
      builtin: []           # No tools — text reasoning only
    system_prompt_file: "prompts/planning_system_prompt.txt"
    user_prompt_file: "prompts/planning_user_prompt.txt"

  coding:
    llm: "openai"
    max_turns: 50
    tools:
      builtin: ["*"]        # All tools for code execution
    system_prompt_file: "prompts/coding_system_prompt.txt"
    user_prompt_file: "prompts/coding_user_prompt.txt"
```

### Prompt Design

- **Planning System Prompt**: Instructs the agent to analyze requirements, decompose the task, and output a structured plan
- **Planning User Prompt**: Template with `{description}` filled from the task
- **Coding User Prompt**: Template with `{description}` and `{planning_result}` — the plan from the planning agent

## How to Run

```bash
python run.py --agent minimal_multi_agent \
  --config configs/minimal_multi_agent/config.yaml \
  --task "Build a web scraper that extracts product prices from an e-commerce site"
```

### Key Design Decisions

1. **Separation of concerns**: The planning agent focuses on reasoning without being distracted by tool calls. The coding agent focuses on implementation guided by a clear plan.
2. **Different `max_turns`**: Planning needs fewer turns (10) than coding (50), saving API costs.
3. **Shared session**: Both agents share the same `LocalSession`, so files created by one agent are visible to the other.
4. **`_extract_agent_response()`**: This utility extracts the agent's final answer, preferring the `finish` tool's `message` parameter, falling back to the last assistant message content.
