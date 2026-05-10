# Minimal Multi-Agent Parallel: Concurrent Task Execution

## Design Pattern

The `minimal_multi_agent_parallel` playground extends the sequential multi-agent pattern to run **multiple independent experiment instances concurrently**. Each experiment gets independent agent copies (with their own LLM instances and context) but shares the session and tool registry.

### Architecture

```
User Task
    │
    ├──────────────────┬──────────────────┐
    ▼                  ▼                  ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Exp 0   │    │  Exp 1   │    │  Exp 2   │
│ Plan→Code│    │ Plan→Code│    │ Plan→Code│
│ (copies) │    │ (copies) │    │ (copies) │
└──────────┘    └──────────┘    └──────────┘
    │                │                │
    ▼                ▼                ▼
  Result 0        Result 1        Result 2
```

### Application Scenarios

- Best-of-N sampling: Run the same task N times and pick the best result
- Parallel experimentation: Try different approaches simultaneously
- Resource-efficient exploration: Utilize multiple GPUs for independent agent runs

## Core Code Logic

### Playground Class

```python
@register_playground("minimal_multi_agent_parallel")
class MultiAgentParallelPlayground(BasePlayground):
    def __init__(self, config_dir=None, config_path=None):
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.agents.declare("planning_agent", "coding_agent")

        # Read parallel config
        session_config = self.config.session.get("local", {})
        parallel_config = session_config.get("parallel", {})
        if parallel_config.get("enabled", False):
            self.max_workers = parallel_config.get("max_parallel", 3)
        else:
            self.max_workers = 3
```

### Agent Copying for Parallel Safety

The key difference from the sequential version is `copy_agent()`:

```python
def _create_exp(self, exp_index):
    planning_agent_copy = self.copy_agent(
        self.agents.planning_agent,
        new_agent_name=f"planning_exp_{exp_index}"
    )
    coding_agent_copy = self.copy_agent(
        self.agents.coding_agent,
        new_agent_name=f"coding_exp_{exp_index}"
    )
    return MultiAgentExp(
        planning_agent=planning_agent_copy,
        coding_agent=coding_agent_copy,
        config=self.config,
        exp_index=exp_index
    )
```

`copy_agent()` creates a new agent that:
- Has an **independent LLM instance** (new API client, not shared)
- Has **independent context** (conversation history, trajectory)
- **Shares** session, tools, skill_registry, config

### Parallel Execution

The `run()` method uses `execute_parallel_tasks()` from `BasePlayground`:

```python
def run(self, task_description, output_file=None):
    self.setup()
    tasks = []
    for i in range(self.max_workers):
        exp = self._create_exp(exp_index=i)
        task_func = partial(exp.run, task_description=task_description)
        tasks.append(task_func)
    results = self.execute_parallel_tasks(tasks, max_workers=self.max_workers)
    return results
```

`execute_parallel_tasks()` uses `ThreadPoolExecutor` and supports:
- **Parallel index**: Sets `session.set_parallel_index(i)` for GPU/CPU resource partitioning
- **Workspace splitting**: When `split_workspace_for_exp: true`, each experiment gets an independent workspace directory (`workspace/exp_0/`, `workspace/exp_1/`, etc.)

### Configuration

```yaml
session:
  type: "local"
  local:
    working_dir: "./playground/minimal_multi_agent/workspace"
    gpu_devices: ["0", "1", "2"]    # One GPU per parallel worker
    parallel:
      enabled: true
      max_parallel: 3
      split_workspace_for_exp: true  # Independent workspaces per experiment
```

### Resource Partitioning

When parallel is enabled with GPU devices configured as a list:
- Exp 0 gets `CUDA_VISIBLE_DEVICES=0`
- Exp 1 gets `CUDA_VISIBLE_DEVICES=1`
- Exp 2 gets `CUDA_VISIBLE_DEVICES=2`

This is handled by `LocalSession.set_parallel_index()` which modifies the environment for subsequent `exec_bash` calls.

## How to Run

```bash
python run.py --agent minimal_multi_agent_parallel \
  --config configs/minimal_multi_agent_parallel/deepseek-v3.2-example.yaml \
  --task "Implement a sorting algorithm and benchmark it"
```

### Key Design Decisions

1. **Thread-based parallelism**: Uses `ThreadPoolExecutor` (not process-based) because agents share the session and MCP connections which are not easily serializable across processes.
2. **Independent LLM instances**: Each copied agent gets a new LLM client to avoid concurrent API call conflicts.
3. **Workspace isolation**: `split_workspace_for_exp` prevents file conflicts when parallel agents write to the same filenames.
4. **Exp index tracking**: `BaseAgent.set_exp_info(exp_index=i)` allows trajectory files to distinguish which parallel run produced each result.
