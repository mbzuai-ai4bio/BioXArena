# Playground

Playground is the workspace where developers build their own research agents. Each playground defines a full experimental workflow by inheriting EvoMaster’s base components (`BasePlayground`, `BaseExp`) to automate a specific scientific experiment.

**Developers should create their own playground under this directory and implement their research agents here.**

## Existing Examples

| Playground | Description | Docs |
|------------|-------------|------|
| `minimal` | Basic single agent | [README](./minimal/README.md) |
| `minimal_bohrium` | Bohrium platform scientific computing | [README](./minimal_bohrium/README.md) |
| `minimal_kaggle` | Simple Kaggle competition automation | [README](./minimal_kaggle/README.md) |
| `minimal_multi_agent` | Simple multi-agent | [README](./minimal_multi_agent/README.md) |
| `minimal_multi_agent_parallel` | Parallel multi-agent experiments | [README](./minimal_multi_agent_parallel/README.md) |
| `minimal_openclaw_skill` | TypeScript skill integration | [README](./minimal_openclaw_skill/README.md) |
| `minimal_skill_task` | Anthropic native skill integration | [README](./minimal_skill_task/README.md) |
| `ml_master` | ML-Master 1.0 autonomous machine learning | [README](./ml_master/README.md) |
| `ml_master_2` | ML-Master 2.0 cognitive accumulation framework | [README](./ml_master_2/README.md) |
| `x_master` | X-Master scientific agent | [README](./x_master/README.md) |
| `browse_master` | Browse-Master web search agent | [README](./browse_master/README.md) |

## Quick Start: Create Your Playground

### 1. Create Directory Structure

```bash
mkdir -p playground/my_agent/core
mkdir -p playground/my_agent/prompts
mkdir -p configs/my_agent
```

### 2. Implement the Playground Class

`playground/my_agent/core/playground.py`:

```python
import logging
from pathlib import Path
from evomaster.core import BasePlayground, register_playground

@register_playground("my_agent")
class MyPlayground(BasePlayground):
    def __init__(self, config_dir=None, config_path=None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "my_agent"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
```

This is the minimal setup. For multi-agent setups, custom tools, or custom experiment flows, see `minimal_multi_agent`, `x_master`, and `minimal_kaggle`.

### 3. Write Prompts

`playground/my_agent/prompts/system_prompt.txt`:

```
You are a research agent. Analyze, experiment, and summarize based on the task description.
```

`playground/my_agent/prompts/user_prompt.txt`:

```
Task ID: {task_id}
Description: {description}
{input_data}
```

### 4. Configuration

`configs/my_agent/config.yaml`:

### 5. Run

```bash
python run.py --agent my_agent --task "your task description"
```

For more details, see the [development documentation](../docs/architecture.md).
