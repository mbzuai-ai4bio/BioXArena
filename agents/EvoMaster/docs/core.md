# Core Module

The Core module provides the workflow components: `BaseExp` and `BasePlayground`.

## Overview

```
evomaster/core/
├── exp.py          # BaseExp class
└── playground.py   # BasePlayground class
```

## BaseExp

`BaseExp` is the base class for single experiment execution.

### Class Definition

```python
class BaseExp:
    """Experiment base class

    Defines common execution logic for single experiments.
    Specific playgrounds can inherit and override methods.
    """
```

### Constructor

```python
def __init__(self, agent, config):
    """Initialize experiment

    Args:
        agent: Agent instance
        config: EvoMasterConfig instance
    """
```

### Properties

```python
@property
def exp_name(self) -> str:
    """Get Exp name (auto-inferred from class name)

    Example: SolverExp -> Solver, CriticExp -> Critic
    Subclasses can override this property.
    """
```

### Methods

#### set_run_dir(run_dir)
```python
def set_run_dir(self, run_dir: str | Path) -> None:
    """Set run directory

    Args:
        run_dir: Run directory path
    """
```

#### run(task_description, task_id, images)
```python
def run(self, task_description: str, task_id: str = "exp_001", images: list[str] | None = None, on_step=None) -> dict:
    """Run a single experiment

    Args:
        task_description: Task description
        task_id: Task ID
        images: Optional list of image file paths (for multimodal tasks)
        on_step: Optional step callback

    Returns:
        Result dictionary with:
        - trajectory: Execution trajectory
        - status: Completion status
        - steps: Number of steps taken
    """
```

#### save_results(output_file)
```python
def save_results(self, output_file: str):
    """Save experiment results

    Args:
        output_file: Output file path
    """
```

### Internal Methods

```python
def _extract_agent_response(self, trajectory: Any) -> str:
    """Extract agent's final response from trajectory

    Args:
        trajectory: Execution trajectory

    Returns:
        Agent's response text
    """
```

## BasePlayground

`BasePlayground` is the workflow orchestrator.

### Class Definition

```python
class BasePlayground:
    """Playground base class

    Defines common workflow lifecycle:
    1. Load configuration
    2. Initialize all components
    3. Create and run experiments
    4. Clean up resources

    Specific playgrounds can:
    - Inherit this class
    - Override _create_exp() for custom Exp class
    - Override setup() for additional initialization
    """
```

### Constructor

```python
def __init__(
    self,
    config_dir: str | Path | None = None,
    config_path: str | Path | None = None
):
    """Initialize Playground

    Args:
        config_dir: Config directory (default: configs/)
        config_path: Full config file path (overrides config_dir if provided)
    """
```

### Lifecycle Methods

#### set_run_dir(run_dir, task_id)
```python
def set_run_dir(self, run_dir: str | Path, task_id: str | None = None) -> None:
    """Set run directory and create structure

    Creates:
    - run_dir/config.yaml (config copy)
    - run_dir/logs/ (log files)
    - run_dir/trajectories/ (dialog trajectories)
    - run_dir/workspace/ or run_dir/workspaces/{task_id}/ (workspace)

    Args:
        run_dir: Run directory path
        task_id: Optional task ID for batch mode
    """
```

#### setup()
```python
def setup(self) -> None:
    """Initialize all components

    Steps:
    1. Create Session (if not already created) via _setup_session()
    2. Create Agents via _setup_agents(), which automatically:
       - Reads per-agent config (LLM, tools, skills)
       - Creates independent LLM instances
       - Creates tool registries with MCP support
       - Loads per-agent skill registries
       - Registers agents to self.agents (AgentSlots)
    """
```

#### run(task_description, output_file)
```python
def run(self, task_description: str, output_file: str | None = None) -> dict:
    """Run workflow

    Args:
        task_description: Task description
        output_file: Optional result file (auto-saved to trajectories/ if run_dir set)

    Returns:
        Run result dictionary
    """
```

#### cleanup()
```python
def cleanup(self) -> None:
    """Clean up resources

    For DockerSession with auto_remove=False, keeps container running
    for reuse in subsequent runs.
    """
```

### Component Creation Methods

#### _create_agent(name, agent_config, llm_config, tool_config, skill_config)
```python
def _create_agent(
    self,
    name: str,
    agent_config: dict | None = None,
    llm_config: dict | None = None,
    tool_config: dict | None = None,
    skill_config: dict | None = None,
) -> Agent:
    """Create Agent instance

    Each Agent uses independent LLM instance for isolated logging.
    All parameters are optional; if not provided, they are automatically
    fetched from the config manager based on the agent name.

    Args:
        name: Agent name
        agent_config: Agent configuration dict (auto-fetched if None)
        llm_config: LLM config dict (auto-fetched if None)
        tool_config: Tool config dict, e.g. {"builtin": list[str], "mcp": str} (auto-fetched if None)
        skill_config: Skill config dict (auto-fetched if None)

    Returns:
        Agent instance
    """
```

#### copy_agent(agent, new_agent_name)
```python
def copy_agent(self, agent, new_agent_name: str | None = None) -> Agent:
    """Copy Agent instance with independent LLM and context

    Creates a new Agent that:
    - Has an independent LLM instance
    - Shares session, tools, skill_registry, config_dir, enable_tools
    - Has independent context (context_manager, current_dialog, trajectory)

    Args:
        agent: Agent instance to copy
        new_agent_name: Optional name for the new agent

    Returns:
        New Agent instance
    """
```

#### _create_exp()
```python
def _create_exp(self):
    """Create Exp instance

    Subclasses can override for custom Exp class.
    """
```

### MCP Methods

#### _setup_mcp_tools()
```python
def _setup_mcp_tools(self) -> MCPToolManager | None:
    """Initialize MCP tools

    Reads MCP config file (JSON format), initializes connections,
    and registers tools.

    Returns:
        MCPToolManager instance, or None if config invalid
    """
```

#### _parse_mcp_servers(mcp_config)
```python
def _parse_mcp_servers(self, mcp_config: dict) -> list[dict]:
    """Parse MCP server configuration

    Supports standard MCP format and extended format.

    Args:
        mcp_config: MCP config dictionary

    Returns:
        List of server configurations
    """
```

### Internal Methods

```python
def _setup_session(self) -> None:
    """Create and open Session (if not created)"""

def _setup_agents(self) -> None:
    """Auto-create agents from config and register to self.agents (AgentSlots)"""

def _setup_agent_llm(self, name: str) -> dict:
    """Get per-agent LLM configuration"""

def _setup_agent_tools(self, name: str) -> dict:
    """Get per-agent tool configuration"""

def _setup_agent_skills(self, name: str) -> dict:
    """Get per-agent skills configuration"""

def _setup_tools(self, skill_config=None, tool_config=None) -> None:
    """Create tool registry and init MCP tools"""

def _get_output_config(self) -> dict:
    """Get LLM output configuration"""

def _setup_logging(self) -> None:
    """Set up logging file path"""

def _update_workspace_path(self, workspace_path: Path) -> None:
    """Dynamically update workspace_path in config"""

def _setup_trajectory_file(self, output_file: str | Path | None = None) -> Path | None:
    """Set trajectory file path"""

def execute_parallel_tasks(self, tasks, max_workers=None) -> list:
    """Execute multiple experiments in parallel using ThreadPoolExecutor"""
```

## Usage Examples

### Custom Playground

```python
from evomaster.core import BasePlayground, BaseExp

class MyExp(BaseExp):
    def run(self, task_description: str, task_id: str = "exp_001") -> dict:
        # Custom experiment logic
        result = super().run(task_description, task_id)
        # Post-processing
        return result

class MyPlayground(BasePlayground):
    def _create_exp(self):
        return MyExp(self.agent, self.config)

    def setup(self):
        super().setup()
        # Additional initialization
        self.logger.info("Custom setup complete")
```

### Multi-Agent Playground

```python
class MultiAgentPlayground(BasePlayground):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Declare agent slots (IDE-friendly)
        self.agents.declare("planning_agent", "coding_agent")

    def setup(self):
        # Two lines: base class auto-handles LLM/Tools/Skills per agent
        self._setup_session()
        self._setup_agents()
        # Agents are now accessible as self.agents.planning_agent, etc.

    def run(self, task_description: str, output_file: str | None = None) -> dict:
        self.setup()

        # Access agents via AgentSlots
        planning_result = self._run_single_agent(
            self.agents.planning_agent, task_description
        )
        coding_result = self._run_single_agent(
            self.agents.coding_agent, task_description
        )

        self.cleanup()
        return {"results": [planning_result, coding_result]}
```

### Running a Playground

```python
from pathlib import Path

# Create playground
playground = MyPlayground(config_path=Path("configs/my_config/config.yaml"))

# Set run directory
playground.set_run_dir("runs/my_run_001")

# Run task
result = playground.run("Complete the following task...")

print(f"Status: {result['status']}")
print(f"Steps: {result['steps']}")
```

## Configuration File Format

### config.yaml

```yaml
# LLM Configuration
llm:
  openai:
    provider: "openai"
    model: "gpt-4"
    api_key: "${OPENAI_API_KEY}"
    temperature: 0.7
  default: "openai"

# Session Configuration
session:
  type: "local"  # or "docker"
  local:
    workspace_path: "./workspace"

# Multi-Agent Configuration (v0.0.2)
agents:
  Solver:
    llm: "openai"                  # per-agent LLM binding
    max_turns: 50
    tools:                          # per-agent tool config
      builtin: ["*"]               # all builtin tools
    skills:                         # per-agent skills
      - "*"                        # load all skills
  Critic:
    llm: "openai"
    max_turns: 30
    tools:
      builtin: []                  # no tools (pure text response)
    # No skills key -> no skills loaded
  Coder:
    llm: "openai"
    max_turns: 100
    tools:
      builtin: ["execute_bash", "str_replace_editor", "finish"]
      mcp: "mcp_config.json"      # per-agent MCP config
    skills:
      - "rag"
      - "pdf"

# Global Skills Configuration (root directory)
skills:
  enabled: false
  skills_root: "evomaster/skills"

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Related Documentation

- [Architecture Overview](./architecture.md)
- [Agent Module](./agent.md)
- [Tools Module](./tools.md)
