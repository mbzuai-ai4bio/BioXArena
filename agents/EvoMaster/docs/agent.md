# Agent Module

The Agent module provides the core intelligent components of EvoMaster, including Agent, Context Management, and Session.

## Overview

```
evomaster/agent/
├── agent.py          # BaseAgent, Agent classes
├── context.py        # Context management
├── session/          # Session implementations
│   ├── base.py       # BaseSession
│   ├── local.py      # LocalSession
│   └── docker.py     # DockerSession
└── tools/            # Tool system (see tools.md)
```

## BaseAgent

`BaseAgent` is the abstract base class for all agents.

### Class Definition

```python
class BaseAgent(ABC):
    """Agent base class providing:
    - Dialog management
    - Trajectory recording
    - Tool call execution
    - Context management
    """

    VERSION: str = "1.0"
```

### Constructor

```python
def __init__(
    self,
    llm: BaseLLM,
    session: BaseSession,
    tools: ToolRegistry,
    config: AgentConfig | None = None,
    skill_registry: SkillRegistry | None = None,
    output_config: dict[str, Any] | None = None,
    config_dir: Path | str | None = None,
    enable_tools: bool = True,
    enabled_tool_names: list[str] | None = None,
)
```

**Parameters:**
- `llm`: LLM instance for model queries
- `session`: Environment session for tool execution
- `tools`: Tool registry (always registered, but only shown in prompt if `enable_tools=True`)
- `config`: Agent configuration
- `skill_registry`: Optional skill registry
- `output_config`: Output display configuration
- `config_dir`: Config directory path for loading prompt files
- `enable_tools`: Whether to include tool info in prompts (default `True`). If `False`, tools are still registered but not exposed to the LLM
- `enabled_tool_names`: List of tool names to expose to the LLM (optional). `None` or `["*"]` means all registered tools are enabled. Only affects the tools visible to the LLM, not tools callable from code

### Key Methods

#### run(task, on_step)
```python
def run(self, task: TaskInstance, on_step=None) -> Trajectory:
    """Execute a task

    Args:
        task: Task instance
        on_step: Optional step callback, signature (StepRecord, step_number, max_steps) -> None

    Returns:
        Execution trajectory
    """
```

#### continue_run(user_message, on_step)
```python
def continue_run(self, user_message: str, on_step=None) -> Trajectory:
    """Continue execution with a new user message on existing dialog

    Unlike run(), does not call _initialize(), preserves existing dialog context.
    Suitable for multi-turn conversation scenarios.

    Args:
        user_message: New user message
        on_step: Optional step callback

    Returns:
        Execution trajectory for this round
    """
```

#### load_prompt_from_file(prompt_file, format_kwargs)
```python
def load_prompt_from_file(
    self,
    prompt_file: str | Path,
    format_kwargs: dict[str, Any] | None = None,
) -> str:
    """Load prompt from file with optional formatting

    Args:
        prompt_file: Path to prompt file (relative or absolute)
        format_kwargs: Dictionary for string formatting

    Returns:
        Formatted prompt content
    """
```

#### reset_context()
```python
def reset_context(self) -> None:
    """Reset agent context to initial state"""
```

#### add_user_message(content)
```python
def add_user_message(self, content: str) -> None:
    """Add user message to current dialog"""
```

### Abstract Methods (must implement)

```python
@abstractmethod
def _get_system_prompt(self) -> str:
    """Get system prompt"""

@abstractmethod
def _get_user_prompt(self, task: TaskInstance) -> str:
    """Get user prompt for task"""
```

### Instance and Class Methods

```python
def set_trajectory_file_path(self, trajectory_file_path: str | Path) -> None:
    """Set trajectory file path (per-instance, each agent independent)"""

@classmethod
def set_exp_info(cls, exp_name: str, exp_index: int) -> None:
    """Set current exp info for trajectory recording (class-level, shared by all instances)"""

def set_agent_name(self, name: str) -> None:
    """Set agent name (used to identify different agents in trajectory files)"""
```

## Agent

`Agent` is the standard implementation of `BaseAgent`.

### Constructor

```python
def __init__(
    self,
    llm: BaseLLM,
    session: BaseSession,
    tools: ToolRegistry,
    system_prompt_file: str | Path | None = None,
    user_prompt_file: str | Path | None = None,
    prompt_format_kwargs: dict[str, Any] | None = None,
    config: AgentConfig | None = None,
    skill_registry: SkillRegistry | None = None,
    output_config: dict[str, Any] | None = None,
    config_dir: Path | str | None = None,
    enable_tools: bool = True,
    enabled_tool_names: list[str] | None = None,
)
```

## AgentConfig

```python
class AgentConfig(BaseModel):
    """Agent configuration"""
    max_turns: int = Field(default=100, description="Maximum execution turns")
    context_config: ContextConfig = Field(
        default_factory=ContextConfig,
        description="Context management config"
    )
    finish_on_text_response: bool = Field(
        default=False,
        description="Finish when LLM replies with pure text (no tool call), suitable for chat scenarios"
    )
```

## Context Management

### ContextConfig

```python
class ContextConfig(BaseModel):
    """Context management configuration"""
    max_tokens: int = Field(default=128000, description="Maximum tokens")
    truncation_strategy: TruncationStrategy = Field(
        default=TruncationStrategy.LATEST_HALF,
        description="Truncation strategy"
    )
    preserve_system_messages: bool = Field(default=True)
    preserve_recent_turns: int = Field(default=5)
```

### TruncationStrategy

```python
class TruncationStrategy(str, Enum):
    """History truncation strategy"""
    NONE = "none"                    # No truncation
    LATEST_HALF = "latest_half"      # Keep latest half
    SLIDING_WINDOW = "sliding_window" # Sliding window
    SUMMARY = "summary"               # Summary compression
```

### ContextManager

```python
class ContextManager:
    """Context manager for dialog history"""

    def estimate_tokens(self, dialog: Dialog) -> int:
        """Estimate token count for dialog"""

    def should_truncate(self, dialog: Dialog) -> bool:
        """Check if truncation is needed"""

    def truncate(self, dialog: Dialog) -> Dialog:
        """Truncate dialog based on strategy"""

    def prepare_for_query(self, dialog: Dialog) -> Dialog:
        """Prepare dialog for LLM query"""
```

## Session

Session is the interface between Agent and environment.

### BaseSession

```python
class BaseSession(ABC):
    """Session abstract base class"""

    @abstractmethod
    def open(self) -> None:
        """Open session, establish connection"""

    @abstractmethod
    def close(self) -> None:
        """Close session, release resources"""

    @abstractmethod
    def exec_bash(
        self,
        command: str,
        timeout: int | None = None,
        is_input: bool = False,
    ) -> dict[str, Any]:
        """Execute bash command

        Returns:
            Dict with stdout, stderr, exit_code, working_dir
        """

    @abstractmethod
    def upload(self, local_path: str, remote_path: str) -> None:
        """Upload file to remote environment"""

    @abstractmethod
    def download(self, remote_path: str, timeout: int | None = None) -> bytes:
        """Download file from remote environment"""

    # Convenience methods
    def read_file(self, remote_path: str, encoding: str = "utf-8") -> str
    def write_file(self, remote_path: str, content: str, encoding: str = "utf-8") -> None
    def path_exists(self, remote_path: str) -> bool
    def is_file(self, remote_path: str) -> bool
    def is_directory(self, remote_path: str) -> bool
```

### SessionConfig

```python
class SessionConfig(BaseModel):
    """Session base configuration"""
    timeout: int = Field(default=300, description="Default timeout (seconds)")
    workspace_path: str = Field(default="/workspace", description="Workspace path")
```

### LocalSession

For local environment execution:

```python
class LocalSessionConfig(SessionConfig):
    working_dir: str = Field(default=".")
```

### DockerSession

For Docker container execution:

```python
class DockerSessionConfig(SessionConfig):
    image: str = Field(description="Docker image name")
    working_dir: str = Field(default="/workspace")
    volumes: dict[str, str] = Field(default_factory=dict)
    auto_remove: bool = Field(default=True)
```

## Usage Examples

### Basic Agent Usage

```python
from evomaster.agent import Agent, AgentConfig, create_registry
from evomaster.agent.session import LocalSession, LocalSessionConfig
from evomaster.utils import LLMConfig, create_llm
from evomaster.utils.types import TaskInstance

# Create components
llm = create_llm(LLMConfig(provider="openai", model="gpt-4", api_key="..."))
session = LocalSession(LocalSessionConfig(workspace_path="./workspace"))
tools = create_registry()  # creates default registry with all builtin tools

# Create agent
agent = Agent(
    llm=llm,
    session=session,
    tools=tools,
    config=AgentConfig(max_turns=50),
)

# Run task
session.open()
try:
    task = TaskInstance(task_id="001", task_type="discovery", description="Find patterns...")
    trajectory = agent.run(task)
finally:
    session.close()
```

### Custom Agent with Prompts

```python
agent = Agent(
    llm=llm,
    session=session,
    tools=tools,
    system_prompt_file="prompts/system.txt",
    user_prompt_file="prompts/user.txt",
    prompt_format_kwargs={"domain": "physics"},
    config_dir=Path("./configs/my_agent"),
)
```

### Agent without Tools

```python
# For agents that only provide answers without tool calls
agent = Agent(
    llm=llm,
    session=session,
    tools=tools,
    enable_tools=False,  # Tools registered but not shown in prompt
)
```

## Related Documentation

- [Architecture Overview](./architecture.md)
- [Tools Module](./tools.md)
- [Core Module](./core.md)
