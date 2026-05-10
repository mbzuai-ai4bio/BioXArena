"""EvoMaster Agent module

Agent is the intelligent agent component of EvoMaster, including:
- Type definitions (Message, Dialog, Trajectory) - imported from utils
- Context management (ContextManager)
- Agent base class and implementations
- Session (interface for interacting with Env)
- Tools (tool system)
"""

from evomaster.utils.types import (
    Message,
    MessageRole,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    ToolCall,
    FunctionCall,
    ToolSpec,
    FunctionSpec,
    Dialog,
    StepRecord,
    Trajectory,
    TaskInstance,
)
from .context import ContextManager, ContextConfig, TruncationStrategy
from .agent import BaseAgent, Agent, AgentConfig

# Session submodule
from .session import (
    BaseSession,
    SessionConfig,
    DockerSession,
    DockerSessionConfig,
)

# Tools submodule
from .tools import (
    BaseTool,
    ToolRegistry,
    ToolError,
    create_default_registry,
    create_registry,
    BashTool,
    EditorTool,
    ThinkTool,
    FinishTool,
)

__all__ = [
    # Types
    "Message",
    "MessageRole",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "ToolCall",
    "FunctionCall",
    "ToolSpec",
    "FunctionSpec",
    "Dialog",
    "StepRecord",
    "Trajectory",
    "TaskInstance",
    # Context
    "ContextManager",
    "ContextConfig",
    "TruncationStrategy",
    # Agent
    "BaseAgent",
    "Agent",
    "AgentConfig",
    # Session
    "BaseSession",
    "SessionConfig",
    "DockerSession",
    "DockerSessionConfig",
    # Tools
    "BaseTool",
    "ToolRegistry",
    "ToolError",
    "create_default_registry",
    "create_registry",
    "BashTool",
    "EditorTool",
    "ThinkTool",
    "FinishTool",
]
