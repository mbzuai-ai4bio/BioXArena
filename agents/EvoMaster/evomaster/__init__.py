"""EvoMaster - Iterative Scientific Experiment Agent System

EvoMaster is an agent system for iteratively completing scientific experiment tasks,
primarily targeting MLE, Physics, Embodied AI, and other scientific experiment scenarios.

Core components (three-layer architecture):
- agent: Intelligent agent (includes Session, Tools)
- env: Environment (cluster scheduling, Docker sandbox)
- skills: Skills
"""

__version__ = "0.1.0"

# Export commonly used classes from the agent module
from evomaster.agent import (
    # Agent
    BaseAgent,
    Agent,
    AgentConfig,
    # Types
    Dialog,
    Message,
    Trajectory,
    TaskInstance,
    # Session
    BaseSession,
    SessionConfig,
    DockerSession,
    DockerSessionConfig,
    # Tools
    BaseTool,
    ToolRegistry,
    create_default_registry,
)

# Export utility classes and types from the utils module
from evomaster.utils import (
    # LLM
    BaseLLM,
    LLMConfig,
    LLMResponse,
    OpenAILLM,
    AnthropicLLM,
    create_llm,
    # Types
    MessageRole,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    Message,
    FunctionCall,
    ToolCall,
    FunctionSpec,
    ToolSpec,
    Dialog,
    StepRecord,
    Trajectory,
    TaskInstance,
)

# Export configuration management from the config module
from evomaster.config import (
    # Config base class
    BaseConfig,
    # Env config
    # EnvConfig,
    ClusterConfig,
    ClusterPoolConfig,
    DockerEnvConfig,
    SchedulerConfig,
    # Logging config
    LoggingConfig,
    # Top-level config
    EvoMasterConfig,
    # Config manager
    ConfigManager,
    get_config_manager,
    load_config,
    get_config,
)

__all__ = [
    # Agent
    "BaseAgent",
    "Agent",
    "AgentConfig",
    # Types (from utils)
    "MessageRole",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "Message",
    "FunctionCall",
    "ToolCall",
    "FunctionSpec",
    "ToolSpec",
    "Dialog",
    "StepRecord",
    "Trajectory",
    "TaskInstance",
    # Session
    "BaseSession",
    "SessionConfig",
    "DockerSession",
    "DockerSessionConfig",
    # Tools
    "BaseTool",
    "ToolRegistry",
    "create_default_registry",
    # Utils - LLM
    "BaseLLM",
    "LLMConfig",
    "LLMResponse",
    "OpenAILLM",
    "AnthropicLLM",
    "create_llm",
    # Config
    "BaseConfig",
    "EnvConfig",
    "ClusterConfig",
    "ClusterPoolConfig",
    "DockerEnvConfig",
    "SchedulerConfig",
    "LoggingConfig",
    "EvoMasterConfig",
    "ConfigManager",
    "get_config_manager",
    "load_config",
    "get_config",
]
