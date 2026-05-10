"""EvoMaster Utils module

Utility classes and helper functions, including:
- LLM interface wrappers
- Core type definitions
- Other general-purpose utilities
"""

from .llm import (
    BaseLLM,
    LLMConfig,
    LLMResponse,
    OpenAILLM,
    AnthropicLLM,
    create_llm,
    build_multimodal_content,
    encode_image_to_base64,
    get_image_media_type,
)

from .types import (
    # Message types
    MessageRole,
    BaseMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    Message,
    # Function/Tool definitions
    FunctionCall,
    ToolCall,
    FunctionSpec,
    ToolSpec,
    # Dialog and Trajectory
    Dialog,
    StepRecord,
    Trajectory,
    TaskInstance,
)

__all__ = [
    # LLM
    "BaseLLM",
    "LLMConfig",
    "LLMResponse",
    "OpenAILLM",
    "AnthropicLLM",
    "create_llm",
    "build_multimodal_content",
    "encode_image_to_base64",
    "get_image_media_type",
    # Types
    "MessageRole",
    "BaseMessage",
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
]
