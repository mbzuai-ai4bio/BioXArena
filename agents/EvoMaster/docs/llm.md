# LLM Module

The LLM module provides a unified abstraction layer for interacting with different LLM providers.

## Overview

```
evomaster/utils/
└── llm.py           # LLM classes and factory
```

## Supported Providers

- **OpenAI**: GPT-4, GPT-3.5-turbo, and compatible APIs
- **Anthropic**: Claude series models
- **DeepSeek**: DeepSeek models with Chat and Completion APIs
- **OpenRouter**: Via OpenAI-compatible interface

## LLMConfig

Configuration for LLM instances.

```python
class LLMConfig(BaseModel):
    """LLM configuration"""
    provider: Literal["openai", "anthropic", "deepseek", "openrouter"]
    model: str = Field(description="Model name")
    api_key: str = Field(description="API Key, required in config")
    base_url: str | None = Field(default=None, description="API Base URL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, description="Max generation tokens")
    timeout: int = Field(default=300, description="Request timeout (seconds)")
    max_retries: int = Field(default=3, description="Max retry attempts")
    retry_delay: float = Field(default=1.0, description="Retry delay (seconds)")
    use_completion_api: bool = Field(default=False, description="Use Completion API instead of Chat API")
```

## LLMResponse

Response from LLM API calls.

```python
class LLMResponse(BaseModel):
    """LLM response"""
    content: str | None = Field(default=None, description="Generated text content")
    tool_calls: list[ToolCall] | None = Field(default=None, description="Tool call list")
    finish_reason: str | None = Field(default=None, description="Finish reason")
    usage: dict[str, int] = Field(default_factory=dict, description="Token usage stats")
    meta: dict[str, Any] = Field(default_factory=dict, description="Other metadata")

    def to_assistant_message(self) -> AssistantMessage:
        """Convert to AssistantMessage"""
```

## BaseLLM

Abstract base class for all LLM implementations.

```python
class BaseLLM(ABC):
    """LLM base class

    Defines unified LLM call interface.
    """

    def __init__(self, config: LLMConfig, output_config: dict[str, Any] | None = None):
        """Initialize LLM

        Args:
            config: LLM configuration
            output_config: Output display config:
                - show_in_console: Whether to display in terminal
                - log_to_file: Whether to log to file
        """

    def query(
        self,
        dialog: Dialog,
        **kwargs: Any,
    ) -> AssistantMessage:
        """Query LLM

        Args:
            dialog: Dialog object
            **kwargs: Extra parameters (override config)

        Returns:
            Assistant message
        """

    @abstractmethod
    def _call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call LLM API (subclass implementation)

        Args:
            messages: Message list (API format)
            tools: Tool spec list (API format)
            **kwargs: Extra parameters

        Returns:
            LLM response
        """

    def _call_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call with retry and exponential backoff"""

    def _convert_tools(self, tool_specs: list) -> list[dict[str, Any]]:
        """Convert tool specs to API format"""
```

## OpenAILLM

OpenAI API implementation, also supports compatible APIs (vLLM, Ollama, etc.).

```python
class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation

    Supports OpenAI API and compatible interfaces.
    """

    def _setup(self) -> None:
        """Set up OpenAI client"""

    def _call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call OpenAI API"""
```

## AnthropicLLM

Anthropic Claude API implementation.

```python
class AnthropicLLM(BaseLLM):
    """Anthropic LLM implementation

    Supports Claude series models.
    """

    def _setup(self) -> None:
        """Set up Anthropic client"""

    def _call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call Anthropic API

        Note: Anthropic requires separating system messages.
        """
```

## DeepSeekLLM

DeepSeek API implementation with Chat and Completion API support.

```python
class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM implementation

    Supports Chat Completion API and Completion API.
    """

    def _call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call DeepSeek API

        Uses Chat API by default, Completion API if use_completion_api=True.
        """

    def _call_chat(self, messages, tools, **kwargs) -> LLMResponse:
        """Call Chat Completion API"""

    def _call_completion(self, messages, **kwargs) -> LLMResponse:
        """Call Completion API (for DeepSeek R1)"""

    def _messages_to_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Convert messages to single prompt string (for Completion API)"""
```

## Factory Function

```python
def create_llm(config: LLMConfig, output_config: dict[str, Any] | None = None) -> BaseLLM:
    """LLM factory function

    Args:
        config: LLM configuration
        output_config: Output display config

    Returns:
        LLM instance

    Raises:
        ValueError: Unsupported provider
    """
```

## Utility Function

```python
def truncate_content(
    content: str,
    max_length: int = 5000,
    head_length: int = 2500,
    tail_length: int = 2500
) -> str:
    """Truncate content, keeping head and tail if exceeds max length

    Args:
        content: Content to truncate
        max_length: Max length threshold
        head_length: Head portion length to keep
        tail_length: Tail portion length to keep

    Returns:
        Truncated content
    """
```

## Usage Examples

### Basic Usage

```python
from evomaster.utils import LLMConfig, create_llm
from evomaster.utils.types import Dialog, SystemMessage, UserMessage

# Create LLM
config = LLMConfig(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key",
    temperature=0.7,
)
llm = create_llm(config)

# Create dialog
dialog = Dialog(
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="Hello, how are you?"),
    ]
)

# Query LLM
response = llm.query(dialog)
print(response.content)
```

### With Tool Calling

```python
from evomaster.utils.types import ToolSpec, FunctionSpec

# Create dialog with tools
dialog = Dialog(
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What's the weather in Tokyo?"),
    ],
    tools=[
        ToolSpec(
            type="function",
            function=FunctionSpec(
                name="get_weather",
                description="Get weather for a city",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }
            )
        )
    ]
)

response = llm.query(dialog)

if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call.function.name}")
        print(f"Args: {tool_call.function.arguments}")
```

### Using Different Providers

```python
# OpenAI
openai_llm = create_llm(LLMConfig(
    provider="openai",
    model="gpt-4-turbo",
    api_key="sk-...",
))

# Anthropic
anthropic_llm = create_llm(LLMConfig(
    provider="anthropic",
    model="claude-3-opus-20240229",
    api_key="sk-ant-...",
))

# DeepSeek
deepseek_llm = create_llm(LLMConfig(
    provider="deepseek",
    model="deepseek-chat",
    api_key="sk-...",
    base_url="https://api.deepseek.com",
))

# DeepSeek R1 (Completion API)
deepseek_r1 = create_llm(LLMConfig(
    provider="deepseek",
    model="deepseek-reasoner",
    api_key="sk-...",
    base_url="https://api.deepseek.com",
    use_completion_api=True,
))

# OpenRouter
openrouter_llm = create_llm(LLMConfig(
    provider="openrouter",
    model="anthropic/claude-3-opus",
    api_key="sk-or-...",
    base_url="https://openrouter.ai/api/v1",
))
```

### With Output Logging

```python
llm = create_llm(
    config,
    output_config={
        "show_in_console": True,   # Print to terminal
        "log_to_file": True,       # Log to file
    }
)
```

## Configuration in YAML

```yaml
llm:
  openai:
    provider: "openai"
    model: "gpt-4-turbo"
    api_key: "your-api-key"
    temperature: 0.7
    max_tokens: 4096
    timeout: 300

  anthropic:
    provider: "anthropic"
    model: "claude-3-opus-20240229"
    api_key: "your-api-key"
    temperature: 0.7

  deepseek:
    provider: "deepseek"
    model: "deepseek-chat"
    api_key: "your-api-key"
    base_url: "https://api.deepseek.com"

llm_output:
  show_in_console: false
  log_to_file: true
```

## Related Documentation

- [Architecture Overview](./architecture.md)
- [Agent Module](./agent.md)
- [Core Module](./core.md)
