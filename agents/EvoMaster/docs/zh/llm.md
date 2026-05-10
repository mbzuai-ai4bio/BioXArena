# LLM 模块

LLM 模块提供统一的抽象层，用于与不同的 LLM 提供商交互。

## 概述

```
evomaster/utils/
└── llm.py           # LLM 类和工厂函数
```

## 支持的提供商

- **OpenAI**：GPT-4、GPT-3.5-turbo 及兼容 API
- **Anthropic**：Claude 系列模型
- **DeepSeek**：支持 Chat 和 Completion API 的 DeepSeek 模型
- **OpenRouter**：通过 OpenAI 兼容接口

## LLMConfig

LLM 实例的配置。

```python
class LLMConfig(BaseModel):
    """LLM 配置"""
    provider: Literal["openai", "anthropic", "deepseek", "openrouter"]
    model: str = Field(description="模型名称")
    api_key: str = Field(description="API Key，必须在配置中提供")
    base_url: str | None = Field(default=None, description="API Base URL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, description="最大生成 token 数")
    timeout: int = Field(default=300, description="请求超时时间（秒）")
    max_retries: int = Field(default=3, description="最大重试次数")
    retry_delay: float = Field(default=1.0, description="重试延迟（秒）")
    use_completion_api: bool = Field(default=False, description="使用 Completion API 而非 Chat API")
```

## LLMResponse

LLM API 调用的响应。

```python
class LLMResponse(BaseModel):
    """LLM 响应"""
    content: str | None = Field(default=None, description="生成的文本内容")
    tool_calls: list[ToolCall] | None = Field(default=None, description="工具调用列表")
    finish_reason: str | None = Field(default=None, description="结束原因")
    usage: dict[str, int] = Field(default_factory=dict, description="Token 使用统计")
    meta: dict[str, Any] = Field(default_factory=dict, description="其他元数据")

    def to_assistant_message(self) -> AssistantMessage:
        """转换为 AssistantMessage"""
```

## BaseLLM

所有 LLM 实现的抽象基类。

```python
class BaseLLM(ABC):
    """LLM 基类

    定义统一的 LLM 调用接口。
    """

    def __init__(self, config: LLMConfig, output_config: dict[str, Any] | None = None):
        """初始化 LLM

        Args:
            config: LLM 配置
            output_config: 输出显示配置：
                - show_in_console: 是否在终端显示
                - log_to_file: 是否记录到日志文件
        """

    def query(
        self,
        dialog: Dialog,
        **kwargs: Any,
    ) -> AssistantMessage:
        """查询 LLM

        Args:
            dialog: 对话对象
            **kwargs: 额外参数（覆盖配置）

        Returns:
            助手消息
        """

    @abstractmethod
    def _call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """调用 LLM API（子类实现）

        Args:
            messages: 消息列表（API 格式）
            tools: 工具规格列表（API 格式）
            **kwargs: 额外参数

        Returns:
            LLM 响应
        """

    def _call_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """带重试和指数退避的调用"""

    def _convert_tools(self, tool_specs: list) -> list[dict[str, Any]]:
        """转换工具规格为 API 格式"""
```

## OpenAILLM

OpenAI API 实现，也支持兼容 API（vLLM、Ollama 等）。

```python
class OpenAILLM(BaseLLM):
    """OpenAI LLM 实现

    支持 OpenAI API 和兼容接口。
    """

    def _setup(self) -> None:
        """设置 OpenAI 客户端"""

    def _call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """调用 OpenAI API"""
```

## AnthropicLLM

Anthropic Claude API 实现。

```python
class AnthropicLLM(BaseLLM):
    """Anthropic LLM 实现

    支持 Claude 系列模型。
    """

    def _setup(self) -> None:
        """设置 Anthropic 客户端"""

    def _call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """调用 Anthropic API

        注意：Anthropic 需要分离系统消息。
        """
```

## DeepSeekLLM

DeepSeek API 实现，支持 Chat 和 Completion API。

```python
class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM 实现

    支持 Chat Completion API 和 Completion API。
    """

    def _call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """调用 DeepSeek API

        默认使用 Chat API，如果 use_completion_api=True 则使用 Completion API。
        """

    def _call_chat(self, messages, tools, **kwargs) -> LLMResponse:
        """调用 Chat Completion API"""

    def _call_completion(self, messages, **kwargs) -> LLMResponse:
        """调用 Completion API（用于 DeepSeek R1）"""

    def _messages_to_prompt(self, messages: list[dict[str, Any]]) -> str:
        """将消息转换为单个 prompt 字符串（用于 Completion API）"""
```

## 工厂函数

```python
def create_llm(config: LLMConfig, output_config: dict[str, Any] | None = None) -> BaseLLM:
    """LLM 工厂函数

    Args:
        config: LLM 配置
        output_config: 输出显示配置

    Returns:
        LLM 实例

    Raises:
        ValueError: 不支持的提供商
    """
```

## 工具函数

```python
def truncate_content(
    content: str,
    max_length: int = 5000,
    head_length: int = 2500,
    tail_length: int = 2500
) -> str:
    """截断内容，如果超过最大长度则保留开头和结尾

    Args:
        content: 要截断的内容
        max_length: 最大长度阈值
        head_length: 保留的开头部分长度
        tail_length: 保留的结尾部分长度

    Returns:
        截断后的内容
    """
```

## 使用示例

### 基本用法

```python
from evomaster.utils import LLMConfig, create_llm
from evomaster.utils.types import Dialog, SystemMessage, UserMessage

# 创建 LLM
config = LLMConfig(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key",
    temperature=0.7,
)
llm = create_llm(config)

# 创建对话
dialog = Dialog(
    messages=[
        SystemMessage(content="你是一个有帮助的助手。"),
        UserMessage(content="你好，你好吗？"),
    ]
)

# 查询 LLM
response = llm.query(dialog)
print(response.content)
```

### 带工具调用

```python
from evomaster.utils.types import ToolSpec, FunctionSpec

# 创建带工具的对话
dialog = Dialog(
    messages=[
        SystemMessage(content="你是一个有帮助的助手。"),
        UserMessage(content="东京的天气怎么样？"),
    ],
    tools=[
        ToolSpec(
            type="function",
            function=FunctionSpec(
                name="get_weather",
                description="获取城市天气",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
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
        print(f"工具: {tool_call.function.name}")
        print(f"参数: {tool_call.function.arguments}")
```

### 使用不同提供商

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

# DeepSeek R1（Completion API）
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

### 带输出日志

```python
llm = create_llm(
    config,
    output_config={
        "show_in_console": True,   # 打印到终端
        "log_to_file": True,       # 记录到文件
    }
)
```

## YAML 配置

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

## 相关文档

- [架构概述](./architecture.md)
- [Agent 模块](./agent.md)
- [Core 模块](./core.md)
