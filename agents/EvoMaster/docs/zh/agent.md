# Agent 模块

Agent 模块提供 EvoMaster 的核心智能组件，包括 Agent、上下文管理和 Session。

## 概述

```
evomaster/agent/
├── agent.py          # BaseAgent, Agent 类
├── context.py        # 上下文管理
├── session/          # Session 实现
│   ├── base.py       # BaseSession
│   ├── local.py      # LocalSession
│   └── docker.py     # DockerSession
└── tools/            # 工具系统（见 tools.md）
```

## BaseAgent

`BaseAgent` 是所有 Agent 的抽象基类。

### 类定义

```python
class BaseAgent(ABC):
    """Agent 基类，提供：
    - 对话管理
    - 轨迹记录
    - 工具调用执行
    - 上下文管理
    """

    VERSION: str = "1.0"
```

### 构造函数

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

**参数：**
- `llm`：用于模型查询的 LLM 实例
- `session`：用于工具执行的环境会话
- `tools`：工具注册表（始终注册，但仅在 `enable_tools=True` 时出现在提示词中）
- `config`：Agent 配置
- `skill_registry`：可选的技能注册表
- `output_config`：输出显示配置
- `config_dir`：用于加载提示词文件的配置目录路径
- `enable_tools`：是否在提示词中包含工具信息（默认 `True`）。如果为 `False`，工具仍然注册但不会暴露给 LLM
- `enabled_tool_names`：启用的工具名称列表（可选）。`None` 或 `["*"]` 表示所有已注册工具都启用。仅影响暴露给 LLM 的工具列表，不影响代码中手动调用工具

### 关键方法

#### run(task, on_step)
```python
def run(self, task: TaskInstance, on_step=None) -> Trajectory:
    """执行任务

    Args:
        task: 任务实例
        on_step: 可选的步骤回调，签名 (StepRecord, step_number, max_steps) -> None

    Returns:
        执行轨迹
    """
```

#### continue_run(user_message, on_step)
```python
def continue_run(self, user_message: str, on_step=None) -> Trajectory:
    """在已有对话上追加用户消息，继续执行

    与 run() 的区别：不调用 _initialize()，保留已有对话上下文。
    适用于多轮对话场景。

    Args:
        user_message: 新的用户消息
        on_step: 可选的步骤回调

    Returns:
        本轮执行轨迹
    """
```

#### load_prompt_from_file(prompt_file, format_kwargs)
```python
def load_prompt_from_file(
    self,
    prompt_file: str | Path,
    format_kwargs: dict[str, Any] | None = None,
) -> str:
    """从文件加载提示词，支持可选格式化

    Args:
        prompt_file: 提示词文件路径（相对或绝对）
        format_kwargs: 用于字符串格式化的字典

    Returns:
        格式化后的提示词内容
    """
```

#### reset_context()
```python
def reset_context(self) -> None:
    """重置 Agent 上下文到初始状态"""
```

#### add_user_message(content)
```python
def add_user_message(self, content: str) -> None:
    """添加用户消息到当前对话"""
```

### 抽象方法（必须实现）

```python
@abstractmethod
def _get_system_prompt(self) -> str:
    """获取系统提示词"""

@abstractmethod
def _get_user_prompt(self, task: TaskInstance) -> str:
    """获取任务的用户提示词"""
```

### 实例和类方法

```python
def set_trajectory_file_path(self, trajectory_file_path: str | Path) -> None:
    """设置轨迹文件路径（实例级别，每个 agent 独立）"""

@classmethod
def set_exp_info(cls, exp_name: str, exp_index: int) -> None:
    """设置当前 exp 信息用于轨迹记录（类级别，所有实例共享）"""

def set_agent_name(self, name: str) -> None:
    """设置 Agent 名称（用于在轨迹文件中标识不同的 agent）"""
```

## Agent

`Agent` 是 `BaseAgent` 的标准实现。

### 构造函数

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
    """Agent 配置"""
    max_turns: int = Field(default=100, description="最大执行轮数")
    context_config: ContextConfig = Field(
        default_factory=ContextConfig,
        description="上下文管理配置"
    )
    finish_on_text_response: bool = Field(
        default=False,
        description="当 LLM 回复纯文本（无 tool call）时直接视为任务完成，适用于对话场景"
    )
```

## 上下文管理

### ContextConfig

```python
class ContextConfig(BaseModel):
    """上下文管理配置"""
    max_tokens: int = Field(default=128000, description="最大 token 数")
    truncation_strategy: TruncationStrategy = Field(
        default=TruncationStrategy.LATEST_HALF,
        description="截断策略"
    )
    preserve_system_messages: bool = Field(default=True)
    preserve_recent_turns: int = Field(default=5)
```

### TruncationStrategy

```python
class TruncationStrategy(str, Enum):
    """历史截断策略"""
    NONE = "none"                    # 不截断
    LATEST_HALF = "latest_half"      # 保留最新一半
    SLIDING_WINDOW = "sliding_window" # 滑动窗口
    SUMMARY = "summary"               # 摘要压缩
```

### ContextManager

```python
class ContextManager:
    """对话历史的上下文管理器"""

    def estimate_tokens(self, dialog: Dialog) -> int:
        """估算对话的 token 数"""

    def should_truncate(self, dialog: Dialog) -> bool:
        """检查是否需要截断"""

    def truncate(self, dialog: Dialog) -> Dialog:
        """根据策略截断对话"""

    def prepare_for_query(self, dialog: Dialog) -> Dialog:
        """为 LLM 查询准备对话"""
```

## Session

Session 是 Agent 与环境之间的接口。

### BaseSession

```python
class BaseSession(ABC):
    """Session 抽象基类"""

    @abstractmethod
    def open(self) -> None:
        """打开会话，建立连接"""

    @abstractmethod
    def close(self) -> None:
        """关闭会话，释放资源"""

    @abstractmethod
    def exec_bash(
        self,
        command: str,
        timeout: int | None = None,
        is_input: bool = False,
    ) -> dict[str, Any]:
        """执行 bash 命令

        Returns:
            包含 stdout, stderr, exit_code, working_dir 的字典
        """

    @abstractmethod
    def upload(self, local_path: str, remote_path: str) -> None:
        """上传文件到远程环境"""

    @abstractmethod
    def download(self, remote_path: str, timeout: int | None = None) -> bytes:
        """从远程环境下载文件"""

    # 便捷方法
    def read_file(self, remote_path: str, encoding: str = "utf-8") -> str
    def write_file(self, remote_path: str, content: str, encoding: str = "utf-8") -> None
    def path_exists(self, remote_path: str) -> bool
    def is_file(self, remote_path: str) -> bool
    def is_directory(self, remote_path: str) -> bool
```

### SessionConfig

```python
class SessionConfig(BaseModel):
    """Session 基础配置"""
    timeout: int = Field(default=300, description="默认超时时间（秒）")
    workspace_path: str = Field(default="/workspace", description="工作空间路径")
```

### LocalSession

用于本地环境执行：

```python
class LocalSessionConfig(SessionConfig):
    working_dir: str = Field(default=".")
```

### DockerSession

用于 Docker 容器执行：

```python
class DockerSessionConfig(SessionConfig):
    image: str = Field(description="Docker 镜像名称")
    working_dir: str = Field(default="/workspace")
    volumes: dict[str, str] = Field(default_factory=dict)
    auto_remove: bool = Field(default=True)
```

## 使用示例

### 基本 Agent 使用

```python
from evomaster.agent import Agent, AgentConfig, create_registry
from evomaster.agent.session import LocalSession, LocalSessionConfig
from evomaster.utils import LLMConfig, create_llm
from evomaster.utils.types import TaskInstance

# 创建组件
llm = create_llm(LLMConfig(provider="openai", model="gpt-4", api_key="..."))
session = LocalSession(LocalSessionConfig(workspace_path="./workspace"))
tools = create_registry()  # 创建默认注册表，包含所有内置工具

# 创建 agent
agent = Agent(
    llm=llm,
    session=session,
    tools=tools,
    config=AgentConfig(max_turns=50),
)

# 运行任务
session.open()
try:
    task = TaskInstance(task_id="001", task_type="discovery", description="发现规律...")
    trajectory = agent.run(task)
finally:
    session.close()
```

### 带自定义提示词的 Agent

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

### 不使用工具的 Agent

```python
# 用于只提供答案而不调用工具的 agent
agent = Agent(
    llm=llm,
    session=session,
    tools=tools,
    enable_tools=False,  # 工具已注册但不会出现在提示词中
)
```

## 相关文档

- [架构概述](./architecture.md)
- [Tools 模块](./tools.md)
- [Core 模块](./core.md)
