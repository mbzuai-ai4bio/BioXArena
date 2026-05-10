# Tools 模块

Tools 模块为 Agent 提供工具系统，包括内置工具和 MCP 集成。

## 概述

```
evomaster/agent/tools/
├── base.py           # BaseTool, ToolRegistry
├── builtin/          # 内置工具
│   ├── bash.py       # BashTool
│   ├── editor.py     # EditorTool
│   ├── think.py      # ThinkTool
│   └── finish.py     # FinishTool
├── skill.py          # SkillTool
└── mcp/              # MCP 集成
    ├── mcp.py            # MCPTool
    ├── mcp_connection.py # 连接处理
    └── mcp_manager.py    # MCPToolManager
```

## BaseTool

所有工具的抽象基类。

### 类定义

```python
class BaseTool(ABC):
    """工具基类

    每个工具需要：
    1. 定义参数类（继承 BaseToolParams）
    2. 实现 execute 方法
    """

    # 工具名称
    name: ClassVar[str]

    # 参数类
    params_class: ClassVar[type[BaseToolParams]]
```

### 方法

```python
@abstractmethod
def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
    """执行工具

    Args:
        session: 环境会话
        args_json: 参数 JSON 字符串

    Returns:
        (observation, info) 元组
        - observation: 返回给 Agent 的结果
        - info: 额外信息
    """

def parse_params(self, args_json: str) -> BaseToolParams:
    """从 JSON 字符串解析参数"""

def get_tool_spec(self) -> ToolSpec:
    """获取 LLM function calling 的工具规格"""
```

## BaseToolParams

工具参数基类。

```python
class BaseToolParams(BaseModel):
    """工具参数基类

    子类应定义：
    - name: ClassVar[str] - 工具名称（暴露给 LLM）
    - __doc__: 工具描述（作为 function description）
    """

    name: ClassVar[str]
```

## ToolRegistry

管理所有可用工具的注册表。

```python
class ToolRegistry:
    """工具注册中心"""

    def register(self, tool: BaseTool) -> None:
        """注册工具"""

    def register_many(self, tools: list[BaseTool]) -> None:
        """批量注册工具"""

    def unregister(self, name: str) -> None:
        """取消注册工具"""

    def get_tool(self, name: str) -> BaseTool | None:
        """按名称获取工具"""

    def get_all_tools(self) -> list[BaseTool]:
        """获取所有已注册的工具"""

    def get_tool_names(self) -> list[str]:
        """获取所有工具名称"""

    def get_tool_specs(self) -> list[ToolSpec]:
        """获取所有工具规格用于 LLM"""

    # MCP 相关方法
    def get_mcp_tools(self) -> list[BaseTool]:
        """获取所有 MCP 工具"""

    def get_builtin_tools(self) -> list[BaseTool]:
        """获取所有内置工具（非 MCP）"""

    def get_tools_by_server(self, server_name: str) -> list[BaseTool]:
        """获取特定 MCP 服务器的工具"""

    def get_mcp_server_names(self) -> list[str]:
        """获取所有 MCP 服务器名称"""
```

### 工厂函数

```python
def create_registry(
    builtin_names: list[str] | None = None,
    skill_registry: SkillRegistry | None = None,
    openclaw_bridge: object | None = None,
    enabled_skills: list[str] | None = None,
) -> ToolRegistry:
    """创建工具注册表，支持按名称筛选 builtin 工具

    Args:
        builtin_names: 需要注册的 builtin 工具名称列表。
            - None 或 ["*"] -> 注册全部 builtin 工具
            - [] -> 不注册任何 builtin（仅 skill / MCP）
            - ["execute_bash", "finish"] -> 仅注册指定工具
        skill_registry: 可选的 SkillRegistry，如果提供则注册 SkillTool
        openclaw_bridge: 可选的 OpenclawBridge 实例，传递给 SkillTool
        enabled_skills: 可选，配置的 skill 名称列表
    """

def create_default_registry(skill_registry: SkillRegistry | None = None) -> ToolRegistry:
    """创建默认的工具注册表，包含所有内置工具

    是 create_registry(builtin_names=["*"], ...) 的便捷封装。

    Args:
        skill_registry: 可选的 SkillRegistry，如果提供则注册 SkillTool
    """
```

## 内置工具

### BashTool

执行 bash 命令。

```python
class BashToolParams(BaseToolParams):
    """在环境中执行 bash 命令。

    示例：
        {"command": "ls -la"}
        {"command": "python script.py", "timeout": 60}
    """
    name: ClassVar[str] = "execute_bash"
    command: str = Field(description="要执行的 bash 命令")
    timeout: int | None = Field(default=None, description="超时时间（秒）")
```

### EditorTool

查看、创建和编辑文件。

```python
class EditorToolParams(BaseToolParams):
    """使用 str_replace_editor 查看、创建和编辑文件。

    命令：
    - view: 查看文件内容（带行号）
    - create: 创建新文件
    - str_replace: 替换文件中的文本（old_str 必须唯一）
    - insert: 在某行后插入文本
    - undo_edit: 撤销上次编辑
    """
    name: ClassVar[str] = "str_replace_editor"
    command: Literal["view", "create", "str_replace", "insert", "undo_edit"]
    path: str = Field(description="文件路径")
    # 不同命令的可选字段...
```

### ThinkTool

思考问题（不影响环境）。

```python
class ThinkToolParams(BaseToolParams):
    """思考问题。不影响环境。"""
    name: ClassVar[str] = "think"
    thought: str = Field(description="对当前问题的思考")
```

### FinishTool

标志任务完成。

```python
class FinishToolParams(BaseToolParams):
    """标志你已完成任务。"""
    name: ClassVar[str] = "finish"
    result: str = Field(description="最终结果或答案")
    success: bool = Field(default=True, description="任务是否成功完成")
```

## SkillTool

与 Skill 系统交互的工具。

```python
class SkillToolParams(BaseToolParams):
    """使用技能获取信息或执行操作。

    操作：
    - get_info: 获取技能的详细信息
    - get_reference: 获取参考文档
    - run_script: 运行 Operator 技能的脚本
    """
    name: ClassVar[str] = "use_skill"
    action: Literal["get_info", "get_reference", "run_script"]
    skill_name: str = Field(description="技能名称")
    reference_name: str | None = Field(default=None, description="get_reference 的参考名称")
    script_name: str | None = Field(default=None, description="run_script 的脚本名称")
    script_args: str | None = Field(default=None, description="脚本参数")
```

## MCP 集成

### MCPTool

MCP 服务器工具的包装器。

```python
class MCPTool(BaseTool):
    """MCP 工具包装器

    将远程 MCP 工具包装为本地工具。
    工具名称添加服务器前缀：{server}_{tool_name}
    """

    def __init__(
        self,
        mcp_connection: MCPConnection,
        tool_name: str,
        tool_description: str,
        input_schema: dict,
        remote_tool_name: str | None = None,
    ):
        """初始化 MCP 工具

        Args:
            mcp_connection: MCP 连接实例
            tool_name: 带前缀的工具名称（如 "github_create_issue"）
            tool_description: 工具描述
            input_schema: 工具输入 schema
            remote_tool_name: MCP 服务器上的原始工具名称
        """

    # 属性
    _is_mcp_tool: bool = True
    _mcp_server: str | None = None
    _mcp_loop: asyncio.AbstractEventLoop | None = None
```

### MCPToolManager

MCP 服务器连接和工具的管理器。

```python
class MCPToolManager:
    """MCP 工具管理器

    管理 MCP 服务器连接和工具注册。
    采用混合方案：
    - 对外：注册到统一的 ToolRegistry
    - 对内：独立管理 MCP 连接和工具

    职责：
    1. 管理 MCP 服务器连接
    2. 创建 MCPTool 实例
    3. 按服务器组织工具
    4. 注册到 ToolRegistry
    5. 生命周期管理（添加/移除服务器）
    """

    async def add_server(self, name: str, transport: str, **connection_kwargs) -> None:
        """添加 MCP 服务器

        Args:
            name: 服务器名称
            transport: 传输类型（"stdio", "http", "sse"）
            **connection_kwargs: 连接参数
        """

    def register_tools(self, tool_registry: ToolRegistry) -> None:
        """将所有 MCP 工具注册到 ToolRegistry"""

    async def remove_server(self, server_name: str) -> None:
        """移除 MCP 服务器及其工具"""

    async def reload_server(self, server_name: str) -> None:
        """重新加载 MCP 服务器工具（热重载）"""

    async def cleanup(self) -> None:
        """清理所有 MCP 连接"""

    def get_tool_names(self) -> list[str]:
        """获取所有 MCP 工具名称"""

    def get_server_names(self) -> list[str]:
        """获取所有 MCP 服务器名称"""

    def get_tools_by_server(self, server_name: str) -> list[MCPTool]:
        """获取特定服务器的工具"""

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
```

### MCP 配置

#### mcp_config.json

```json
{
  "mcpServers": {
    "sandbox": {
      "transport": "sse",
      "url": "http://localhost:8001/sse"
    },
    "search": {
      "command": "python",
      "args": ["mcp_servers/search_server.py"],
      "env": {
        "API_KEY": "your-key"
      }
    },
    "github": {
      "transport": "http",
      "url": "http://localhost:8080/mcp",
      "headers": {
        "Authorization": "Bearer token"
      }
    }
  }
}
```

## 使用示例

### 创建自定义工具

```python
from evomaster.agent.tools import BaseTool, BaseToolParams
from pydantic import Field
from typing import ClassVar, Any

class MyToolParams(BaseToolParams):
    """我的自定义工具描述。

    做一些有用的事情。
    """
    name: ClassVar[str] = "my_tool"
    input_data: str = Field(description="要处理的输入数据")
    option: bool = Field(default=False, description="可选标志")

class MyTool(BaseTool):
    name: ClassVar[str] = "my_tool"
    params_class: ClassVar[type[BaseToolParams]] = MyToolParams

    def execute(self, session, args_json: str) -> tuple[str, dict[str, Any]]:
        params = self.parse_params(args_json)

        # 执行逻辑
        result = f"已处理: {params.input_data}"

        return result, {"status": "success"}

# 注册工具
registry = create_registry()
registry.register(MyTool())
```

也可以创建仅包含特定内置工具的注册表：

```python
# 仅注册 bash 和 finish 工具，加上自定义工具
registry = create_registry(builtin_names=["execute_bash", "finish"])
registry.register(MyTool())
```

### 使用 MCP 工具

```python
from evomaster.agent.tools import MCPToolManager

# 创建管理器
manager = MCPToolManager()

# 添加服务器
await manager.add_server(
    name="github",
    transport="stdio",
    command="python",
    args=["mcp_servers/github_server.py"]
)

# 注册到 ToolRegistry
registry = create_registry()
manager.register_tools(registry)

# 现在所有 MCP 工具都可用，名称为 github_*
print(registry.get_tool_names())
# ['execute_bash', 'str_replace_editor', 'think', 'finish', 'github_create_issue', ...]
```

## 相关文档

- [架构概述](./architecture.md)
- [Agent 模块](./agent.md)
- [Skills 模块](./skills.md)
