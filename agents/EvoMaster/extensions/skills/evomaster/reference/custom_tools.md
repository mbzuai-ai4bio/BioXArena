# Custom Tools and MCP Tools

## Overview

EvoMaster supports three categories of external tools beyond the built-in set:

1. **Custom Tools** — User-defined Python tools loaded from the playground directory
2. **MCP Tools** — Tools provided by external MCP (Model Context Protocol) servers
3. **Openclaw Tools** — TypeScript-based plugin tools executed via a Node.js bridge

## Custom Tools

### Creating a Custom Tool

Every custom tool consists of two classes: a **parameter class** and a **tool class**.

#### Step 1: Define the Parameter Class

```python
# playground/{your_playground}/tools/my_tool.py

from evomaster.agent.tools.base import BaseTool, BaseToolParams
from pydantic import Field
from typing import ClassVar, Any

class MyToolParams(BaseToolParams):
    """Search the web and return relevant results.

    This docstring becomes the tool description shown to the LLM.
    Keep it concise and action-oriented.
    """
    name: ClassVar[str] = "my_tool"

    query: str = Field(description="The search query string")
    max_results: int = Field(default=5, description="Maximum number of results to return")
```

Key requirements:
- Inherit from `BaseToolParams` (which inherits from Pydantic `BaseModel`)
- Set `name: ClassVar[str]` — this is the tool name the LLM sees
- The class docstring becomes the function description in the LLM's tool spec
- Use `Field(description=...)` for each parameter — these become parameter descriptions

#### Step 2: Define the Tool Class

```python
class MyTool(BaseTool):
    name: ClassVar[str] = "my_tool"
    params_class: ClassVar[type[BaseToolParams]] = MyToolParams

    def execute(self, session, args_json: str) -> tuple[str, dict[str, Any]]:
        params = self.parse_params(args_json)

        # Your tool logic here
        results = do_search(params.query, params.max_results)

        # Return (observation_string, info_dict)
        return f"Found {len(results)} results:\n{results}", {"status": "ok"}
```

Key requirements:
- Inherit from `BaseTool`
- Set `name` to match the params class name
- Set `params_class` to your parameter class
- Implement `execute(self, session, args_json)` returning `(str, dict)`
  - `session`: A `BaseSession` instance (provides `exec_bash`, `read_file`, `write_file`)
  - `args_json`: Raw JSON string from the LLM
  - Return tuple: `(observation, info)` where `observation` is returned to the agent

#### Step 3: Register in Configuration

```yaml
agents:
  my_agent:
    tools:
      builtin: ["*"]
      my_tool: "my_tool"    # key: arbitrary label, value: Python filename (without .py)
```

The framework auto-discovers the tool:
1. Infers the playground directory from the config path
2. Looks for `playground/{name}/tools/{value}.py`
3. Scans the module for classes inheriting `BaseTool`
4. Instantiates and registers the tool

### Advanced: Constructor Arguments

If your tool needs initialization parameters, override `_create_custom_tool_instance()` in your playground:

```python
class MyPlayground(BasePlayground):
    def _create_custom_tool_instance(self, tool_class, tool_name, tool_key):
        if tool_name == "my_tool":
            return tool_class(api_key=os.environ["MY_API_KEY"])
        return super()._create_custom_tool_instance(tool_class, tool_name, tool_key)
```

### Tool Execution Model

```
LLM generates tool_call → Agent parses args_json
    → ToolRegistry.get_tool(name) → tool.execute(session, args_json)
    → (observation, info) returned → observation added to conversation
    → LLM sees the observation and continues reasoning
```

## MCP Tools

### What is MCP?

MCP (Model Context Protocol) is an open standard for connecting AI agents to external tool providers. EvoMaster supports all three MCP transport types:

| Transport | Use Case | Config Key |
|---|---|---|
| **stdio** | Local process (e.g., `npx my-server`) | `command`, `args`, `env` |
| **http** / **streamable_http** | HTTP endpoint | `transport: "http"`, `url` |
| **sse** | Server-Sent Events | `transport: "sse"`, `url` |

### MCP Configuration

Create a JSON file (e.g., `mcp_config.json`) in your config directory:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-filesystem", "/workspace"]
    },
    "my-api": {
      "transport": "http",
      "url": "http://localhost:8080/mcp",
      "headers": {
        "Authorization": "Bearer ${MY_API_TOKEN}"
      }
    },
    "realtime-data": {
      "transport": "sse",
      "url": "http://data-server:9090/events"
    }
  }
}
```

### Enabling MCP in Config

```yaml
agents:
  my_agent:
    tools:
      builtin: ["*"]
      mcp: "mcp_config.json"       # Relative to config directory
      # mcp: "*"                    # Use default mcp_config.json
      # mcp: ""                     # Disable MCP (default)
```

### How MCP Tools Are Loaded

1. `BasePlayground._setup_mcp_tools()` reads the JSON config
2. Creates an `MCPToolManager` with a dedicated asyncio event loop (in a daemon thread)
3. For each server, creates an `MCPConnection` via the appropriate transport
4. Calls `connection.list_tools()` to discover available tools
5. Wraps each tool as an `MCPTool` instance (inherits `BaseTool`)
6. Tool names are prefixed: `{server_name}_{original_tool_name}`
7. All MCP tools are registered in the shared `ToolRegistry`

### MCP Tool Naming

If server `filesystem` exposes tools `read_file` and `write_file`:
- Registered as `filesystem_read_file` and `filesystem_write_file`
- The LLM sees these names and can call them like any other tool

### Path Adaptation

MCP tools may need path translation (e.g., local paths ↔ container paths in Docker sessions). The `MCPToolManager` supports a `path_adaptor_factory` hook:

```python
class MyPlayground(BasePlayground):
    def _configure_mcp_manager(self, manager, mcp_config):
        manager.path_adaptor_factory = lambda: MyPathAdaptor()
```

### Placeholder Replacement

MCP configs support the `__EVOMASTER_WORKSPACES__` placeholder, which is automatically replaced with the actual workspaces directory path at runtime (useful for batch task mode).

### MCP Lifecycle

- MCP connections are initialized once and reused across agents
- The asyncio event loop runs in a dedicated daemon thread
- `cleanup()` properly shuts down all MCP connections and the event loop

## Openclaw Tools (TypeScript Bridge)

Openclaw tools are TypeScript-based plugin skills executed through a Node.js subprocess bridge.

### Configuration

```yaml
agents:
  my_agent:
    tools:
      openclaw:
        enabled: true
        skills_ts_dir: "./evomaster/skills_ts"
        plugins: ["feishu"]
```

### How It Works

1. An `OpenclawBridge` subprocess is started (`evomaster/skills_ts/bridge/server.ts`)
2. The bridge loads specified plugins and exposes their tools
3. When the agent calls a `use_skill` action with `type: "openclaw"`, the `SkillTool` delegates to the bridge
4. Arguments are passed as JSON, results are returned as strings

### Available Openclaw Plugins

| Plugin | Tools | Description |
|---|---|---|
| `feishu` | `feishu_doc`, `feishu_drive`, `feishu_wiki`, `feishu_chat`, `feishu_perm`, `feishu_bitable` | Feishu/Lark workspace operations |

## Built-in Tools Reference

| Tool Name | Description | Key Parameters |
|---|---|---|
| `execute_bash` | Execute shell commands | `command` (str) |
| `str_replace_editor` | File operations (view, create, edit) | `command` (view/create/str_replace/insert/undo_edit), `path`, `file_text`/`old_str`/`new_str` |
| `think` | Record reasoning (no side effects) | `thought` (str) |
| `finish` | Signal task completion | `message` (str) |
| `use_skill` | Interact with Skills | `skill_name`, `action` (get_info/get_reference/run_script), etc. |
