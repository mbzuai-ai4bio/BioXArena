"""EvoMaster Agent Tools module.

Provides an extensible tool system that supports Agents in calling various tools to complete tasks.

Directory structure:
- base.py: Tool base class and registry
- builtin/: Built-in tools (Bash, Editor, Think, Finish)
- mcp/: MCP protocol tool support
"""

from .base import BaseTool, ToolRegistry, ToolError, create_default_registry, create_registry

# Built-in tools
from .builtin import (
    BashTool,
    BashToolParams,
    EditorTool,
    EditorToolParams,
    ThinkTool,
    ThinkToolParams,
    FinishTool,
    FinishToolParams,
)

# MCP tools
from .mcp import (
    MCPTool,
    MCPToolManager,
    MCPConnection,
    create_connection,
)

from .skill import SkillTool, SkillToolParams

__all__ = [
    # Base
    "BaseTool",
    "ToolRegistry",
    "ToolError",
    "create_default_registry",
    "create_registry",
    # Builtin Tools
    "BashTool",
    "BashToolParams",
    "EditorTool",
    "EditorToolParams",
    "ThinkTool",
    "ThinkToolParams",
    "FinishTool",
    "FinishToolParams",
    #Skill Tools
    "SkillTool",
    "SkillToolParams",
    # MCP Tools
    "MCPTool",
    "MCPToolManager",
    "MCPConnection",
    "create_connection",
]
