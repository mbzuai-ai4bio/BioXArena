"""EvoMaster Agent Tools base class.

Provides base abstractions and registration mechanisms for tools.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from evomaster.utils.types import FunctionSpec, ToolSpec
    from evomaster.agent.session import BaseSession
    from evomaster.skills import SkillRegistry


class ToolError(Exception):
    """Tool execution error."""
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ToolParameterError(ToolError):
    """Tool parameter error."""
    
    def __init__(self, param_name: str, value: Any, hint: str = ""):
        self.param_name = param_name
        self.value = value
        message = f"Invalid parameter `{param_name}`: {value}"
        if hint:
            message += f". {hint}"
        super().__init__(message)


def _remove_unused_schema_info(schema: dict, model: type[BaseModel]) -> None:
    """Remove unnecessary information from the schema to make it more concise."""
    def _remove_recursive(schema: dict, keys: list[str]):
        for key in keys:
            schema.pop(key, None)
        for _, v in schema.items():
            if isinstance(v, dict):
                _remove_recursive(v, keys)

    _remove_recursive(schema, ["default", "title", "additionalProperties"])
    schema.pop("description", None)


class BaseToolParams(BaseModel):
    """Base class for tool parameters.

    All tool parameter classes should inherit from this class and define:
    - name: ClassVar[str] - Tool name (exposed to LLM)
    - __doc__: Tool description (used as function description)
    """
    
    name: ClassVar[str]
    model_config = ConfigDict(
        json_schema_extra=_remove_unused_schema_info,
    )


class BaseTool(ABC):
    """Base class for tools.

    Each tool requires:
    1. A parameter class (inheriting BaseToolParams)
    2. An implemented execute method
    """

    # Tool name
    name: ClassVar[str]

    # Parameter class
    params_class: ClassVar[type[BaseToolParams]]
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """Execute the tool.

        Args:
            session: Environment session.
            args_json: Parameter JSON string.

        Returns:
            (observation, info) tuple:
            - observation: Observation result returned to the Agent.
            - info: Additional information.
        """
        pass

    def parse_params(self, args_json: str) -> BaseToolParams:
        """Parse parameters.

        Args:
            args_json: Parameters in JSON string format.

        Returns:
            Parsed parameter object.
        """
        return self.params_class.model_validate_json(args_json)

    def get_description(self) -> str:
        """Get tool description (used for the LLM function calling description field).

        By default extracts from the params_class docstring.
        Subclasses can override this method to provide dynamic descriptions.
        """
        return (self.params_class.__doc__ or "").strip().replace("\n    ", "\n")

    def get_tool_spec(self) -> ToolSpec:
        """Get the tool specification (used for LLM function calling)."""
        from evomaster.utils.types import FunctionSpec, ToolSpec

        return ToolSpec(
            type="function",
            function=FunctionSpec(
                name=self.name,
                description=self.get_description(),
                parameters=self.params_class.model_json_schema(),
                strict=None,
            )
        )


class ToolRegistry:
    """Tool registry.

    Manages all available tools with support for dynamic registration and retrieval.
    """
    
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def register(self, tool: BaseTool) -> None:
        """Register a tool.

        Args:
            tool: Tool instance.
        """
        if tool.name in self._tools:
            self.logger.warning(f"Tool {tool.name} already registered, overwriting")
        self._tools[tool.name] = tool
        self.logger.debug(f"Registered tool: {tool.name}")

    def register_many(self, tools: list[BaseTool]) -> None:
        """Register multiple tools at once."""
        for tool in tools:
            self.register(tool)

    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            self.logger.debug(f"Unregistered tool: {name}")

    def get_tool(self, name: str) -> BaseTool | None:
        """Get a tool by name.

        Args:
            name: Tool name.

        Returns:
            Tool instance, or None if not found.
        """
        return self._tools.get(name)

    def get_all_tools(self) -> list[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_tool_names(self) -> list[str]:
        """Get all tool names."""
        return list(self._tools.keys())

    def get_tool_specs(self) -> list[ToolSpec]:
        """Get the specification list for all tools (used for LLM)."""
        return [tool.get_tool_spec() for tool in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    # MCP tool-related methods (supporting hybrid approach C)

    def get_mcp_tools(self) -> list[BaseTool]:
        """Get all MCP tools.

        Identifies MCP tools by checking the _is_mcp_tool attribute.

        Returns:
            List of MCP tools.
        """
        return [
            tool for tool in self._tools.values()
            if getattr(tool, '_is_mcp_tool', False)
        ]

    def get_builtin_tools(self) -> list[BaseTool]:
        """Get all built-in tools (non-MCP tools).

        Returns:
            List of built-in tools.
        """
        return [
            tool for tool in self._tools.values()
            if not getattr(tool, '_is_mcp_tool', False)
        ]

    def get_tools_by_server(self, server_name: str) -> list[BaseTool]:
        """Get all tools from a specific MCP server.

        Args:
            server_name: MCP server name.

        Returns:
            List of tools from that server.
        """
        return [
            tool for tool in self._tools.values()
            if getattr(tool, '_mcp_server', None) == server_name
        ]

    def get_mcp_server_names(self) -> list[str]:
        """Get all MCP server names.

        Returns:
            Deduplicated list of server names.
        """
        servers = set()
        for tool in self._tools.values():
            server = getattr(tool, '_mcp_server', None)
            if server:
                servers.add(server)
        return sorted(list(servers))



def create_default_registry(skill_registry: SkillRegistry | None = None) -> ToolRegistry:
    """Create the default tool registry containing all built-in tools.

    Args:
        skill_registry: Optional SkillRegistry instance; if provided, SkillTool is registered.
    """
    return create_registry(builtin_names=["*"], skill_registry=skill_registry)


# Mapping of all builtin tool names to factory functions
_BUILTIN_TOOL_FACTORIES: dict[str, Any] = None  # type: ignore[assignment]


def _get_builtin_factories() -> dict[str, Any]:
    """Lazily load the builtin tool factory mapping (name -> no-arg constructor)."""
    global _BUILTIN_TOOL_FACTORIES
    if _BUILTIN_TOOL_FACTORIES is None:
        from .builtin import BashTool, EditorTool, ThinkTool, FinishTool
        _BUILTIN_TOOL_FACTORIES = {
            "execute_bash": BashTool,
            "str_replace_editor": EditorTool,
            "think": ThinkTool,
            "finish": FinishTool,
        }
    return _BUILTIN_TOOL_FACTORIES


ALL_BUILTIN_TOOL_NAMES: list[str] | None = None  # populated lazily


def get_all_builtin_tool_names() -> list[str]:
    """Return all builtin tool names."""
    global ALL_BUILTIN_TOOL_NAMES
    if ALL_BUILTIN_TOOL_NAMES is None:
        ALL_BUILTIN_TOOL_NAMES = list(_get_builtin_factories().keys())
    return ALL_BUILTIN_TOOL_NAMES


def create_registry(
    builtin_names: list[str] | None = None,
    skill_registry: SkillRegistry | None = None,
    openclaw_bridge: object | None = None,
    enabled_skills: list[str] | None = None,
) -> ToolRegistry:
    """Create a tool registry with optional filtering of builtin tools by name.

    Args:
        builtin_names: List of builtin tool names to register.
            - None or ["*"] -> Register all builtin tools
            - [] -> Do not register any builtin (skill / MCP only)
            - ["execute_bash", "finish"] -> Register only specified tools
        skill_registry: Optional SkillRegistry instance; if provided, SkillTool is registered.
        openclaw_bridge: Optional OpenclawBridge instance, passed to SkillTool.
        enabled_skills: Optional list of skill names from config. If provided and non-empty,
            skill_context exposes only these skills to the agent; otherwise all are exposed.
            Execution still uses the full registry.
    """
    factories = _get_builtin_factories()

    registry = ToolRegistry()
    tools: list[BaseTool] = []

    # Determine which builtin tools to instantiate
    if builtin_names is None or builtin_names == ["*"]:
        # All
        tools.extend(factory() for factory in factories.values())
    else:
        for name in builtin_names:
            if name == "*":
                # Mixing in "*" is equivalent to all
                tools = [factory() for factory in factories.values()]
                break
            if name not in factories:
                raise ValueError(
                    f"Unknown builtin tool '{name}'. "
                    f"Available: {list(factories.keys())}"
                )
            tools.append(factories[name]())

    # If a skill_registry is provided and contains skills, register SkillTool
    if skill_registry is not None and skill_registry.get_all_skills():
        from .skill import SkillTool
        tools.append(SkillTool(skill_registry, bridge=openclaw_bridge, enabled_skills=enabled_skills))

    registry.register_many(tools)
    return registry

