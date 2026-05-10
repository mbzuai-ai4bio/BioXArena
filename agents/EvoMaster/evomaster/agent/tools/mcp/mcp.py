"""MCP tool integration.

Wraps MCP (Model Context Protocol) server tools as EvoMaster tools.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from ..base import BaseTool, ToolError

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession
    from evomaster.utils.types import ToolSpec


class MCPTool(BaseTool):
    """MCP tool wrapper.

    Wraps a single MCP tool as an EvoMaster BaseTool.

    Features:
    - Dynamic tools: Obtained from MCP servers at runtime
    - Async-to-sync: MCP is asynchronous and needs conversion
    - Schema conversion: MCP schema -> ToolSpec
    - Metadata tagging: Tags tool origin (MCP server)

    Usage example:
        mcp_tool = MCPTool(
            mcp_connection=connection,
            tool_name="github_create_issue",
            tool_description="Create a new GitHub issue",
            input_schema={...}
        )
        observation, info = mcp_tool.execute(session, args_json)
    """

    # Class attributes (required by BaseTool)
    name: ClassVar[str] = "mcp_tool"  # Will be overridden by instance attribute
    params_class: ClassVar[type] = None  # MCP tools do not use params_class

    def __init__(
        self,
        mcp_connection,  # MCPConnection instance
        tool_name: str,
        tool_description: str,
        input_schema: dict,
        remote_tool_name: str | None = None,
    ):
        """Initialize the MCP tool.

        Args:
            mcp_connection: MCP connection instance.
            tool_name: Tool name (with server prefix added).
            tool_description: Tool description.
            input_schema: Input parameter schema (JSON Schema format).
        """
        super().__init__()

        # MCP-related attributes
        self.mcp_connection = mcp_connection
        self._tool_name = tool_name
        self._tool_description = tool_description
        self._input_schema = input_schema
        self._remote_tool_name = remote_tool_name
        # Dedicated MCP event loop (injected by MCPToolManager or Playground)
        self._mcp_loop = None

        # Override class attribute
        self.name = tool_name

        # Metadata tags (set by MCPToolManager)
        self._is_mcp_tool = True
        self._mcp_server = None  # Server name

        # Statistics
        self._call_count = 0
        self._last_error = None

    def execute(
        self,
        session: BaseSession,
        args_json: str
    ) -> tuple[str, dict[str, Any]]:
        """Execute the MCP tool.

        Args:
            session: Session instance (not used by MCP tools, kept for interface consistency).
            args_json: JSON-formatted arguments.

        Returns:
            (observation, info) tuple:
            - observation: Observation result returned to the Agent.
            - info: Additional information (including MCP metadata).
        """
        try:
            # 1. Parse arguments
            args = json.loads(args_json)
            self.logger.debug(f"Executing MCP tool {self._tool_name} with args: {args}")

            # 2. Apply path adaptor (if configured via playground hook)
            # Transforms arguments before sending to MCP tool (e.g., path conversion, credential injection)
            path_adaptor = getattr(self, "_path_adaptor", None)
            if path_adaptor is not None:
                workspace_path = (
                    getattr(getattr(session, "config", None), "workspace_path", None)
                    if session
                    else None
                ) or ""
                args = path_adaptor.resolve_args(
                    workspace_path,
                    args,
                    self._tool_name,
                    self._mcp_server or "",
                    input_schema=getattr(self, "_input_schema", None),
                )

            # 3. Call MCP tool (async-to-sync)
            result = self._call_mcp_tool_sync(args)

            # 4. Format output
            observation = self._format_mcp_result(result)

            # 5. Update statistics
            self._call_count += 1
            self._last_error = None

            info = {
                "mcp_tool": self._tool_name,
                "mcp_server": self._mcp_server,
                "success": True,
                "call_count": self._call_count,
            }

            return observation, info

        except json.JSONDecodeError as e:
            self._last_error = str(e)
            raise ToolError(f"Invalid JSON arguments: {str(e)}")
        except Exception as e:
            self._last_error = str(e)
            self.logger.error(f"MCP tool {self._tool_name} failed: {e}")
            raise ToolError(f"MCP tool execution failed: {str(e)}")

    def _call_mcp_tool_sync(self, args: dict) -> Any:
        """Synchronously call an MCP tool (handles async internally).

        IMPORTANT: Do not use asyncio.run() which creates a temporary loop. The coroutine
        must be submitted to the same long-lived loop to avoid anyio/mcp stream deadlocks,
        ClosedResourceError, and cancel scope misalignment.
        """
        # 1) A persistent MCP loop must have been injected
        loop = getattr(self, "_mcp_loop", None)
        if loop is None:
            raise ToolError(
                "MCP loop not injected into MCPTool. "
                "Please set mcp_tool._mcp_loop = <persistent_event_loop> when creating tools."
            )
        if loop.is_closed():
            raise ToolError("MCP loop is closed; cannot call MCP tool")

        coro = self.mcp_connection.call_tool(self._remote_tool_name, args)

        try:
            # 2) If the loop is not currently running (most common: synchronous Agent scenario), run directly
            if not loop.is_running():
                return loop.run_until_complete(coro)

            # 3) If the loop is running (e.g. run_forever in a background thread), use thread-safe submission
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            return fut.result(timeout=60)

        except concurrent.futures.TimeoutError:
            raise ToolError("MCP tool call timed out after 60 seconds")
        except Exception as e:
            raise ToolError(f"Failed to call MCP tool: {str(e)}")

    def _format_mcp_result(self, result: Any) -> str:
        """Format the MCP tool return result.

        MCP returns a content list; text content needs to be extracted.
        Supports multiple content types: text, json, image, etc.

        Args:
            result: Raw result returned by the MCP tool.

        Returns:
            Formatted string.
        """
        if isinstance(result, list):
            # MCP returns a content list
            parts = []
            for item in result:
                # Handle different content types
                if hasattr(item, 'text'):
                    # Pydantic model
                    parts.append(item.text)
                elif isinstance(item, dict):
                    if 'text' in item:
                        parts.append(item['text'])
                    elif 'type' in item and item['type'] == 'text':
                        parts.append(item.get('text', ''))
                    else:
                        # Other content types, convert to JSON
                        parts.append(json.dumps(item, indent=2))
                else:
                    parts.append(str(item))
            return "\n".join(parts) if parts else ""
        elif isinstance(result, str):
            return result
        elif result is None:
            return ""
        else:
            # Other types, convert to JSON
            return json.dumps(result, indent=2, default=str)

    def get_tool_spec(self) -> ToolSpec:
        """Get tool specification (used for LLM function calling).

        Converts MCP schema to EvoMaster ToolSpec.

        Returns:
            ToolSpec instance.
        """
        from evomaster.utils.types import FunctionSpec, ToolSpec

        return ToolSpec(
            type="function",
            function=FunctionSpec(
                name=self._tool_name,
                description=self._tool_description,
                parameters=self._input_schema,
                strict=None,
            )
        )

    def get_stats(self) -> dict[str, Any]:
        """Get tool statistics.

        Returns:
            Statistics dictionary.
        """
        return {
            "tool_name": self._tool_name,
            "mcp_server": self._mcp_server,
            "call_count": self._call_count,
            "last_error": self._last_error,
        }
