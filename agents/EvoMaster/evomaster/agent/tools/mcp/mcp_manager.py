"""MCP tool manager.

Responsible for MCP connection initialization, tool registration, and lifecycle management.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..base import ToolRegistry
    from .mcp import MCPTool


class MCPToolManager:
    """MCP tool manager.

    Manages MCP server connections and tool registration.
    Uses a hybrid approach (Approach C):
    - Externally: Registers tools to the unified ToolRegistry
    - Internally: Independently manages MCP connections and tools

    Responsibilities:
    1. Manage MCP server connections
    2. Create MCPTool instances
    3. Organize tools by server
    4. Register to ToolRegistry
    5. Lifecycle management (add/remove servers)

    Usage example:
        manager = MCPToolManager()

        # Add MCP server
        await manager.add_server(
            name="github",
            transport="stdio",
            command="python",
            args=["mcp_servers/github_server.py"]
        )

        # Register to ToolRegistry
        manager.register_tools(tool_registry)

        # Cleanup
        await manager.cleanup()
    """

    def __init__(self):
        """Initialize the MCP tool manager."""
        # MCP connections: {server_name: MCPConnection}
        self.connections: dict[str, Any] = {}

        # Tools organized by server: {server_name: {tool_name: MCPTool}}
        self.tools_by_server: dict[str, dict[str, MCPTool]] = {}

        # The ToolRegistry this manager has registered to (for subsequent tool removal)
        self._registered_registry: ToolRegistry | None = None

        self.logger = logging.getLogger(self.__class__.__name__)
        self.loop: asyncio.AbstractEventLoop | None = None

        # server runner task：{name: asyncio.Task}
        self._server_tasks: dict[str, asyncio.Task] = {}

        # stop signal：{name: asyncio.Event}
        self._server_stop: dict[str, asyncio.Event] = {}

        # ready signal：{name: asyncio.Event}
        self._server_ready: dict[str, asyncio.Event] = {}

        # Optional path adaptor: transform tool arguments before execution
        # Use case: convert local paths to remote URLs, inject credentials, etc.
        # Set via playground._configure_mcp_manager() hook
        self.path_adaptor_servers: set[str] = set()
        self.path_adaptor_factory: Any = None

        # Optional per-server tool filter: only register specified tools (by original names)
        # If set for a server, tools not in the list are excluded
        # Example: {"server1": ["tool-a", "tool-b"]} -> only load tool-a and tool-b from server1
        self.tool_include_only: dict[str, list[str]] = {}

    def _build_tools(self, server_name: str, connection: Any, tools_info: list[dict]) -> None:
        """Build MCPTool instances for a server.

        Args:
            server_name: Name of the MCP server.
            connection: MCP connection instance.
            tools_info: List of tool information dicts from the server.
        """
        from .mcp import MCPTool

        include_only = self.tool_include_only.get(server_name)
        if include_only is not None:
            tools_info = [t for t in tools_info if t.get("name") in include_only]
            self.logger.info(f"Filtered to {len(tools_info)} tools for server '{server_name}' (include_only: {include_only})")

        server_tools: dict[str, MCPTool] = {}
        for tool_info in tools_info:
            original_name = tool_info["name"]
            prefixed_name = f"{server_name}_{original_name}"

            mcp_tool = MCPTool(
                mcp_connection=connection,
                tool_name=prefixed_name,
                tool_description=tool_info.get("description", ""),
                input_schema=tool_info.get("input_schema", {}),
                remote_tool_name=original_name,
            )
            mcp_tool._mcp_server = server_name
            mcp_tool._mcp_loop = self.loop  # Preserve original loop injection logic
            if self.path_adaptor_servers and self.path_adaptor_factory and server_name in self.path_adaptor_servers:
                mcp_tool._path_adaptor = self.path_adaptor_factory()

            server_tools[prefixed_name] = mcp_tool

        self.tools_by_server[server_name] = server_tools

    async def add_server(self, name: str, transport: str, **connection_kwargs) -> None:
        """Add an MCP server and load its tools.

        Args:
            name: Server name.
            transport: Transport method (e.g., "stdio", "sse", "http").
            **connection_kwargs: Additional connection arguments.
        """
        if name in self._server_tasks:
            raise ValueError(f"MCP server '{name}' already exists")

        if self.loop is None:
            # The manager.loop must already be set to a long-running loop
            raise RuntimeError("MCPToolManager.loop is None. Set a long-running event loop before add_server().")

        self.logger.info(f"Adding MCP server: {name} ({transport})")

        stop_evt = asyncio.Event()
        ready_evt = asyncio.Event()
        self._server_stop[name] = stop_evt
        self._server_ready[name] = ready_evt

        async def runner():
            from .mcp_connection import create_connection
            try:
                async with create_connection(transport=transport, **connection_kwargs) as conn:
                    # Enter completed within the runner task
                    self.connections[name] = conn

                    tools_info = await conn.list_tools()
                    self.logger.info(f"Found {len(tools_info)} tools from MCP server '{name}'")

                    self._build_tools(name, conn, tools_info)

                    # If already registered to a ToolRegistry, register the new tools immediately
                    if self._registered_registry:
                        for tool in self.tools_by_server[name].values():
                            self._registered_registry.register(tool)

                    ready_evt.set()

                    # Suspend until remove_server sends the stop signal
                    await stop_evt.wait()

            except Exception as e:
                # If the runner fails to start, ensure ready_evt is set to avoid blocking the caller
                self.logger.error(f"MCP server runner failed for '{name}': {e}")
                ready_evt.set()
                raise
            finally:
                # After exiting async with, the connection is already closed
                # Do not delete dict entries in finally; leave that to remove_server for unified cleanup
                pass

        # The runner task must be created inside self.loop
        if asyncio.get_running_loop() is not self.loop:
            raise RuntimeError(
                "add_server() must be called inside MCP loop. "
                "Use run_coroutine_threadsafe(...) to submit it to manager.loop."
            )

        task = asyncio.create_task(runner())


        self._server_tasks[name] = task

        # Wait for tools to finish loading
        await ready_evt.wait()
        if task.done() and (exc := task.exception()) is not None:
            # Clean up registration entries
            self._server_tasks.pop(name, None)
            self._server_stop.pop(name, None)
            self._server_ready.pop(name, None)
            self.connections.pop(name, None)
            self.tools_by_server.pop(name, None)
            raise exc
        self.logger.info(f"Successfully added MCP server '{name}'")


    def register_tools(self, tool_registry: ToolRegistry) -> None:
        """Register all MCP tools to a ToolRegistry.

        Args:
            tool_registry: Target tool registry.
        """
        self._registered_registry = tool_registry

        total_count = 0
        for server_name, tools in self.tools_by_server.items():
            for tool_name, tool in tools.items():
                tool_registry.register(tool)
                total_count += 1
                self.logger.debug(f"Registered MCP tool: {tool_name} (from {server_name})")

        self.logger.info(f"Registered {total_count} MCP tools to ToolRegistry")

    async def remove_server(self, server_name: str) -> None:
        """Remove an MCP server and unregister its tools.

        Args:
            server_name: Server name.
        """
        if asyncio.get_running_loop() is not self.loop:
            raise RuntimeError("remove_server() must be called inside MCP loop.")
        if server_name not in self._server_tasks:
            raise ValueError(f"MCP server '{server_name}' not found")

        self.logger.info(f"Removing MCP server: {server_name}")

        # 1) Remove tools from ToolRegistry
        if self._registered_registry and server_name in self.tools_by_server:
            for tool_name in list(self.tools_by_server[server_name].keys()):
                self._registered_registry.unregister(tool_name)

        # 2) Let the runner exit its async with block (__aexit__ executes within the same task)
        stop_evt = self._server_stop.get(server_name)
        if stop_evt:
            stop_evt.set()

        task = self._server_tasks.get(server_name)
        if task:
            await task  # Wait for clean exit

        # 3) Clean up local records
        self._server_tasks.pop(server_name, None)
        self._server_stop.pop(server_name, None)
        self._server_ready.pop(server_name, None)

        self.connections.pop(server_name, None)
        tool_count = len(self.tools_by_server.get(server_name, {}))
        self.tools_by_server.pop(server_name, None)

        self.logger.info(f"Removed {tool_count} tools from server '{server_name}'")

    async def reload_server(self, server_name: str) -> None:
        """Reload tools from an MCP server.

        Used for hot-reloading tools.

        Args:
            server_name: Server name.

        Raises:
            ValueError: Server not found.
        """
        if server_name not in self.connections:
            raise ValueError(f"MCP server '{server_name}' not found")

        self.logger.info(f"Reloading MCP server: {server_name}")

        # Save connection config (simplified; in practice, full config may need to be saved)
        connection = self.connections[server_name]

        # Remove and re-add
        # Note: Simplified handling; in practice, the original config should be saved
        await self.remove_server(server_name)

        # Re-fetch tools
        tools_info = await connection.list_tools()

        # Re-create tools (similar to add_server logic)
        from .mcp import MCPTool

        server_tools = {}
        for tool_info in tools_info:
            original_name = tool_info["name"]
            prefixed_name = f"{server_name}_{original_name}"

            mcp_tool = MCPTool(
                mcp_connection=connection,
                tool_name=prefixed_name,
                tool_description=tool_info.get("description", ""),
                input_schema=tool_info.get("input_schema", {}),
            )
            mcp_tool._mcp_server = server_name

            server_tools[prefixed_name] = mcp_tool

        self.tools_by_server[server_name] = server_tools
        self.connections[server_name] = connection

        # Re-register
        if self._registered_registry:
            for tool in server_tools.values():
                self._registered_registry.register(tool)

        self.logger.info(f"Reloaded {len(server_tools)} tools from server '{server_name}'")

    async def cleanup(self) -> None:
        """Clean up all MCP connections."""
        self.logger.info("Cleaning up MCP connections")
        
        if asyncio.get_running_loop() is not self.loop:
            raise RuntimeError("cleanup() must be called inside MCP loop.")

        failed: list[str] = []

        for name in list(self._server_tasks.keys()):
            try:
                await self.remove_server(name)
            except Exception as e:
                failed.append(name)
                self.logger.warning(f"Error cleaning up MCP server '{name}': {e}")

        if failed:
            self.logger.warning(f"MCP cleanup incomplete; failed servers: {failed}")
            return

        self.connections.clear()
        self.tools_by_server.clear()
        self._registered_registry = None

        self.logger.info("MCP cleanup complete")

    def get_tool_names(self) -> list[str]:
        """Get all MCP tool names.

        Returns:
            List of tool names.
        """
        names = []
        for tools in self.tools_by_server.values():
            names.extend(tools.keys())
        return names

    def get_server_names(self) -> list[str]:
        """Get all MCP server names.

        Returns:
            List of server names.
        """
        return list(self.connections.keys())

    def get_tools_by_server(self, server_name: str) -> list[MCPTool]:
        """Get all tools from a specific server.

        Args:
            server_name: Server name.

        Returns:
            List of tools.
        """
        return list(self.tools_by_server.get(server_name, {}).values())

    def get_stats(self) -> dict[str, Any]:
        """Get statistics.

        Returns:
            Statistics dictionary.
        """
        stats = {
            "total_servers": len(self.connections),
            "total_tools": len(self.get_tool_names()),
            "servers": {}
        }

        for server_name, tools in self.tools_by_server.items():
            server_stats = {
                "tool_count": len(tools),
                "tools": {}
            }
            for tool_name, tool in tools.items():
                server_stats["tools"][tool_name] = tool.get_stats()
            stats["servers"][server_name] = server_stats

        return stats
