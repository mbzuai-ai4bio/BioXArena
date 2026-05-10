"""Chat Agent Memory Tools

Provides memory_search / memory_save / memory_forget agent tools,
allowing the agent to actively search, save, and delete user long-term memories.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession
    from evomaster.memory.manager import MemoryManager


# ======================================================================
# memory_search
# ======================================================================


class MemorySearchToolParams(BaseToolParams):
    """Search the user's long-term memory for relevant past information.

    Use this when you need to recall something from previous conversations,
    such as user preferences, facts, decisions, or entities mentioned before.
    """

    name: ClassVar[str] = "memory_search"

    query: str = Field(description="Search keywords or description of the information to recall")
    limit: int = Field(default=5, description="Maximum number of results to return")


class MemorySearchTool(BaseTool):
    """Memory search tool."""

    name: ClassVar[str] = "memory_search"
    params_class: ClassVar[type[BaseToolParams]] = MemorySearchToolParams

    def __init__(self, memory_manager: MemoryManager, user_id: str):
        super().__init__()
        self._manager = memory_manager
        self._user_id = user_id

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter error: {e}", {"error": str(e)}

        assert isinstance(params, MemorySearchToolParams)
        entries = self._manager.search(self._user_id, params.query, limit=params.limit)

        if not entries:
            return "No memories found matching the query.", {"count": 0}

        lines = []
        for i, e in enumerate(entries, 1):
            lines.append(
                f"{i}. [{e.category_label}] {e.content}\n"
                f"   (id: {e.id}, importance: {e.importance:.1f})"
            )

        result = f"Found {len(entries)} memories:\n\n" + "\n\n".join(lines)
        return result, {"count": len(entries)}


# ======================================================================
# memory_save
# ======================================================================


class MemorySaveToolParams(BaseToolParams):
    """Save important information to the user's long-term memory.

    Use this to remember things the user explicitly asks to remember,
    or key information you believe is worth retaining across sessions
    (preferences, facts, decisions, entity names, etc.).
    """

    name: ClassVar[str] = "memory_save"

    content: str = Field(description="The information to save as a memory")
    category: str = Field(
        default="other",
        description="Category: preference, fact, decision, entity, or other",
    )


class MemorySaveTool(BaseTool):
    """Memory save tool."""

    name: ClassVar[str] = "memory_save"
    params_class: ClassVar[type[BaseToolParams]] = MemorySaveToolParams

    def __init__(self, memory_manager: MemoryManager, user_id: str):
        super().__init__()
        self._manager = memory_manager
        self._user_id = user_id

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter error: {e}", {"error": str(e)}

        assert isinstance(params, MemorySaveToolParams)
        memory_id = self._manager.save(
            self._user_id, params.content, category=params.category
        )

        if memory_id:
            return f"Memory saved successfully (id: {memory_id}).", {"memory_id": memory_id}
        else:
            return "Memory already exists (duplicate detected).", {"duplicate": True}


# ======================================================================
# memory_forget
# ======================================================================


class MemoryForgetToolParams(BaseToolParams):
    """Delete specific information from the user's long-term memory.

    Use this when the user asks to forget something, or when information
    is no longer accurate or relevant. Provide either a query to search
    and delete matching memories, or a specific memory_id.
    """

    name: ClassVar[str] = "memory_forget"

    query: str | None = Field(
        default=None,
        description="Search query to find and delete matching memories",
    )
    memory_id: str | None = Field(
        default=None,
        description="Specific memory ID to delete (from memory_search results)",
    )


class MemoryForgetTool(BaseTool):
    """Memory delete tool."""

    name: ClassVar[str] = "memory_forget"
    params_class: ClassVar[type[BaseToolParams]] = MemoryForgetToolParams

    def __init__(self, memory_manager: MemoryManager, user_id: str):
        super().__init__()
        self._manager = memory_manager
        self._user_id = user_id

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter error: {e}", {"error": str(e)}

        assert isinstance(params, MemoryForgetToolParams)
        result = self._manager.forget(
            self._user_id, query=params.query, memory_id=params.memory_id
        )
        return result, {}
