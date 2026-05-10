"""Chat Agent Delegation Tool

Allows chat_agent to delegate tasks to specialized Agents (e.g., agent_builder).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession

logger = logging.getLogger(__name__)

# List of delegatable Agents (just add a line to extend)
DELEGATABLE_AGENTS = {
    "agent_builder": "创建/设计/构建新的 AI Agent",
}


class DelegateToAgentParams(BaseToolParams):
    """Delegate a task to a specialized Agent.

    Available Agents:
    - agent_builder: Create/design/build new AI Agents. Use when user wants to create an agent.
      Examples: "Help me create an xxx agent", "I want to build a code review agent"

    Only delegate when the user explicitly needs specialized Agent capabilities.
    Handle normal conversation, search, and Q&A by yourself.
    """

    name: ClassVar[str] = "delegate_to_agent"

    agent_name: str = Field(
        description="委派目标 Agent 名称，当前可用: 'agent_builder'（创建新 Agent）"
    )
    task: str = Field(
        description="任务描述，使用用户原始语言，包含完整上下文"
    )


class DelegateToAgentTool(BaseTool):
    """Delegation tool: forwards tasks to specialized Agents."""

    name: ClassVar[str] = "delegate_to_agent"
    params_class: ClassVar[type[BaseToolParams]] = DelegateToAgentParams

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """Execute delegation: validate agent name and store delegation info.

        The returned info dict contains a delegated=True marker.
        The dispatcher detects delegation by scanning the trajectory's ToolMessage.meta["info"].
        """
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"参数错误: {e}", {"error": str(e)}

        assert isinstance(params, DelegateToAgentParams)

        agent_name = params.agent_name
        task = params.task

        if agent_name not in DELEGATABLE_AGENTS:
            available = ", ".join(DELEGATABLE_AGENTS.keys())
            return (
                f"未知 Agent: '{agent_name}'。可用: {available}",
                {"error": "unknown_agent", "agent_name": agent_name},
            )

        self.logger.info("Delegation requested: agent=%s, task=%s", agent_name, task[:100])

        return (
            f"委派已接受。任务将由 '{agent_name}' 处理。请告知用户请求正在处理。",
            {"delegated": True, "agent_name": agent_name, "task": task},
        )
