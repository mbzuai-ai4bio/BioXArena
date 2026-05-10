"""AskUser tool -- only injected in interactive contexts

Allows the Agent to ask the user structured questions, pausing execution to wait for the user's answer.
This tool is not built-in (not universal to all agents); it is injected by the dispatcher on demand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession


class AskUserToolParams(BaseToolParams):
    """Ask the user 1-2 clarification questions when critical information is missing.

    Use this tool when:
    - The user's request is ambiguous and you need to choose between fundamentally different approaches
    - Critical information is missing that would significantly change your design

    Do NOT use this tool when:
    - The request is clear enough to proceed with reasonable defaults
    - The missing details are minor and can be decided for the user

    Guidelines:
    - Ask at most 1-2 questions at a time
    - Each question should have 2-4 concise options
    - Only ask about decisions that fundamentally affect the architecture or approach
    - For minor details, make reasonable defaults and mention them in your plan
    """

    name: ClassVar[str] = "ask_user"

    questions: list[dict[str, Any]] = Field(
        description=(
            "List of questions to ask. Each question is an object with: "
            "'question' (the question text), "
            "'header' (a short label displayed as group title, max 12 chars, e.g. 'Auth method'), "
            "'options' (list of objects with 'label' (1-5 words) and optional 'description')."
        ),
    )


class AskUserTool(BaseTool):
    """User question tool -- pauses agent execution upon invocation, continues after the user answers."""

    name: ClassVar[str] = "ask_user"
    params_class: ClassVar[type[BaseToolParams]] = AskUserToolParams

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """Parse question parameters (actual interception happens in agent.py; this method serves as a fallback)."""
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {str(e)}", {"error": str(e)}

        assert isinstance(params, AskUserToolParams)

        self.logger.info(
            "ask_user called with %d question(s)", len(params.questions)
        )

        return "Questions sent to user. Waiting for response.", {
            "ask_user": True,
            "questions": params.questions,
        }