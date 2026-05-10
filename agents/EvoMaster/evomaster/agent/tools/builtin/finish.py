"""EvoMaster Finish tool.

Used to mark task completion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field

from ..base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession


class FinishToolParams(BaseToolParams):
    """Signals the completion of the current task or conversation.

    Use this tool when:
    - You have successfully completed the user's requested task
    - You cannot proceed further due to technical limitations or missing information

    The message should include:
    - A clear summary of actions taken and their results
    - Any next steps for the user
    - Explanation if you're unable to complete the task
    - Any follow-up questions if more information is needed

    The task_completed field should be set to True if you believed you have completed the task, and False otherwise.
    """
    
    name: ClassVar[str] = "finish"

    message: str = Field(description="Final message to send to the user")
    task_completed: Literal["true", "false", "partial"] = Field(
        description="Whether you have completed the task."
    )


class FinishTool(BaseTool):
    """Finish tool."""
    
    name: ClassVar[str] = "finish"
    params_class: ClassVar[type[BaseToolParams]] = FinishToolParams

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """Mark the task as complete."""
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {str(e)}", {"error": str(e)}
        
        assert isinstance(params, FinishToolParams)
        
        # Log completion information
        self.logger.info(f"Task finished. Completed: {params.task_completed}")
        self.logger.info(f"Final message: {params.message[:200]}...")
        
        return f"Task marked as {params.task_completed}. Message: {params.message}", {
            "task_completed": params.task_completed,
            "message": params.message,
        }

