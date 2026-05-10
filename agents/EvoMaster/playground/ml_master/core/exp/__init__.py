from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance

from ..utils.engine import extract_agent_response, parse_metric_content, run_code_via_bash

logger = logging.getLogger(__name__)


class NodeExp(BaseExp):
    """Base experiment for a single UCT node."""

    def __init__(
        self,
        agent: Any,
        metric_agent: Any,
        session: Any,
        workspace: Path,
        exp_id: str | None,
        data_preview: str,
        node: Any,
        exp_index: int = 0,
    ) -> None:
        """Initialize NodeExp.

        Args:
            agent: Agent instance or agent mapping.
            metric_agent: Agent instance or agent mapping.
            session: Execution session object.
            workspace: Workspace path.
            exp_id: Identifier string.
            data_preview: Value for data preview.
            node: UCT node object.
            exp_index: Numeric control parameter.

        Returns:
            None.
        """
        super().__init__(agent=agent, config=None)
        self.metric_agent = metric_agent
        self.session = session
        self.workspace = workspace
        self.exp_id = exp_id
        self.data_preview = data_preview
        self.node = node
        self.exp_index = exp_index

    @contextmanager
    def _temporary_prompt_kwargs(self, agent: Any, overrides: dict[str, Any]) -> Iterator[None]:
        """Execute temporary prompt kwargs.

        Args:
            agent: Agent instance or agent mapping.
            overrides: Value for overrides.

        Returns:
            Generated values from the iterator.
        """
        original = dict(getattr(agent, "_prompt_format_kwargs", {}) or {})
        agent._prompt_format_kwargs.update(overrides)
        try:
            yield
        finally:
            agent._prompt_format_kwargs = original

    def _run_agent_task(
        self,
        *,
        agent: Any,
        task_id: str,
        task_type: str,
        description: str,
        prompt_kwargs: dict[str, Any],
    ) -> str:
        """Execute run agent task.

        Args:
            agent: Agent instance or agent mapping.
            task_id: Task identifier string.
            task_type: Value for task type.
            description: Value for description.
            prompt_kwargs: Value for prompt kwargs.

        Returns:
            str: Result of this function.
        """
        with self._temporary_prompt_kwargs(agent, prompt_kwargs):
            task = TaskInstance(
                task_id=task_id,
                task_type=task_type,
                description=description,
                input_data={},
            )
            trajectory = agent.run(task)
        return extract_agent_response(trajectory)

    def _run_metric_agent(self, code: str, stdout: str) -> dict[str, Any]:
        """Execute run metric agent.

        Args:
            code: Generated Python code string.
            stdout: Standard output text.

        Returns:
            dict[str, Any]: Result of this function.
        """
        if not self.metric_agent:
            return {"metric": None, "is_bug": True, "has_submission": False, "error": "metric agent missing"}

        text = self._run_agent_task(
            agent=self.metric_agent,
            task_id=f"{self.node.id}_metric",
            task_type="metric",
            description="parse metric",
            prompt_kwargs={"code": code, "stdout": stdout},
        )
        return parse_metric_content(text)

    def _execute_and_evaluate(self, code: str) -> dict[str, Any]:
        """Execute execute and evaluate.

        Args:
            code: Generated Python code string.

        Returns:
            dict[str, Any]: Result of this function.
        """
        exec_result = run_code_via_bash(self.agent, self.workspace, code, self.node.id)
        metric_detail = self._run_metric_agent(code, exec_result.get("stdout", ""))
        return {
            "exec": exec_result,
            "metric_detail": metric_detail,
            "metric": metric_detail.get("metric"),
        }


__all__ = ["NodeExp"]
