from __future__ import annotations

from evomaster.agent import BaseAgent

from . import NodeExp
from ..utils.engine import extract_python_code, extract_text_up_to_code


class ImproveExp(NodeExp):
    """Improve stage for incremental optimization from best-known candidate."""

    def run(
        self,
        task_description: str,
        best_code: str,
        best_metric: float | None,
        memory: str,
        term_out: str,
    ) -> dict:
        """Run this experiment stage.

        Args:
            task_description: Natural language task description.
            best_code: Current best Python code string.
            best_metric: Metric value or metric-related input.
            memory: Context memory text.
            term_out: Terminal output text.

        Returns:
            dict: Result of this function.
        """
        node_id = self.node.id
        BaseAgent.set_exp_info(exp_name=f"improve_{node_id[:8]}", exp_index=self.exp_index)

        raw_response = self._run_agent_task(
            agent=self.agent,
            task_id=f"{node_id}_improve",
            task_type="improve",
            description=task_description,
            prompt_kwargs={
                "task_description": task_description,
                "best_code": best_code,
                "best_metric": best_metric,
                "memory": memory,
                "execution_output": term_out,
                "data_preview": self.data_preview,
                "SUBMISSION_FILE": str(self.workspace / "submission" / "submission.csv"),
                "SERVER_URL": "http://localhost:5003/validate",
            },
        )
        plan = extract_text_up_to_code(raw_response)
        code = extract_python_code(raw_response)

        eval_result = self._execute_and_evaluate(code)
        return {
            "plan": plan,
            "code": code,
            "raw_response": raw_response,
            "exec": eval_result["exec"],
            "metric": eval_result["metric"],
            "metric_detail": eval_result["metric_detail"],
        }
