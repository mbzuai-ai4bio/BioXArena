from __future__ import annotations

from evomaster.agent import BaseAgent

from . import NodeExp
from ..utils.engine import extract_python_code


class DebugExp(NodeExp):
    """Debug stage for repairing failed nodes."""

    def run(self, task_description: str, prev_code: str, term_out: str, issue: str) -> dict:
        """Run this experiment stage.

        Args:
            task_description: Natural language task description.
            prev_code: Previous Python code string.
            term_out: Terminal output text.
            issue: Issue description used for debugging.

        Returns:
            dict: Result of this function.
        """
        node_id = self.node.id
        BaseAgent.set_exp_info(exp_name=f"debug_{node_id[:8]}", exp_index=self.exp_index)

        raw_response = self._run_agent_task(
            agent=self.agent,
            task_id=f"{node_id}_debug",
            task_type="debug",
            description=task_description,
            prompt_kwargs={
                "task_description": task_description,
                "previous_code": prev_code,
                "terminal_output": term_out,
                "issue": issue,
                "data_preview": self.data_preview,
                "SUBMISSION_FILE": str(self.workspace / "submission" / "submission.csv"),
                "SERVER_URL": "http://localhost:5003/validate",
            },
        )
        code = extract_python_code(raw_response)

        eval_result = self._execute_and_evaluate(code)
        return {
            "code": code,
            "raw_response": raw_response,
            "exec": eval_result["exec"],
            "metric": eval_result["metric"],
            "metric_detail": eval_result["metric_detail"],
        }
