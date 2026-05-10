"""AnalyzeExp: contains only the analyze agent, which inspects the database structure and outputs query writing guidelines."""

import logging
from pathlib import Path
from typing import Any
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance

from ..utils.rag_utils import extract_agent_response, update_agent_format_kwargs


def _project_root() -> Path:
    """EvoMaster project root directory (contains evomaster/, playground/, and configs/; go five levels up from exp)."""
    return Path(__file__).resolve().parent.parent.parent.parent.parent


class AnalyzeExp(BaseExp):
    def __init__(self, analyze_agent, config):
        super().__init__(analyze_agent, config)
        self.analyze_agent = analyze_agent
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(
        self,
        task_description: str,
        db: dict,
        task_id: str = "exp_001",
    ) -> tuple[str, Any]:
        """Run the Analyze Agent and return (analyze_output, trajectory)."""
        self.logger.info("Starting AnalyzeExp")
        root = _project_root()
        vec_dir = db["vec_dir"]
        # db has already been converted to absolute paths in the workflow; if still relative, resolve it against the project root.
        vec_path = Path(vec_dir) if Path(vec_dir).is_absolute() else root / vec_dir
        nodes_jsonl_path = vec_path / "nodes.jsonl"

        # Same as minimal_kaggle: inject task_description and other values at runtime; the template uses {task_description}.
        update_agent_format_kwargs(
            self.analyze_agent,
            task_description=task_description,
            vec_dir=vec_dir,
            nodes_data=db["nodes_data"],
            model=db["model"],
            nodes_jsonl_path=str(nodes_jsonl_path),
        )
        task = TaskInstance(
            task_id=f"{task_id}_analyze",
            task_type="analyze",
            description=task_description,
            input_data={},
        )
        trajectory = self.analyze_agent.run(task)
        output = extract_agent_response(trajectory)
        self.logger.info("AnalyzeExp completed")
        return output, trajectory
