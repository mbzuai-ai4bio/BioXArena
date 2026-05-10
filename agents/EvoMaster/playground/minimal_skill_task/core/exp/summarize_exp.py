"""SummarizeExp: contains only the summarize agent, which selects from multi-round retrieval results and outputs a PDF."""

import logging
from typing import Any
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance

from ..utils.rag_utils import extract_agent_response, update_agent_format_kwargs


class SummarizeExp(BaseExp):
    def __init__(self, summarize_agent, config):
        super().__init__(summarize_agent, config)
        self.summarize_agent = summarize_agent
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(
        self,
        task_description: str,
        search_results: str,
        db: dict,
        task_id: str = "exp_001",
    ) -> tuple[str, Any]:
        """Run the Summarize Agent and return (summarize_result, trajectory)."""
        self.logger.info("Starting SummarizeExp")
        update_agent_format_kwargs(
            self.summarize_agent,
            task_description=task_description,
            search_results=search_results,
            nodes_data=db["nodes_data"],
        )
        task = TaskInstance(
            task_id=f"{task_id}_summarize",
            task_type="summarize",
            description=task_description,
            input_data={},
        )
        trajectory = self.summarize_agent.run(task)
        output = extract_agent_response(trajectory)
        self.logger.info("SummarizeExp completed")
        return output, trajectory
