"""Research experiment implementation for Kaggle competitions.

Manages the research agent workflow for generating improvement plans.
"""

import logging
from typing import Any
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance
import uuid
import os
import json
from evomaster.agent import BaseAgent

class ResearchExp(BaseExp):
    """Research experiment class for Kaggle competitions.

    Uses a research agent to analyze data, previous solutions, and existing knowledge
    to generate structured improvement plans.
    """

    def __init__(self, research_agent, config,exp_index):
        """Initialize the research experiment.

        Args:
            research_agent: Agent responsible for researching improvement strategies
            config: EvoMasterConfig instance
            exp_index: Experiment index for identification
        """
        super().__init__(research_agent, config)
        self.research_agent = research_agent
        self.uid = uuid.uuid4()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workspace_path = self.research_agent.session.config.workspace_path
        self.exp_index = exp_index
    @property
    def exp_name(self) -> str:
        """Return the experiment phase name."""
        return f"Research_{self.exp_index}"


    def run(self, task_description: str, data_preview: str, best_solution: str, knowledge: str, task_id: str = "exp_001") -> dict:
        """Run the research experiment workflow.

        Args:
            task_description: Description of the task
            data_preview: Preview of the dataset
            best_solution: The best solution so far
            knowledge: Accumulated knowledge from previous experiments
            task_id: Task ID

        Returns:
            Research plan dictionary with improvement strategies
        """
        self.logger.info("Starting draft task execution")
        self.logger.info(f"Task: {task_description}")

        try:
            if self.research_agent:
                self.logger.info("=" * 60)
                self.logger.info("Step 1: Research Agent analyzing task...")
                self.logger.info("=" * 60)
                BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=1)
                research_original_format_kwargs = self.research_agent._prompt_format_kwargs.copy()
                self.research_agent._prompt_format_kwargs.update({
                    'task_description': task_description,
                    'data_preview': data_preview,
                    'best_code': best_solution,
                    'memory': knowledge,
                })

                research_task = TaskInstance(
                    task_id=f"{task_id}_research",
                    task_type="research",
                    description=task_description,
                    input_data={},
                )

                research_trajectory = self.research_agent.run(research_task)
                research_result = self._extract_agent_response(research_trajectory)
                research_plan = json.loads(research_result.strip())
                
                self.logger.info("Research completed")
                self.logger.info(f"Research result: {research_result[:2000]}...")
                self.logger.info(f"Research plan: {research_plan}")
                self.research_agent._prompt_format_kwargs = research_original_format_kwargs

            return research_plan

        except Exception as e:
            self.logger.error(f"Research task execution failed: {e}", exc_info=True)
            raise ValueError(f"Research task execution failed: {e}")




