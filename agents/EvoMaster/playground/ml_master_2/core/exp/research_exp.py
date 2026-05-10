import logging
import re
from typing import Any
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance
import uuid
import os
import json
from evomaster.agent import BaseAgent


def _parse_json_from_response(text: str) -> dict:
    """Parse JSON from model response, compatible with pure JSON and ```json ... ``` code block format.

    Args:
        text: The raw text response from the model.

    Returns:
        Parsed JSON as a dictionary.
    """
    text = text.strip()
    # Try to extract ```json ... ``` or ``` ... ``` code block
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()
    return json.loads(text)

class ResearchExp(BaseExp):
    """Experiment for generating research plans with improvement directions.

    Uses the research agent to analyze the current best solution and propose
    structured improvement directions with specific ideas.
    """

    def __init__(self, research_agent, config, initial_code, exp_name):
        super().__init__(research_agent, config)
        self.research_agent = research_agent
        self.uid = uuid.uuid4()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workspace_path = self.research_agent.session.config.workspace_path
        self.initial_code = initial_code
        self._exp_name = exp_name
    @property
    def exp_name(self) -> str:
        """Return the experiment stage name."""
        return self._exp_name


    def run(self, task_description: str, data_preview: str, best_solution: str, research_plan_and_result: list, task_id: str = "exp_001") -> dict:
        """Execute the research experiment to generate improvement directions.

        Analyzes the current best solution and past improvement history to propose
        structured research plans with major directions and specific ideas.

        Args:
            task_description: Natural language description of the ML task.
            data_preview: Textual preview of the dataset.
            best_solution: Current best solution code.
            research_plan_and_result: List alternating between research plans and their results.
            task_id: Unique task identifier.

        Returns:
            A dictionary mapping directions to ideas: {direction: {idea_key: idea_description}}.
        """
        self.logger.info("Starting draft task execution")
        self.logger.info(f"Task: {task_description}")

        # Concatenate research_plan_and_result as text: odd positions are plans, even positions are corresponding results
        if not research_plan_and_result:
            research_plan_and_result_text = "You have not made any improvement attempts and results yet."
            best_solution = "Best solution is the same as the initial draft code."
        else:
            research_plan_and_result_text = ""
            for i in range(0, len(research_plan_and_result), 2):
                plan = research_plan_and_result[i] if i < len(research_plan_and_result) else ""
                result = research_plan_and_result[i + 1] if i + 1 < len(research_plan_and_result) else ""
                block = (
                    "Based on the above code, you tried the following research plan:\n"
                    f"{plan}\n"
                    "Conclusion:\n"
                    f"{result}"
                )
                research_plan_and_result_text += block
                if i + 2 < len(research_plan_and_result):
                    research_plan_and_result_text += "\n\n"

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
                    "initial_code": self.initial_code,
                    'best_code': best_solution,
                    'research_plan_and_result_text': research_plan_and_result_text,
                })

                research_task = TaskInstance(
                    task_id=f"{task_id}_research",
                    task_type="research",
                    description=task_description,
                    input_data={},
                )

                research_trajectory = self.research_agent.run(research_task)
                research_result = self._extract_agent_response(research_trajectory)
                self.logger.info(f"Research result: {research_result[:2000]}...")

                research_plan = _parse_json_from_response(research_result)
                
                self.logger.info("Research completed")
                self.logger.info(f"Research plan: {research_plan}")
                self.research_agent._prompt_format_kwargs = research_original_format_kwargs

            return research_plan

        except Exception as e:
            self.logger.error(f"Research task execution failed: {e}", exc_info=True)
            raise ValueError(f"Research task execution failed: {e}")




