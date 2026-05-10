"""Multi-Agent Experiment Implementation

Defines the experiment execution logic for multi-agent collaboration.
"""

import logging
from typing import Any
from evomaster.core.exp import BaseExp
from evomaster.agent import BaseAgent
from evomaster.utils.types import TaskInstance


class MultiAgentExp(BaseExp):
    """Multi-Agent Experiment Class

    Implements the collaborative workflow of Planning Agent and Coding Agent:
    1. Planning Agent analyzes the task and formulates a plan
    2. Coding Agent executes code tasks based on the plan
    """

    def __init__(self, planning_agent, coding_agent, config):
        """Initialize multi-agent experiment.

        Args:
            planning_agent: Planning Agent instance
            coding_agent: Coding Agent instance
            config: EvoMasterConfig instance
        """
        # For base class compatibility, pass the first agent (planning_agent)
        # But multiple agents will be used in actual execution
        super().__init__(planning_agent, config)
        self.planning_agent = planning_agent
        self.coding_agent = coding_agent
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def exp_name(self) -> str:
        """Return the experiment phase name."""
        return "MultiAgent"

    def run(self, task_description: str, task_id: str = "exp_001") -> dict:
        """Run the multi-agent experiment.

        Workflow:
        1. Planning Agent analyzes the task and formulates a plan
        2. Coding Agent executes code tasks based on the plan

        Args:
            task_description: Task description
            task_id: Task ID

        Returns:
            Execution result dictionary
        """
        self.logger.info("Starting multi-agent task execution")
        self.logger.info(f"Task: {task_description}")

        results = {
            'task_description': task_description,
            'planning_result': None,
            'coding_result': None,
            'status': 'running',
        }

        # Set current exp info for trajectory recording
        # exp_name is automatically inferred from the class name (MultiAgentExp -> MultiAgent)
        BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=0)

        try:
            # Step 1: Planning Agent formulates the plan
            if self.planning_agent:
                self.logger.info("=" * 60)
                self.logger.info("Step 1: Planning Agent analyzing task...")
                self.logger.info("=" * 60)

                planning_task = TaskInstance(
                    task_id=f"{task_id}_planning",
                    task_type="planning",
                    description=task_description,
                    input_data={},
                )

                planning_trajectory = self.planning_agent.run(planning_task)
                results['planning_trajectory'] = planning_trajectory

                # Extract Planning Agent's response
                planning_result = self._extract_agent_response(planning_trajectory)
                results['planning_result'] = planning_result

                self.logger.info("Planning completed")
                self.logger.info(f"Planning result: {planning_result[:200]}...")

            # Step 2: Coding Agent executes the task
            if self.coding_agent:
                self.logger.info("=" * 60)
                self.logger.info("Step 2: Coding Agent executing task...")
                self.logger.info("=" * 60)

                # Prepare user prompt formatting parameters for Coding Agent
                # Use prompt_format_kwargs to pass planning_result
                original_format_kwargs = self.coding_agent._prompt_format_kwargs.copy()
                self.coding_agent._prompt_format_kwargs.update({
                    'planning_result': results.get('planning_result', '')
                })

                # Create task instance
                coding_task = TaskInstance(
                    task_id=f"{task_id}_coding",
                    task_type="coding",
                    description=task_description,
                    input_data={},
                )

                coding_trajectory = self.coding_agent.run(coding_task)

                # Restore original formatting parameters
                self.coding_agent._prompt_format_kwargs = original_format_kwargs

                # Extract Coding Agent's result
                coding_result = self._extract_agent_response(coding_trajectory)
                results['coding_result'] = coding_result
                results['coding_trajectory'] = coding_trajectory

                self.logger.info("Coding completed")
                self.logger.info(f"Coding status: {coding_trajectory.status}")

            results['status'] = 'completed'
            self.logger.info("Multi-agent task execution completed")

            # Save results to self.results (for save_results)
            result = {
                "task_id": task_id,
                "status": results['status'],
                "steps": 0,  # Step counting differs in multi-agent scenarios
                "planning_trajectory": results.get('planning_trajectory'),
                "coding_trajectory": results.get('coding_trajectory'),
                "planning_result": results.get('planning_result'),
                "coding_result": results.get('coding_result'),
            }
            self.results.append(result)

        except Exception as e:
            self.logger.error(f"Multi-agent task execution failed: {e}", exc_info=True)
            results['status'] = 'failed'
            results['error'] = str(e)

            # Save failed results
            result = {
                "task_id": task_id,
                "status": "failed",
                "steps": 0,
                "error": str(e),
            }
            self.results.append(result)

        return results

