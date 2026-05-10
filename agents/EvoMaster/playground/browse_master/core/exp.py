"""
pe_exp Implementation

Implements the workflow of Planner and Executor
"""

import logging
import re
from typing import Any, List
from evomaster.core.exp import BaseExp
from evomaster.agent import BaseAgent
from evomaster.utils.types import TaskInstance


def extract_planner_answer(text: str) -> str:
    """Extract the final answer from Planner's response

    First attempt to extract content within <answer> tags,
    then extract content after ,
    finally return the original text

    Args:
        text: Planner's response text

    Returns:
        Extracted answer text
    """
    pattern = r'<answer>\s*((?:(?!</answer>).)*?)</answer>'
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    else:
        pattern = r'\s*(.*?)$'
        matches = list(re.finditer(pattern, text, re.DOTALL))
        if matches:
            return matches[-1].group(1).strip()
        else:
            return text.strip()


def extract_tasks(text: str) -> List[str]:
    """Extract all tasks from Planner's response

    Args:
        text: Planner's response text

    Returns:
        List of tasks
    """
    task_pattern = r'<task>\s*(.*?)\s*</task>'
    matches = re.findall(task_pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


class PlanExecuteExp(BaseExp):
    """Multi-agent experiment class

    Implements the collaborative workflow of Planner and Executor:
    1. Planner analyzes the task and formulates a plan
    2. Executor calls tools to execute code tasks according to the plan
    """

    def __init__(self, planner, executor, config):
        """Initialize multi-agent experiment

        Args:
            Planner: Instance of planner Agent
            Executor: Instance of executor Agent
            config: Instance of EvoMasterConfig
        """
        # For compatibility with the base class, pass the first agent (Planner)
        # But multiple agents will be used in actual usage
        super().__init__(planner, config)
        self.planner = planner
        self.executor = executor
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def exp_name(self) -> str:
        """Return the experiment phase name"""
        return "PlanExecute"

    def run(self, task_description: str, task_id: str = "exp_001") -> dict:
        """Run the multi-agent experiment

        Workflow:
        1. Planner analyzes the task and formulates a plan
        2. Executor calls tools to execute code tasks according to the plan
        3. Planner iterates based on Executor's feedback until the answer is found

        Args:
            task_description: Task description
            task_id: Task ID

        Returns:
            Execution result dictionary
        """
        self.logger.info("Starting plan-execute task execution")
        self.logger.info(f"Task: {task_description}")

        results = {
            'task_description': task_description,
            'status': 'running',
        }

        BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=0)
        self.final_found = False

        try:
            # Step 1: Planning Agent formulates the plan
            if self.planner:
                self.logger.info("=" * 60)
                self.logger.info("Step 1: Planning Agent analyzing task...")
                self.logger.info("=" * 60)

                planner_task = TaskInstance(
                    task_id=f"{task_id}_planner",
                    task_type="planner",
                    description=task_description,
                    input_data={},
                )

                planner_trajectory = self.planner.run(planner_task)
                results['planner_trajectory'] = planner_trajectory

                # Extract Planning Agent's response
                planner_result = self._extract_agent_response(planner_trajectory)
                results['planner_result'] = planner_result

                self.logger.info("Planning completed")
                self.logger.info(f"Planning result: {planner_result[:200]}...")

                if "<answer>" in planner_result:
                    # Extract final answer and end the loop
                    final_answer = extract_planner_answer(planner_result)
                    results['final_answer'] = final_answer
                    results['final_found'] = 1
                    self.logger.info("=" * 60)
                    self.logger.info(f"Final answer found: {final_answer}")
                    self.logger.info("=" * 60)
                    self.final_found = True

                elif "<task>" in planner_result:
                    # Extract subtasks
                    tasks = extract_tasks(planner_result)
                    results['final_found'] = 0
                    search_target = tasks[-1] if tasks else ""
                    results['search_target'] = search_target
                    self.logger.info(f"Task assigned to Executor: {search_target}")

                else:
                    # Exception handling: neither task nor answer found
                    self.logger.warning("Neither <task> nor <answer> found in planner output")
                    results['final_found'] = 0
                    results['status'] = 'failed'
                    results['error'] = "Neither <task> nor <answer> found in planner output"
                    

            # Step 2: Coding Agent executes the task
            if self.final_found == False :
                if self.executor:
                    self.logger.info("=" * 60)
                    self.logger.info("Step 2: Coding Agent executing task...")
                    self.logger.info("=" * 60)

                    # Prepare user prompt formatting parameters for Coding Agent
                    # Use prompt_format_kwargs to pass search_target
                    original_format_kwargs = self.executor._prompt_format_kwargs.copy()
                    self.executor._prompt_format_kwargs.update({
                        'search_target': results.get('search_target')
                    })

                    # Create task instance
                    executor_task = TaskInstance(
                        task_id=f"{task_id}_executor",
                        task_type="executor",
                        description=task_description,
                        input_data={},
                    )

                    executor_trajectory = self.executor.run(executor_task)

                    # Restore original formatting parameters
                    self.executor._prompt_format_kwargs = original_format_kwargs

                    # Extract Coding Agent's result
                    executor_result = self._extract_agent_response(executor_trajectory)
                    results['executor_result'] = executor_result
                    results['executor_trajectory'] = executor_trajectory

                    self.logger.info("Coding completed")
                    self.logger.info(f"Coding status: {executor_trajectory.status}")

                results['status'] = 'completed'
                self.logger.info("Multi-agent task execution completed")

                # Save results to self.results (for save_results)
                result = {
                    "task_id": task_id,
                    "status": results['status'],
                    "steps": 0,  # Steps calculation is different in multi-agent scenarios
                    "final_found": 0,
                    "planner_trajectory": results.get('planner_trajectory'),
                    "executor_trajectory": results.get('executor_trajectory'),
                    "search_target": results.get('search_target'),
                    "planner_result": results.get('planner_result'),
                    "executor_result": results.get('executor_result'),
                }
                self.results.append(result)
            else :
                results['status'] = 'completed'
                self.logger.info("Multi-agent task execution completed")

                # Save results to self.results (for save_results)
                result = {
                    "task_id": task_id,
                    "status": results['status'],
                    "steps": 0,  # Steps calculation is different in multi-agent scenarios
                    "final_found": 1,
                    "final_answer": results.get('final_answer'),
                    "planner_trajectory": results.get('planner_trajectory'),
                    # "executor_trajectory": results.get('executor_trajectory'),
                    "planner_result": results.get('planner_result'),
                    # "executor_result": results.get('executor_result'),
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
