"""Improve experiment implementation for Kaggle competitions.

Manages the improve-debug-metric workflow for enhancing existing code solutions.
"""

import logging
from typing import Any
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
from ..utils.code import read_code,save_code_to_file
import uuid
import os
from evomaster.agent import BaseAgent

class ImproveExp(BaseExp):
    """Improve experiment class for Kaggle competitions.

    Orchestrates improve, debug, and metric agents to enhance, validate,
    and fix code solutions based on previous best solutions.
    """

    def __init__(self, improve_agent, debug_agent, metric_agent, config,exp_index):
        """Initialize the improve experiment.

        Args:
            improve_agent: Agent responsible for improving code solutions
            debug_agent: Agent responsible for debugging failed code
            metric_agent: Agent responsible for extracting validation scores
            config: EvoMasterConfig instance
            exp_index: Experiment index for identification
        """
        super().__init__(improve_agent, config)
        self.improve_agent = improve_agent
        self.debug_agent = debug_agent
        self.metric_agent = metric_agent
        self.uid = uuid.uuid4()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workspace_path = self.improve_agent.session.config.workspace_path
        self.terminal_output = ""
        self.code = ""
        self.debug_times = 0
        self.exp_index = exp_index

    @property
    def exp_name(self) -> str:
        """Return the experiment phase name."""
        return f"Improve_{self.exp_index}"

    def run(self, task_description: str, data_preview: str, best_solution: str, idea: str, task_id: str = "exp_001") -> dict:
        """Run the improve experiment workflow.

        Args:
            task_description: Description of the task
            data_preview: Preview of the dataset
            best_solution: The best solution so far
            idea: Improvement idea to apply
            task_id: Task ID

        Returns:
            Tuple of (is_success, validation_score, uid, code)
        """
        self.logger.info("Starting draft task execution")
        self.logger.info(f"Task: {task_description}")

        try:
            if self.improve_agent:
                self.logger.info("=" * 60)
                self.logger.info("Step 1: Improve Agent analyzing task...")
                self.logger.info("=" * 60)
                BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=1)
                improve_original_format_kwargs = self.improve_agent._prompt_format_kwargs.copy()
                self.improve_agent._prompt_format_kwargs.update({
                    'task_description': task_description,
                    'data_preview': data_preview,
                    'previous_solution': best_solution,
                    'improve_idea': idea,
                })

                improve_task = TaskInstance(
                    task_id=f"{task_id}_improve",
                    task_type="improve",
                    description=task_description,
                    input_data={},
                )

                improve_trajectory = self.improve_agent.run(improve_task)
                
                improve_result = self._extract_agent_response(improve_trajectory)
                improve_code,self.code = read_code(improve_result, self.uid)
                save_code_to_file(self.workspace_path, "run.py", improve_code)
                tool_call_obj = ChatCompletionMessageToolCall(
                    id="call_123",
                    type="function",
                    function=Function(
                        name="execute_bash",
                        arguments='{"command": "python run.py","timeout": "3600"}'
                    )
                )
                observation, info =self.improve_agent._execute_tool(tool_call_obj)
                self.terminal_output = observation
                if info.get("exit_code") == 0 and os.path.exists(os.path.join(self.workspace_path, "submission", f"submission_{self.uid}.csv")):
                    is_success = True
                else:
                    is_success = False
                self.logger.info(f"Improve Agent execute_bash result: {observation}")
                self.logger.info(f"Improve Agent execute_bash info: {info}")

                
                self.logger.info("Improve completed")
                self.logger.info(f"Improve result: {improve_result[:2000]}...")
                self.improve_agent._prompt_format_kwargs = improve_original_format_kwargs


            if self.metric_agent and is_success:
                self.logger.info("=" * 60)
                self.logger.info("Step 2: Metric Agent executing task...")
                self.logger.info("=" * 60)
                metric_original_format_kwargs = self.metric_agent._prompt_format_kwargs.copy()
                self.metric_agent._prompt_format_kwargs.update({
                    'terminal_output': observation
                })
                metric_task = TaskInstance(
                    task_id=f"{task_id}_metric",
                    task_type="metric",
                    input_data={},
                )

                metric_trajectory = self.metric_agent.run(metric_task)

                # Extract Metric Agent's response
                metric_result = self._extract_agent_response(metric_trajectory)
                try:
                    validation_score = float(metric_result.split("\\boxed{")[1].split("}")[0])
                except:
                    is_success = False
                    validation_score = None
                self.logger.info(f"validation score: {validation_score}")
                self.logger.info("Metric completed")
                self.logger.info(f"Metric result: {metric_result[:2000]}...")
                self.metric_agent._prompt_format_kwargs = metric_original_format_kwargs
            
            debug_times = 0
            while is_success==False and debug_times < 3:
                self.logger.info("=" * 60)
                self.logger.info("Step 3: Debug Agent executing task...")
                self.logger.info("=" * 60)
                debug_original_format_kwargs = self.debug_agent._prompt_format_kwargs.copy()
                self.debug_agent._prompt_format_kwargs.update({
                    'task_description': task_description,
                    'terminal_output': self.terminal_output,
                    'buggy_code': self.code,
                    'data_preview': data_preview,
                })
                debug_task = TaskInstance(
                    task_id=f"{task_id}_debug",
                    task_type="debug",
                    task_description=task_description,
                    input_data={},
                )
                debug_trajectory = self.debug_agent.run(debug_task)
                debug_result = self._extract_agent_response(debug_trajectory)
                debug_code,self.code = read_code(debug_result, self.uid)
                save_code_to_file(self.workspace_path, "run.py", debug_code)
                tool_call_obj = ChatCompletionMessageToolCall(
                    id="call_123",
                    type="function",
                    function=Function(
                        name="execute_bash",
                        arguments='{"command": "python run.py","timeout": "3600"}'
                    )
                )
                observation, info =self.debug_agent._execute_tool(tool_call_obj)
                self.terminal_output = observation
                if info.get("exit_code") == 0 and os.path.exists(os.path.join(self.workspace_path, "submission", f"submission_{self.uid}.csv")):
                    debug_success = True
                else:
                    debug_success = False
                self.logger.info(f"Debug Agent execute_bash result: {observation}")
                self.logger.info(f"Debug Agent execute_bash info: {info}")
                self.logger.info("Debug completed")
                self.logger.info(f"Debug result: {debug_result[:2000]}...")
                self.debug_agent._prompt_format_kwargs = debug_original_format_kwargs

                if self.metric_agent and debug_success:
                    self.logger.info("=" * 60)
                    self.logger.info("Step 4: Metric Agent executing task...")
                    self.logger.info("=" * 60)
                    metric_original_format_kwargs = self.metric_agent._prompt_format_kwargs.copy()
                    self.metric_agent._prompt_format_kwargs.update({
                        'terminal_output': observation
                    })
                    metric_task = TaskInstance(
                        task_id=f"{task_id}_metric",
                        task_type="metric",
                        input_data={},
                    )

                    metric_trajectory = self.metric_agent.run(metric_task)

                    metric_result = self._extract_agent_response(metric_trajectory)
                    try:
                        validation_score = float(metric_result.split("\\boxed{")[1].split("}")[0])
                    except:
                        debug_success = False
                        validation_score = None
                    self.logger.info(f"validation score: {validation_score}")
                    self.logger.info("Metric completed")
                    self.logger.info(f"Metric result: {metric_result[:2000]}...")
                    self.metric_agent._prompt_format_kwargs = metric_original_format_kwargs

                if debug_success:
                    is_success = True
                    validation_score = validation_score
                    return is_success, validation_score, self.uid, self.code

                else:
                    is_success = False
                    validation_score = None
                    debug_times += 1
            
            return is_success, validation_score, self.uid, self.code

        except Exception as e:
            self.logger.error(f"Improve task execution failed: {e}", exc_info=True)
            raise ValueError(f"Improve task execution failed: {e}")


