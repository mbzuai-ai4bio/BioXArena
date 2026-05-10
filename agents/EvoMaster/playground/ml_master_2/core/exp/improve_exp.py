import logging
from pathlib import Path
from typing import Any, Tuple
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
from ..utils.code import read_code,save_code_to_file
import uuid
import os
from evomaster.agent import BaseAgent

try:
    from ..utils.grading import validate_submission
    _HAS_GRADING = True
except ImportError as e:
    _HAS_GRADING = False
    _GRADING_IMPORT_ERROR = str(e)

class ImproveExp(BaseExp):
    """Experiment for applying a specific improvement idea to an existing ML solution.

    Orchestrates the improve -> execute -> metric -> debug cycle to implement
    and validate a single improvement idea.
    """

    def __init__(self, improve_agent, debug_agent, metric_agent, config, exp_name):
        super().__init__(improve_agent, config)
        self.improve_agent = improve_agent
        self.debug_agent = debug_agent
        self.metric_agent = metric_agent
        self.uid = uuid.uuid4()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.terminal_output = ""
        self.code = ""
        self.debug_times = 0
        self._exp_name = exp_name
        self.workspace_path = os.path.join(self.improve_agent.session.config.workspace_path, self.exp_name)

    @property
    def exp_name(self) -> str:
        """Return the experiment stage name."""
        return self._exp_name

    def _check_grading_valid(self, submission_path: str) -> Tuple[bool, str]:
        """Validate submission format using the grading server.

        Args:
            submission_path: Path to the submission CSV file.

        Returns:
            A tuple of (passed, reason). reason is empty string if passed.
        """
        if not _HAS_GRADING:
            return True, ""
        servers = getattr(self.config, "grading_servers", []) or []
        exp_id = getattr(self.config, "competition_id", None) or getattr(self.config, "exp_id", None)
        data_root = os.path.join(os.getcwd(), getattr(self.config, "data_root", None))
        if not servers or not exp_id:
            return True, ""
        ok, res = validate_submission(
            exp_id,
            Path(submission_path),
            server_urls=servers,
            dataset_root=data_root,
        )
        if not ok:
            reason = str(res) if res else "grading_server call failed"
            self.logger.warning(
                "grading_server call failed, treating submission format as valid by default: %s", reason
            )
            return True, ""
        if isinstance(res, dict) and not res.get("is_valid", True):
            reason = res.get("result") or res.get("details") or str(res)
            self.logger.warning("grading_server format validation failed: %s", reason)
            return False, reason
        return True, ""

    def run(self, task_description: str, data_preview: str, best_solution: str, idea: str, task_id: str = "exp_001") -> dict:
        """Execute the improvement experiment pipeline.

        Applies an improvement idea to the existing best solution, executes it,
        validates the submission, extracts metrics, and retries with debug agent on failure (up to 3 times).

        Args:
            task_description: Natural language description of the ML task.
            data_preview: Textual preview of the dataset.
            best_solution: Current best solution code to improve upon.
            idea: The specific improvement idea to implement.
            task_id: Unique task identifier.

        Returns:
            Tuple of (is_success, validation_score, uid, code).
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
                        arguments='{"command": "python run.py","timeout": "86400"}'
                    )
                )
                observation, info =self.improve_agent._execute_tool(tool_call_obj)
                self.terminal_output = observation
                submission_path = os.path.join(self.workspace_path, "submission", f"submission_{self.uid}.csv")
                if info.get("exit_code") == 0 and os.path.exists(submission_path):
                    grading_ok, grading_reason = self._check_grading_valid(submission_path)
                    is_success = grading_ok
                    if not grading_ok:
                        self.terminal_output = (
                            f"{self.terminal_output}\n\n"
                            "[grading] Code ran successfully, but submission format is invalid. grading_server validation result: "
                            f"{grading_reason}"
                        )
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
                        arguments='{"command": "python run.py","timeout": "86400"}'
                    )
                )
                observation, info =self.debug_agent._execute_tool(tool_call_obj)
                self.terminal_output = observation
                submission_path = os.path.join(self.workspace_path, "submission", f"submission_{self.uid}.csv")
                if info.get("exit_code") == 0 and os.path.exists(submission_path):
                    grading_ok, grading_reason = self._check_grading_valid(submission_path)
                    debug_success = grading_ok
                    if not grading_ok:
                        self.terminal_output = (
                            f"{self.terminal_output}\n\n"
                            "[grading] Code ran successfully, but submission format is invalid. grading_server validation result: "
                            f"{grading_reason}"
                        )
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


