"""Plan-Execute Playground Implementation

Plan-Execute workflow:
1. planner: generates execution plans
2. executor: executes plans and returns results

"""

import logging
import sys
import re
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evomaster.core import BasePlayground, register_playground
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evomaster.agent import Agent

from .exp import PlanExecuteExp

@register_playground("browse_master")
class BrowseMasterPlayground(BasePlayground):
    """BrowseMaster Playground
    """
    
    def __init__(self, config_dir: str | Path | None = None, config_path: str | Path | None = None):
        """Init BrowseMaster Playground
        
        Args:
            config_dir: Configuration directory path, default to configs/agent/browse_master
            config_path: Full path to configuration file
        """
        # Set default configuration directory (compatible with str/Path types)
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "agent" / "browse_master"
        

        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.agents.declare("planner","executor")
        
        
        self.mcp_manager = None
        
    
    def setup(self) -> None:
        """Initialize all components
        """
        self.logger.info("Setting up Browse-Master playground...")
        self._setup_session()
        self._setup_agents()
        self.logger.info("Browse-Master playground setup complete")
    
    
    def _create_exp(self):
        """Create multi-agent experiment instance

        Overrides base class method, creates MultiAgentExp instance.

        Returns:
            MultiAgentExp instance
        """
        exp = PlanExecuteExp(
            planner=self.agents.planner_agent,    
            executor=self.agents.executor_agent,
            config=self.config
        )
        # Pass run_dir to Exp
        if self.run_dir:
            exp.set_run_dir(self.run_dir)
        return exp

    def run(self, task_description: str, output_file: str | None = None) -> dict:
        """Run workflow (overrides base class method)

        Args:
            task_description: Task description
            output_file: Result save file (optional, automatically saves to trajectories/ if run_dir is set)

        Returns:
            Run result
        """
        try:
            self.setup()

            # Set trajectory file path
            self._setup_trajectory_file(output_file)

            # Create and run experiment
            exp = self._create_exp()

            self.logger.info("Running experiment...")
            # If task_id exists, pass to exp.run()
            task_id = getattr(self, 'task_id', None)

            max_round = 10
            executor_result = None
            answer_list = []
            for _ in range(max_round) :
                if executor_result != None :
                    new_task_description = task_description + executor_result
                else :
                    new_task_description = f"Total task:{task_description}"

                if task_id:
                    result = exp.run(new_task_description, task_id=task_id)
                else:
                    result = exp.run(new_task_description)
                
                if result['final_found'] == 1 :
                    final_answer = result['final_answer']
                    self.logger.info(f"Final answer: {final_answer}")
                    break

                else :
                    tmp_answer = extract_executor_answer(result['executor_result'])
                    answer_list.append(tmp_answer)
                    answer_list_str = " ".join(answer_list)
                    executor_result = f"Here is the previous analysis record:{answer_list_str}"

            return result

        finally:
            self.cleanup()

def extract_executor_answer(text: str) -> str:
    """Extract result from Executor response

    First attempts to extract content within <results> tags,
    finally returns original text if neither is found

    Args:
        text: Executor response text

    Returns:
        Extracted result text
    """
    pattern = r'<results>\s*((?:(?!</results>).)*?)</results>'
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    else:
        pattern = r'</think>\s*(.*?)$'
        matches = list(re.finditer(pattern, text, re.DOTALL))
        if matches:
            return matches[-1].group(1).strip()
        else:
            return text.strip()