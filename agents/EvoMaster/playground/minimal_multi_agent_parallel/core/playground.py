"""Multi-Agent Playground Implementation

Demonstrates how to use multiple Agents collaborating on tasks.
Contains the workflow for Planning Agent and Coding Agent.
"""

import logging
import sys
from pathlib import Path

# Ensure evomaster module can be imported
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evomaster.core import BasePlayground, register_playground
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evomaster.agent import Agent

from .exp import MultiAgentExp
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import List, Any, Callable

@register_playground("minimal_multi_agent_parallel")
class MultiAgentParallelPlayground(BasePlayground):
    """Multi-Agent Parallel Playground

    Implements the collaborative workflow of Planning Agent and Coding Agent:
    1. Planning Agent analyzes the task and formulates a plan
    2. Coding Agent executes code tasks based on the plan

    Usage:
        # Via the unified entry point
        python run.py --agent minimal_multi_agent --task "task description"

        # Or via the standalone entry point
        python playground/minimal_multi_agent/main.py
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        """Initialize Multi-Agent Parallel Playground.

        Args:
            config_dir: Configuration directory path, defaults to configs/minimal_multi_agent/
            config_path: Full path to config file (overrides config_dir if provided)
        """
        if config_path is None and config_dir is None:
            # Default configuration directory
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "agent" / "minimal_multi_agent"

        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("planning_agent", "coding_agent")

        # Read parallel configuration from config
        session_config = self.config.session.get("local", {})
        parallel_config = session_config.get("parallel", {})
        if parallel_config.get("enabled", False):
            self.max_workers = parallel_config.get("max_parallel", 3)
        else:
            self.max_workers = 3
        
        # Initialize mcp_manager (required by BasePlayground.cleanup)
        self.mcp_manager = None

    def setup(self) -> None:
        """Initialize all components."""
        self.logger.info("Setting up minimal multi-agent parallel playground...")

        self._setup_session()
        self._setup_agents()

        self.logger.info("Minimal multi-agent parallel playground setup complete")

    def _create_exp(self, exp_index):
        """Create a multi-agent experiment instance.

        Overrides the base class method to create a MultiAgentExp instance.
        Creates independent Agent copies for each exp to avoid context conflicts during parallel execution.

        Args:
            exp_index: Experiment index

        Returns:
            MultiAgentExp instance
        """
        # Create independent Agent copies for each exp
        # Each agent copy has its own LLM instance (not shared) to avoid conflicts during parallelism
        # Shares session, tools, skill_registry, etc., but has independent context
        planning_agent_copy = self.copy_agent(
            self.agents.planning_agent, 
            new_agent_name=f"planning_exp_{exp_index}"
        ) if self.agents.planning_agent else None
        
        coding_agent_copy = self.copy_agent(
            self.agents.coding_agent, 
            new_agent_name=f"coding_exp_{exp_index}"
        ) if self.agents.coding_agent else None
        
        exp = MultiAgentExp(
            planning_agent=planning_agent_copy,
            coding_agent=coding_agent_copy,
            config=self.config,
            exp_index=exp_index
        )
        # Pass run_dir to Exp
        if self.run_dir:
            exp.set_run_dir(self.run_dir)
        return exp


    def run(self, task_description: str, output_file: str | None = None) -> dict:
        """Run the workflow (overrides base class method).

        Args:
            task_description: Task description
            output_file: Result save file (optional; automatically saves to trajectories/ if run_dir is set)

        Returns:
            Run result
        """
        try:
            self.setup()
            self._setup_trajectory_file(output_file)
            task_description_1 = task_description
            task_description_2 = task_description
            task_description_3 = task_description
            # --- Key step: create task list ---
            task_descriptions = [task_description_1, task_description_2, task_description_3]
            tasks = []
            for i in range(self.max_workers):
                exp = self._create_exp(exp_index=i)
                
                task_func = partial(exp.run, task_description=task_descriptions[i])
                
                tasks.append(task_func)
            
            # --- Call the wrapped parallel execution function ---
            results = self.execute_parallel_tasks(tasks, max_workers=self.max_workers)
            
            result = {
                "status": "completed",
                "steps": 0,
            }
            return result

        finally:
            self.cleanup()


