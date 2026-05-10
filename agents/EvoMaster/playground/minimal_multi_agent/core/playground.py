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


@register_playground("minimal_multi_agent")
class MultiAgentPlayground(BasePlayground):
    """Multi-Agent Playground

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
        """Initialize Multi-Agent Playground.

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
        
        # Initialize mcp_manager (required by BasePlayground.cleanup)
        self.mcp_manager = None

    def setup(self) -> None:
        """Initialize all components."""
        self.logger.info("Setting up minimal multi-agent playground...")

        self._setup_session()
        self._setup_agents()

        self.logger.info("Minimal multi-agent playground setup complete")


    def _create_exp(self):
        """Create a multi-agent experiment instance.

        Overrides the base class method to create a MultiAgentExp instance.

        Returns:
            MultiAgentExp instance
        """
        exp = MultiAgentExp(
            planning_agent=self.agents.planning_agent,
            coding_agent=self.agents.coding_agent,
            config=self.config
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

            # Set trajectory file path
            self._setup_trajectory_file(output_file)

            # Create and run experiment
            exp = self._create_exp()

            self.logger.info("Running experiment...")
            # If task_id exists, pass to exp.run()
            task_id = getattr(self, 'task_id', None)
            if task_id:
                result = exp.run(task_description, task_id=task_id)
            else:
                result = exp.run(task_description)

            return result

        finally:
            self.cleanup()

