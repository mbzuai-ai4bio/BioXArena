"""Agent Builder Playground

Meta-level Agent system containing two agents:
- planner: Deeply researches user requirements, designs Agent schemes, writes to Feishu documents
- builder: Generates Agent files (config, prompts) based on the scheme
"""

import logging
from pathlib import Path

from evomaster.core import BasePlayground, register_playground


@register_playground("agent_builder")
class AgentBuilderPlayground(BasePlayground):
    """Agent Builder Playground

    Dual-agent system:
    1. planner_agent: Research framework -> Analyze requirements -> Architecture decisions -> Design prompts -> Write Feishu doc
    2. builder_agent: Read Feishu doc -> Generate directories/files -> Validate
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        """Initialize AgentBuilderPlayground.

        Args:
            config_dir: Configuration directory path, defaults to configs/agent_builder/
            config_path: Full path to config file (overrides config_dir if provided)
        """
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "agent_builder"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        # Declare two agent slots (corresponding to agents.planner / agents.builder in config.yaml)
        self.agents.declare("planner_agent", "builder_agent")

    def setup(self) -> None:
        """Initialize all components."""
        self.logger.info("Setting up Agent Builder playground...")
        self._setup_session()
        self._setup_agents()
        # planner is the primary agent: dispatcher uses planner for the first run()
        # builder is called separately by the dispatcher after confirmation
        self.agent = self.agents.planner_agent
        self.logger.info("Agent Builder playground setup complete (planner as primary agent)")
