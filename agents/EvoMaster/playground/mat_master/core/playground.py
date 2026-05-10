"""Mat Master Playground Implementation

Materials science / computational materials EvoMaster agent, integrating Mat's MCP tools
(Structure Generator, Science Navigator, Document Parser, DPA Calculator).
Uses MatMasterAgent: supports functions.finish normalization, ends only when task_completed==true.
mat_master overrides _setup_mcp_tools here to set tool_include_only before initializing MCP
(registers only specified tools), without modifying the base class core/playground.py.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict

from evomaster.core import BasePlayground, register_playground
from evomaster.agent.tools import MCPToolManager

from .agent import MatMasterAgent


@register_playground("mat_master")
class MatMasterPlayground(BasePlayground):
    """Mat Master Playground

    A materials-science-oriented playground using Mat's MCP services (structure generation,
    science navigation, document parsing, DPA calculation), supporting LiteLLM and Azure LLM configurations.
    Uses MatMasterAgent: does not end due to partial completion when async tasks are unfinished;
    requires task_completed=true to end.

    Usage:
        python run.py --agent mat_master --task "materials-related task"
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        """Initialize MatMasterPlayground.

        Args:
            config_dir: Configuration directory path, defaults to configs/mat_master/
            config_path: Full path to config file (overrides config_dir if provided)
        """
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "mat_master"

        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _create_agent(
        self,
        name: str,
        agent_config: dict,
        enable_tools: bool = True,
        llm_config_dict: dict | None = None,
        skill_registry=None,
    ):
        """Create a Mat Master specific Agent (MatMasterAgent); otherwise same as base class."""
        from evomaster.agent import AgentConfig
        from evomaster.agent.context import ContextConfig
        from evomaster.utils import LLMConfig, create_llm

        max_turns = agent_config.get("max_turns", 20)
        context_config_dict = agent_config.get("context", {})
        context_config = ContextConfig(**context_config_dict)
        agent_cfg = AgentConfig(max_turns=max_turns, context_config=context_config)
        output_config = self._get_output_config()

        if llm_config_dict is None:
            llm_config_dict = self._setup_llm_config()
        llm = create_llm(LLMConfig(**llm_config_dict), output_config=output_config)
        self.logger.debug(f"Created independent LLM instance for {name} agent")

        system_prompt_file = agent_config.get("system_prompt_file")
        user_prompt_file = agent_config.get("user_prompt_file")
        playground_base = Path(str(self.config_dir).replace("configs", "playground"))
        if system_prompt_file:
            prompt_path = Path(system_prompt_file)
            if not prompt_path.is_absolute():
                system_prompt_file = str((playground_base / prompt_path).resolve())
        if user_prompt_file:
            prompt_path = Path(user_prompt_file)
            if not prompt_path.is_absolute():
                user_prompt_file = str((playground_base / prompt_path).resolve())
        prompt_format_kwargs = agent_config.get("prompt_format_kwargs", {})

        agent = MatMasterAgent(
            llm=llm,
            session=self.session,
            tools=self.tools,
            system_prompt_file=system_prompt_file,
            user_prompt_file=user_prompt_file,
            prompt_format_kwargs=prompt_format_kwargs,
            config=agent_cfg,
            skill_registry=skill_registry,
            output_config=output_config,
            config_dir=self.config_dir,
            enable_tools=enable_tools,
        )
        agent.set_agent_name(name)
        return agent

    def _configure_mcp_manager(self, manager: MCPToolManager, mcp_config: Dict[str, Any]) -> None:
        """Mat Master: configure calculation path adaptor and tool_include_only."""

        # 1. Configure calculation path adaptor
        if mcp_config.get("path_adaptor") == "calculation":
            from playground.mat_master.adaptors.calculation import get_calculation_path_adaptor

            calc_servers = mcp_config.get("calculation_servers")
            if calc_servers:
                manager.path_adaptor_servers = set(calc_servers)
            else:
                # If not specified, the config must specify calculation_servers
                self.logger.warning("calculation_servers not specified in config, path adaptor may not work correctly")
                manager.path_adaptor_servers = set()

            manager.path_adaptor_factory = lambda: get_calculation_path_adaptor(mcp_config)
            self.logger.info("Calculation path adaptor enabled for servers: %s",
                            manager.path_adaptor_servers)

        # 2. Configure tool_include_only (selectively register MCP tools)
        include_only = mcp_config.get("tool_include_only")
        if include_only and isinstance(include_only, dict):
            manager.tool_include_only = {
                k: list(v) if isinstance(v, (list, tuple)) else []
                for k, v in include_only.items()
            }
            self.logger.info("MCP tool_include_only set for servers: %s",
                            list(manager.tool_include_only.keys()))

