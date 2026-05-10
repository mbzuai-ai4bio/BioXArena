"""EvoMaster Playground Base Class

Defines the common execution logic for workflows.
"""
from __future__ import annotations

import threading
import asyncio
import logging
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import random

from evomaster.config import ConfigManager
from evomaster.utils import LLMConfig, create_llm
from evomaster.agent import create_default_registry, create_registry, BaseAgent, Agent, AgentConfig
from evomaster.agent.context import ContextConfig
from evomaster.agent.session import LocalSession, LocalSessionConfig, DockerSession, DockerSessionConfig
from evomaster.agent.tools import MCPToolManager
from evomaster.skills import SkillRegistry
from .exp import BaseExp
from typing import List, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed


# Global mapping: thread_id -> playground instance (overwritten when thread pool reuses threads)
_thread_playground_map: dict[int, object] = {}


class _SessionThreadFilter(logging.Filter):
    """Only passes log records from threads currently working for this session.

    Uses the global _thread_playground_map to determine thread ownership. When the
    thread pool reuses threads, the new register_thread call overwrites the old mapping,
    ensuring logs are only written to the current session's file.
    """

    def __init__(self, playground):
        """Initialize the filter.

        Args:
            playground: The playground instance to filter for.
        """
        super().__init__()
        self._playground = playground

    def filter(self, record):
        """Filter log records based on thread ownership.

        Args:
            record: The log record to filter.

        Returns:
            True if the record should be passed through, False otherwise.
        """
        owner = _thread_playground_map.get(record.thread)
        # No mapping (e.g., CLI mode setup phase, non-worker threads) -> pass through
        if owner is None:
            return True
        return owner is self._playground


class AgentSlots(dict):
    """Agent container compatible with both dict and attribute access (self.agents.xxx)."""

    def declare(self, *names: str) -> "AgentSlots":
        """Pre-declare slot names for IDE auto-completion and to avoid copying YAML strings everywhere.

        Args:
            *names: Slot names to declare.

        Returns:
            Self, for method chaining.
        """
        for name in names:
            self.setdefault(name, None)
        return self

    def __getattr__(self, name: str):
        """Get an agent by attribute access.

        Args:
            name: Agent slot name.

        Returns:
            The agent instance.

        Raises:
            ValueError: If the agent slot exists but is not initialized.
            AttributeError: If the slot does not exist.
        """
        if name in self:
            value = self[name]
            if value is None:
                raise ValueError(f"Agent not initialized: {name}")
            return value
        raise AttributeError(name)

    def __setattr__(self, name: str, value):
        """Set an agent by attribute access.

        Args:
            name: Agent slot name.
            value: The agent instance to set.
        """
        self[name] = value

    def __dir__(self):
        """Return sorted list of all available attributes and keys.

        Returns:
            Sorted list of attribute/key names.
        """
        return sorted(set(super().__dir__()) | set(self.keys()))

    def get_random_agent(self) -> BaseAgent:
        """Return a randomly selected agent from the container.

        Returns:
            A randomly selected BaseAgent instance.
        """        
        return random.choice(list(self.values()))

class BasePlayground:
    """Playground base class.

    Defines the common lifecycle management for workflows:
    1. Load configuration
    2. Initialize all components
    3. Create and run experiments
    4. Clean up resources

    Concrete playgrounds can:
    - Inherit from this class
    - Override _create_exp() to use a custom Exp class
    - Override setup() to add extra initialization logic
    """

    def __init__(self, config_dir: str | Path | None = None, config_path: str | Path | None = None):
        """Initialize the Playground.

        Args:
            config_dir: Configuration directory (defaults to configs/).
            config_path: Full path to the configuration file (overrides config_dir if provided).
        """
        # If config_path is provided, extract config_dir and config_file from it
        if config_path is not None:
            config_path = Path(config_path)
            self.config_dir = config_path.parent
            config_file = config_path.name
        else:
            # Otherwise use config_dir and the default config.yaml
            if config_dir is None:
                config_dir = Path(__file__).parent.parent.parent / "configs"
            self.config_dir = Path(config_dir)
            config_file = None  # Use ConfigManager's default value config.yaml

        self.config_manager = ConfigManager(config_dir=self.config_dir, config_file=config_file)
        self.config = self.config_manager.load()
        self.config_path = self.config_dir / self.config_manager.config_file  # Save the actual config file path used
        self.logger = logging.getLogger(self.__class__.__name__)
        self._mcp_loop = None
        self._mcp_thread = None


        # Run directory management
        self.run_dir = None
        self.log_file_handler = None

        # Component storage
        self.session = None
        self.agents = AgentSlots()
        self.exps = {}
        self.tools = None
        self._mcp_managers: dict[str, Any] = {}
        self._base_skill_registry = None
        self.openclaw_bridge = None

    @property
    def mcp_manager(self):
        """Backward-compatible accessor: returns the first MCP manager or None."""
        if not self._mcp_managers:
            return None
        return next(iter(self._mcp_managers.values()))

    @mcp_manager.setter
    def mcp_manager(self, value):
        """Backward-compatible setter for subclass ``self.mcp_manager = None``."""
        if value is None:
            self._mcp_managers.clear()
        else:
            self._mcp_managers["_default"] = value

    def _resolve_mcp_config_key(self, mcp_config_file: str) -> str:
        """Resolve an MCP config file path to a canonical dict key.

        Args:
            mcp_config_file: Raw config file path (relative or absolute).

        Returns:
            Resolved absolute path string used as key in ``_mcp_managers``.
        """
        config_path = Path(mcp_config_file)
        if not config_path.is_absolute():
            config_path = self.config_manager.config_dir / config_path
        return str(config_path.resolve())

    def _start_loop_in_thread(self) -> threading.Thread:
        """Start an asyncio event loop in a daemon thread for MCP.

        Returns:
            The started daemon thread.
        """
        def _runner():
            asyncio.set_event_loop(self._mcp_loop)
            self._mcp_loop.run_forever()

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        return t
    
    def set_run_dir(self, run_dir: str | Path, task_id: str | None = None) -> None:
        """Set up the run directory and create the directory structure

        Creates the following directory structure:
        - run_dir/config.yaml (configuration file copy)
        - run_dir/logs/ (log files)
        - run_dir/trajectories/ (conversation trajectories)
        - run_dir/workspace/ or run_dir/workspaces/{task_id}/ (workspace)

        Args:
            run_dir: Path to the run directory
            task_id: Task ID (optional). If provided, the workspace will be created under
                    workspaces/{task_id}/, used for batch task scenarios
        """
        self.run_dir = Path(run_dir)
        self.task_id = task_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.run_dir / "logs").mkdir(exist_ok=True)
        (self.run_dir / "trajectories").mkdir(exist_ok=True)

        # Copy config file to run_dir (only for the first task, to avoid concurrency conflicts)
        config_copy = self.run_dir / "config.yaml"
        if self.config_path.exists() and not config_copy.exists():
            shutil.copy2(self.config_path, config_copy)
            self.logger.info(f"Copied config to: {config_copy}")

        # Create workspace directory
        if task_id:
            # Batch task mode: workspaces/{task_id}/
            (self.run_dir / "workspaces").mkdir(exist_ok=True)
            workspace_path = self.run_dir / "workspaces" / task_id
            workspace_path.mkdir(exist_ok=True)
        else:
            # Single task mode: workspace/
            workspace_path = self.run_dir / "workspace"
            workspace_path.mkdir(exist_ok=True)

        # Dynamically update workspace_path in configuration
        self._update_workspace_path(workspace_path)

        # Set log file to run_dir/logs/
        self._setup_logging()

        self.logger.info(f"Run directory: {self.run_dir}")
        if task_id:
            self.logger.info(f"Task ID: {task_id}")
            self.logger.info(f"Workspace: {workspace_path}")

    def _update_workspace_path(self, workspace_path: Path) -> None:
        """Dynamically update the workspace_path in the configuration.

        Called before Session creation to ensure the Session uses the workspace under run_dir.

        Args:
            workspace_path: New workspace path (typically run_dir/workspace or run_dir/workspaces/{task_id}).
        """
        workspace_path_str = str(workspace_path.absolute())

        # Update workspace_path and working_dir in session configuration
        if hasattr(self.config, 'session'):
            session_config = self.config.session

            # For dict type configuration
            if isinstance(session_config, dict):
                session_type = session_config.get('type', 'local')

                # Update Local Session
                if session_type == 'local' and 'local' in session_config:
                    session_config['local']['workspace_path'] = workspace_path_str
                    session_config['local']['working_dir'] = workspace_path_str
                    self.logger.debug(f"Updated local workspace path to: {workspace_path_str}")

                # Update Docker Session
                elif session_type == 'docker' and 'docker' in session_config:
                    docker_config = session_config['docker']
                    container_workspace = docker_config.get('working_dir', '/workspace')

                    # Update volumes mount
                    if 'volumes' not in docker_config:
                        docker_config['volumes'] = {}
                    docker_config['volumes'][workspace_path_str] = container_workspace

                    # Update workspace_path
                    docker_config['workspace_path'] = container_workspace
                    docker_config['working_dir'] = container_workspace

                    self.logger.debug(f"Updated Docker volume: {workspace_path_str} -> {container_workspace}")

            # For Pydantic models (if already loaded)
            elif hasattr(session_config, 'local') and hasattr(session_config.local, 'workspace_path'):
                session_config.local.workspace_path = workspace_path_str
                session_config.local.working_dir = workspace_path_str
            elif hasattr(session_config, 'docker') and hasattr(session_config.docker, 'workspace_path'):
                session_config.docker.workspace_path = workspace_path_str
                session_config.docker.working_dir = workspace_path_str

        self.logger.info(f"Updated workspace path to: {workspace_path_str}")


    def _setup_logging(self) -> None:
        """Set up the log file path.

        Priority:
        1. If run_dir is set, use run_dir/logs/{task_id}.log or run_dir/logs/evomaster.log
        2. Otherwise use log_path from the configuration file
        3. If neither is available, do not log to file
        """
        # Remove old file handler (if exists)
        if self.log_file_handler:
            root_logger = logging.getLogger()
            root_logger.removeHandler(self.log_file_handler)
            self.log_file_handler.close()
            self.log_file_handler = None

        # Determine the log file path
        log_file = None
        if self.run_dir:
            # Prefer using run_dir
            if hasattr(self, 'task_id') and self.task_id:
                # Batch task mode: use task_id.log
                log_file = self.run_dir / "logs" / f"{self.task_id}.log"
            else:
                # Single task mode: use evomaster.log
                log_file = self.run_dir / "logs" / "evomaster.log"
        else:
            # Use path from the configuration file
            log_path = getattr(self.config.logging, 'log_path', None)
            if log_path:
                log_file = Path(log_path)

        if log_file:
            # Ensure the log directory exists
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # Create file handler (overwrite mode)
            self.log_file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            self.log_file_handler.setLevel(getattr(logging, self.config.logging.level))
            self.log_file_handler.setFormatter(logging.Formatter(self.config.logging.format))
            self.log_file_handler.addFilter(_SessionThreadFilter(self))

            # Add to root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(self.log_file_handler)

            self.log_path = str(log_file)
            self.logger.info(f"Logging to file: {log_file}")
        else:
            self.log_path = None

    def register_thread(self) -> None:
        """Register the current thread as a worker thread for this playground (used for log filtering)."""
        _thread_playground_map[threading.current_thread().ident] = self

    def _get_agents_config(self) -> dict:
        """Get the configuration for all agents.

        Returns:
            Agents configuration dictionary.
        """
        return self.config_manager.get_agents_config()

    def _get_agent_config(self, name: str) -> dict:
        """Get the configuration for a specific agent.

        Args:
            name: Agent name.

        Returns:
            Agent configuration dictionary.
        """
        return self.config_manager.get_agent_config(name)

    def _setup_agent_llm(self, agent_name: str) -> dict:
        """Get the LLM configuration for a specific agent.

        Args:
            agent_name: Agent name.

        Returns:
            LLM configuration dictionary.
        """
        return self.config_manager.get_agent_llm_config(agent_name)

    def _setup_agent_tools(self, agent_name: str) -> dict:
        """Get the tools configuration for a specific agent.

        Args:
            agent_name: Agent name.

        Returns:
            Tools configuration dictionary.
        """
        return self.config_manager.get_agent_tools_config(agent_name)

    def _setup_agent_skills(self, agent_name: str) -> dict:
        """Get the skills configuration for a specific agent.

        Args:
            agent_name: Agent name.

        Returns:
            Skills configuration dictionary.
        """
        return self.config_manager.get_agent_skills_config(agent_name)
        

    def _get_or_create_skill_registry(self, skill_config: dict | None = None) -> SkillRegistry:
        """Create a SkillRegistry based on agent config; caches the full registry when skills is '*'.

        Args:
            skill_config: Skill configuration dictionary.

        Returns:
            SkillRegistry instance.
        """
        if skill_config is None:
            skill_config = {}

        skills_root = Path(skill_config.get("skill_dir", "./evomaster/skills"))
        skills = skill_config.get("skills")

        if skills == "*" or skills == ["*"]:
            if self._base_skill_registry is None:
                self.logger.info(f"Loading full skill registry from: {skills_root}")
                self._base_skill_registry = SkillRegistry(skills_root)
                self.logger.info(f"Loaded {len(self._base_skill_registry.get_all_skills())} skills")
            return self._base_skill_registry

        if isinstance(skills, str):
            skills = [skills]

        self.logger.info(f"Loading selected skills from: {skills_root} -> {skills}")
        return SkillRegistry(skills_root, skills=skills)


    def _get_or_create_full_skill_registry(self) -> SkillRegistry:
        """Always create/get the full SkillRegistry (consistent with builtin: always register, config only controls LLM exposure).

        Also scans evomaster/skills (Python skills) and evomaster/skills_ts (TypeScript/Openclaw skills).

        Returns:
            Full SkillRegistry instance.
        """
        skills_config = getattr(self.config, "skills", None)
        if isinstance(skills_config, dict):
            skills_root = Path(skills_config.get("skills_root", "evomaster/skills"))
        else:
            skills_root = Path("evomaster/skills")
        if self._base_skill_registry is None:
            self.logger.info(f"Loading full skill registry from: {skills_root}")
            self._base_skill_registry = SkillRegistry(skills_root)
            # Also scan skills_ts directory (Openclaw skills)
            skills_ts_root = Path(skills_root).parent / "skills_ts"
            if skills_ts_root.exists():
                self.logger.info(f"Also loading skills from: {skills_ts_root}")
                self._base_skill_registry.load_from_directory(skills_ts_root)
            self.logger.info(f"Loaded {len(self._base_skill_registry.get_all_skills())} skills total")
        return self._base_skill_registry

    def _resolve_skill_registry(self, skill_config: dict | None) -> SkillRegistry | None:
        """Resolve a SkillRegistry based on skill_config (may be a subset). Used for the Agent's skill_registry parameter.

        Args:
            skill_config: Skill configuration dictionary.

        Returns:
            SkillRegistry instance, or None if no skills are configured.
        """
        if not skill_config:
            return None

        skills_config = skill_config.get("skills")

        if not skills_config:
            return None

        if isinstance(skills_config, str):
            skills_config = [skills_config]
        if not isinstance(skills_config, list):
            raise ValueError(
                "Invalid skills config. "
                "Expected list[str], '*', or omitted."
            )

        normalized_skill_config = skill_config.copy()
        normalized_skill_config["skills"] = skills_config
        return self._get_or_create_skill_registry(normalized_skill_config)

    def _setup_session(self) -> None:
        """Create and open a Session (if not already created).

        Selects a local or docker session based on configuration.
        """
        if self.session is None:
            session_type = self.config.session.get("type", "local")
            if session_type == "docker":
                session_config_dict = self.config.session.get("docker", {}).copy()
                # Sync working_dir and workspace_path
                if "working_dir" in session_config_dict and "workspace_path" not in session_config_dict:
                    session_config_dict["workspace_path"] = session_config_dict["working_dir"]
                elif "workspace_path" in session_config_dict and "working_dir" not in session_config_dict:
                    session_config_dict["working_dir"] = session_config_dict["workspace_path"]
                elif "workspace_path" not in session_config_dict and "working_dir" not in session_config_dict:
                    session_config_dict["workspace_path"] = "/workspace"
                    session_config_dict["working_dir"] = "/workspace"
                session_config = DockerSessionConfig(**session_config_dict)
                self.session = DockerSession(session_config)
                self.logger.info(f"Using Docker session with image: {session_config.image}")
            else:
                session_config_dict = self.config.session.get("local", {}).copy()
                # Sync working_dir and workspace_path
                if "working_dir" in session_config_dict and "workspace_path" not in session_config_dict:
                    session_config_dict["workspace_path"] = session_config_dict["working_dir"]
                elif "workspace_path" in session_config_dict and "working_dir" not in session_config_dict:
                    session_config_dict["working_dir"] = session_config_dict["workspace_path"]
                # Pass config_dir for resolving relative paths in symlinks
                if "config_dir" not in session_config_dict:
                    session_config_dict["config_dir"] = str(self.config_dir)
                session_config = LocalSessionConfig(**session_config_dict)
                self.session = LocalSession(session_config)
                self.logger.info("Using Local session")
        
        # Open Session (if not already open)
        if not self.session.is_open:
            self.session.open()
        else:
            self.logger.debug("Session already open, reusing existing session")

    def _setup_tools(
        self,
        skill_config: dict | None = None,
        tool_config: dict[str, Any] | None = None,
    ):
        """Create tool registry and initialize MCP tools as needed.

        Regardless of whether certain tools are enabled in tool_config, all builtin tools
        are registered in the registry.
        Skills follow the same pattern as builtin: always register SkillTool, config only
        controls whether use_skill is exposed to LLM.
        Tool "enable/disable" only affects whether tool info is exposed to LLM (via
        enable_tools and _get_tool_specs).

        Args:
            skill_config: Skill configuration, where the skills list controls which skills
                are exposed to the agent.
            tool_config: Per-agent tool configuration, of the form
                {"builtin": list[str], "mcp": str, "custom": dict}.
                mcp is the MCP config file path; empty string means disabled.
                custom is the custom tool configuration, e.g., {"search": "google_search"}.
        """
        tool_config = tool_config or {"builtin": ["*"], "mcp": ""}

        mcp_config_file = tool_config.get("mcp", "")

        # Openclaw bridge initialization (only once)
        # openclaw may be at top level or in custom (get_agent_tools_config puts non-builtin/mcp into custom)
        openclaw_config = tool_config.get("openclaw") or tool_config.get("custom", {}).get("openclaw") or {}
        # openclaw with plugins is considered enabled; "enabled" can be omitted
        openclaw_enabled = bool(openclaw_config.get("plugins")) or openclaw_config.get("enabled", False)
        if openclaw_enabled and self.openclaw_bridge is None:
            self._setup_openclaw_bridge(openclaw_config)

        # Always register all builtin tools and SkillTool (consistent with builtin), config only controls LLM exposure
        # skill_context only exposes the skills configured in config to the agent; execution still uses the full registry
        skill_registry = self._get_or_create_full_skill_registry()
        # skills: ["*"] -> expose all; not configured or skills: [] -> don't expose any skill; [x,y] -> expose only specified skills
        enabled_skills_raw = (skill_config or {}).get("skills")
        if enabled_skills_raw == ["*"]:
            enabled_skills = None  # None means all
        elif enabled_skills_raw is None or enabled_skills_raw == []:
            enabled_skills = []  # Not configured or empty list means no skills enabled
        else:
            enabled_skills = enabled_skills_raw
        self.logger.info("enabled_skills: %s", enabled_skills)
        self.tools = create_registry(
            builtin_names=["*"],
            skill_registry=skill_registry,
            openclaw_bridge=self.openclaw_bridge,
            enabled_skills=enabled_skills,
        )

        # MCP: load MCP tools when mcp_config_file is non-empty
        if mcp_config_file:
            mcp_key = self._resolve_mcp_config_key(mcp_config_file)
            if mcp_key not in self._mcp_managers:
                manager = self._setup_mcp_tools(mcp_config_file)
                if manager is not None:
                    self._mcp_managers[mcp_key] = manager
            else:
                self._mcp_managers[mcp_key].register_tools(self.tools)

        # Auto-register custom tools
        custom_tools = tool_config.get("custom", {})
        if custom_tools:
            self._register_custom_tools(custom_tools)

    def _setup_openclaw_bridge(self, openclaw_config: dict[str, Any]) -> None:
        """Initialize the Openclaw bridge subprocess.

        Creates and starts a Node.js bridge subprocess for executing Openclaw-type skills.

        Args:
            openclaw_config: Openclaw configuration dictionary, e.g.:
                {
                    "enabled": true,
                    "skills_ts_dir": "./evomaster/skills_ts",
                    "plugins": ["feishu"]
                }
        """
        from evomaster.agent.tools.openclaw_bridge import OpenclawBridge

        skills_ts_dir = Path(openclaw_config.get("skills_ts_dir", "./evomaster/skills_ts"))
        if not skills_ts_dir.is_absolute():
            skills_ts_dir = skills_ts_dir.resolve()

        plugins = openclaw_config.get("plugins", [])
        if not plugins:
            self.logger.warning("Openclaw enabled but no plugins specified, skipping bridge init")
            return

        self.logger.info(f"Starting Openclaw bridge from: {skills_ts_dir} with plugins: {plugins}")
        try:
            self.openclaw_bridge = OpenclawBridge(skills_ts_dir)
            self.openclaw_bridge.start(plugins)
            tools_info = self.openclaw_bridge.get_tools_info()
            self.logger.info(
                f"Openclaw bridge started with {len(tools_info)} tools: "
                f"{', '.join(tools_info.keys())}"
            )
        except Exception as e:
            self.logger.error(f"Failed to start Openclaw bridge: {e}", exc_info=True)
            self.openclaw_bridge = None

    def _register_custom_tools(self, custom_tools: dict[str, Any]) -> None:
        """Auto-discover and register custom tools.

        Based on custom tool configuration in the config file, automatically imports
        and registers tool classes. Supports auto-loading tools from the tools
        subdirectory under the playground directory.

        Args:
            custom_tools: Custom tool configuration, e.g., {"search": "google_search", "other": "custom_tool"}.
                Keys are tool types (e.g., "search"), values are tool names (e.g., "google_search").

        Example:
            Config: {"search": "google_search"}
            Will attempt to load: playground/{playground_name}/tools/google_search.py
            and register the tool class within (class name is typically GoogleSearchTool).
        """
        if not custom_tools:
            return

        # Infer the playground directory
        # config_dir is typically /path/to/configs/{playground_name}
        # playground_dir should be /path/to/playground/{playground_name}
        playground_dir = Path(str(self.config_dir).replace("configs", "playground"))
        tools_dir = playground_dir / "tools"

        if not tools_dir.exists():
            self.logger.warning(f"Custom tools directory not found: {tools_dir}")
            return

        self.logger.info(f"Loading custom tools from: {tools_dir}")

        for tool_key, tool_name in custom_tools.items():
            self.logger.info(f"Loading custom tool: {tool_key} -> {tool_name}")
            if not isinstance(tool_name, str):
                self.logger.warning(f"Invalid tool name for '{tool_key}': {tool_name}")
                continue

            # Try to load the tool module
            tool_module_path = tools_dir / f"{tool_name}.py"
            if not tool_module_path.exists():
                self.logger.warning(f"Tool module not found: {tool_module_path}")
                continue

            try:
                # Dynamically import the tool module
                import importlib.util
                import sys

                module_name = f"playground.{playground_dir.name}.tools.{tool_name}"
                spec = importlib.util.spec_from_file_location(module_name, tool_module_path)
                if spec is None or spec.loader is None:
                    self.logger.warning(f"Failed to load module spec for: {tool_module_path}")
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Find the tool class (typically a class inheriting from BaseTool)
                from evomaster.agent.tools.base import BaseTool
                tool_class = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and
                        issubclass(attr, BaseTool) and
                        attr is not BaseTool and
                        hasattr(attr, 'name')):
                        tool_class = attr
                        break

                if tool_class is None:
                    self.logger.warning(f"No tool class found in module: {tool_module_path}")
                    continue

                # Call subclass tool initialization method (if exists)
                tool_instance = self._create_custom_tool_instance(tool_class, tool_name, tool_key)

                if tool_instance is not None:
                    # Register the tool to all agents
                    self.tools.register(tool_instance)
                    self.logger.info(f"Registered custom tool: {tool_name} (class: {tool_class.__name__})")
                else:
                    self.logger.warning(f"Failed to create instance for tool: {tool_name}")

            except Exception as e:
                self.logger.error(f"Failed to load custom tool '{tool_name}': {e}", exc_info=True)

    def _create_custom_tool_instance(self, tool_class: type, tool_name: str, tool_key: str):
        """Create a custom tool instance.

        Subclasses can override this method to provide custom tool initialization logic.

        Args:
            tool_class: The tool class.
            tool_name: Tool name (e.g., "google_search").
            tool_key: Tool configuration key (e.g., "search").

        Returns:
            Tool instance, or None if creation fails.
        """
        # Default implementation: try no-argument construction
        try:
            return tool_class()
        except TypeError:
            # If arguments are required, subclass should override this method
            self.logger.warning(
                f"Tool class {tool_class.__name__} requires constructor arguments. "
                f"Please override _create_custom_tool_instance() in your playground class."
            )
            return None

    def _setup_agents(self) -> None:
        """Create all Agents according to configuration, initializing self.agents."""
        agents_config = self._get_agents_config()

        for agent_name, agent_config in agents_config.items():
            llm_config = self._setup_agent_llm(agent_name)
            tool_config = self._setup_agent_tools(agent_name)
            skill_config = self._setup_agent_skills(agent_name)

            agent = self._create_agent(
                name=agent_name,
                agent_config=agent_config,
                llm_config=llm_config,
                tool_config=tool_config,
                skill_config=skill_config,
            )
            setattr(self.agents, f"{agent_name}_agent", agent)

            self.logger.info(f"{agent_name.capitalize()} Agent created with:")
            self.logger.info(f"  - LLM: {llm_config['model']}")
            self.logger.info(f"  - Available Tools: {agent.tools.get_tool_names()}")
            self.logger.info(f"  - Auto Accessed Tools: {tool_config}")
            self.logger.info(f"  - Skills: {skill_config.get('skills', [])}")

        # Backward compatibility
        self.agent = self.agents.get_random_agent()

    def _setup_exps(self) -> None:
        """Set up experiments.

        TODO: Create exp instances based on the exp configuration in the config file
        and store them in self.exps dictionary. Currently simplified to a list.
        """
        pass

    def _get_output_config(self) -> dict:
        """Get LLM output configuration.

        Returns:
            Output configuration dictionary.
        """
        llm_output_config = self.config.llm_output if hasattr(self.config, 'llm_output') else {}
        if isinstance(llm_output_config, dict):
            return llm_output_config
        else:
            return {}

    def _create_agent(
        self,
        name: str,
        agent_config: dict | None = None,
        llm_config: dict | None = None,
        tool_config: dict | None = None,
        skill_config: dict | None = None,
    ):
        """Create an Agent instance.

        Each Agent uses an independent LLM instance to ensure independent logging.

        Args:
            name: Agent name.
            agent_config: Agent configuration dictionary.
            llm_config: LLM configuration dictionary.
            tool_config: Tool configuration dictionary, of the form {"builtin": list[str], "mcp": list[str]}.
            skill_config: Skill configuration dictionary.

        Returns:
            Agent instance.
        """
        # Backward compatibility: auto-fetch when not passed
        if agent_config is None:
            agent_config = self._get_agent_config(name)
        if llm_config is None:
            llm_config = self._setup_agent_llm(name)
        if tool_config is None:
            tool_config = self._setup_agent_tools(name)
        if skill_config is None:
            skill_config = self._setup_agent_skills(name)

        # Determine whether to enable tools based on tool_config
        builtin = tool_config.get("builtin", ["*"])
        mcp_config_file = tool_config.get("mcp", "")
        custom_tools = tool_config.get("custom", {})
        skills = skill_config.get("skills", [])
        # enable_tools = bool(builtin) or bool(mcp_config_file)
        if builtin == [] and mcp_config_file == "" and skill_config.get("skills", []) == [] and not custom_tools:
            enable_tools = False
        else:
            enable_tools = True

        # Create tool registry (always register all tools)
        self._setup_tools(skill_config=skill_config, tool_config=tool_config)

        enabled_tool_names = []
        if builtin == ["*"]:
            enabled_tool_names.extend(["execute_bash", "str_replace_editor", "think", "finish"])
        elif builtin != []:
            enabled_tool_names.extend(builtin)

        if mcp_config_file != "":
            mcp_key = self._resolve_mcp_config_key(mcp_config_file)
            manager = self._mcp_managers.get(mcp_key)
            if manager is not None:
                enabled_tool_names.extend(manager.get_tool_names())
            else:
                self.logger.warning(f"MCP manager not found for key: {mcp_key}")
        if skills != []:
            enabled_tool_names.extend(["use_skill"])

        # Add custom tools to enabled_tool_names
        # The tool name for custom tools is the value in the config (e.g., "search" -> "google_search" or "ai_search")
        for custom_tool_key, custom_tool_value in custom_tools.items():
            self.logger.info(f"Custom tool: {custom_tool_key} -> {custom_tool_value}")
            enabled_tool_names.append(custom_tool_value)
            continue
            # Infer actual tool names based on config values
            # For example: search: "google" -> enable google_search and web_fetch
            #              search: "ai_search" -> enable ai_search
            if custom_tool_key == "search":
                if custom_tool_value == "google":
                    # Google search mode: enable google_search and web_fetch
                    if self.tools.get_tool("google_search") is not None:
                        enabled_tool_names.append("google_search")
                    if self.tools.get_tool("web_fetch") is not None:
                        enabled_tool_names.append("web_fetch")
                elif custom_tool_value == "ai_search":
                    # AI search mode: enable ai_search
                    if self.tools.get_tool("ai_search") is not None:
                        enabled_tool_names.append("ai_search")
            else:
                # Other custom tools: use key name as tool name directly
                if self.tools.get_tool(custom_tool_key) is not None:
                    enabled_tool_names.append(custom_tool_key)

        self.logger.info(f"Enabled tools: {enabled_tool_names}")

        max_turns = agent_config.get('max_turns', 20)
        context_config_dict = agent_config.get('context', {})
        context_config = ContextConfig(**context_config_dict)
        finish_on_text_response = agent_config.get('finish_on_text_response', False)
        agent_cfg = AgentConfig(
            max_turns=max_turns,
            context_config=context_config,
            finish_on_text_response=finish_on_text_response,
        )

        # Get output configuration
        output_config = self._get_output_config()

        # Create an independent LLM instance for each Agent
        llm = create_llm(LLMConfig(**llm_config), output_config=output_config)
        self.logger.debug(f"Created independent LLM instance for {name} agent")

        # Get prompt file paths
        system_prompt_file = agent_config.get('system_prompt_file')
        user_prompt_file = agent_config.get('user_prompt_file')

        playground_base = Path(str(self.config_dir).replace("configs", "playground"))
        # Resolve system_prompt_file
        if system_prompt_file:
            prompt_path = Path(system_prompt_file)
            if not prompt_path.is_absolute():
                system_prompt_file = str((playground_base / prompt_path).resolve())

        # Resolve user_prompt_file
        if user_prompt_file:
            prompt_path = Path(user_prompt_file)
            if not prompt_path.is_absolute():
                user_prompt_file = str((playground_base / prompt_path).resolve())

        # Get prompt format kwargs (if any)
        prompt_format_kwargs = agent_config.get('prompt_format_kwargs', {})

        skill_registry = self._resolve_skill_registry(skill_config)

        # Create Agent
        agent = Agent(
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
            enabled_tool_names=enabled_tool_names,
        )

        # Set Agent name (used to identify different agents in trajectory files)
        agent.set_agent_name(name)

        # Inject summary LLM for auto-compact context compression
        agent.context_manager.set_summary_llm(llm)

        return agent

    def copy_agent(self, agent, new_agent_name: str | None = None):
        """Copy an Agent instance, creating new context but sharing other configurations.

        Creates a new Agent instance that:
        - Has an independent LLM instance (always creates a new LLM on copy, not shared)
        - Shares session, tools, skill_registry, config_dir, enable_tools, etc.
        - Has independent context (context_manager, current_dialog, trajectory, etc.)
        - Context-related state is reset (current_dialog=None, trajectory=None, _step_count=0, etc.)

        Args:
            agent: The Agent instance to copy.
            new_agent_name: Name for the new Agent (optional, for identification).

        Returns:
            A new Agent instance with the same type as the input agent.
        """
        from evomaster.agent import AgentConfig
        from evomaster.utils import LLMConfig, create_llm

        # Copy AgentConfig, especially context_config needs to be independent
        if agent.config:
            new_config = agent.config.model_copy(deep=True)
        else:
            new_config = AgentConfig()

        agent_class = agent.__class__

        if hasattr(agent.llm, "config"):
            # Compatible with Pydantic v2 (.model_dump()) and v1 (.dict())
            cfg_obj = agent.llm.config
            llm_config_dict = cfg_obj.model_dump() if hasattr(cfg_obj, "model_dump") else cfg_obj.dict()
        else:
            # Fallback: if unable to get from object, try re-reading from config manager using agent name
            source_name = getattr(agent, "name", "default")
            llm_config_dict = self._setup_agent_llm(source_name)
        output_config = agent.output_config.copy() if agent.output_config else self._get_output_config()
        new_llm = create_llm(LLMConfig(**llm_config_dict), output_config=output_config)
        self.logger.debug(f"Created independent LLM instance for copied agent: {new_agent_name}")

        shared_kwargs = {
            'llm': new_llm,
            'session': agent.session,
            'tools': agent.tools,
            'config': new_config,
            'skill_registry': agent.skill_registry,
            'output_config': agent.output_config.copy() if agent.output_config else None,
            'config_dir': agent.config_dir,
            'enable_tools': agent.enable_tools,
            'enabled_tool_names': getattr(agent, 'enabled_tool_names', None),
        }

        if agent_class.__name__ == 'Agent':
            shared_kwargs['prompt_format_kwargs'] = (
                getattr(agent, '_prompt_format_kwargs', {}).copy()
                if hasattr(agent, '_prompt_format_kwargs') else None
            )

        new_agent = agent_class(**shared_kwargs)

        if agent_class.__name__ == 'Agent':
            if hasattr(agent, '_system_prompt') and agent._system_prompt is not None:
                new_agent._system_prompt = agent._system_prompt
            if hasattr(agent, '_user_prompt') and agent._user_prompt is not None:
                new_agent._user_prompt = agent._user_prompt

        if new_agent_name:
            new_agent.set_agent_name(new_agent_name)

        new_agent.current_dialog = None
        new_agent.trajectory = None
        new_agent._step_count = 0
        new_agent._initial_system_prompt = None
        new_agent._initial_user_prompt = None

        return new_agent

    def setup(self) -> None:
        """Initialize all components.

        Concrete implementation includes:
        1. Create Session (if not already created)
        2. Initialize Agent(s) (internally handles llm/tools/skills per agent)
        3. Initialize Exp(s) (TODO)
        """
        self.logger.info("Setting up playground...")

        self._setup_session()
        self._setup_agents()
        # TODO: self._setup_exps()

        self.logger.info("Multi-agent playground setup complete")

    def _setup_mcp_tools(self, config_file: str):
        """Initialize MCP tools.

        Reads the server list from an MCP configuration file (JSON format),
        initializes connections, and registers tools.

        Args:
            config_file: MCP configuration file path (relative to config_dir or absolute path).

        Returns:
            MCPToolManager instance, or None if the configuration is invalid.
        """
        # 1. Resolve the configuration file path
        config_path = Path(config_file)
        if not config_path.is_absolute():
            config_path = self.config_manager.config_dir / config_path

        if not config_path.exists():
            self.logger.error(f"MCP config file not found: {config_path}")
            return None

        # 2. Load MCP configuration
        self.logger.info(f"Loading MCP config from: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                mcp_servers_config = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load MCP config: {e}")
            return None
        
        # --- PATCH: replace placeholder paths in MCP config (global) ---
        PLACEHOLDER = "__EVOMASTER_WORKSPACES__"

        def _deep_replace(obj, old: str, new: str):
            """Recursively replace `old` -> `new` in any string inside dict/list structures."""
            if isinstance(obj, str):
                return obj.replace(old, new)
            if isinstance(obj, list):
                return [_deep_replace(x, old, new) for x in obj]
            if isinstance(obj, dict):
                return {k: _deep_replace(v, old, new) for k, v in obj.items()}
            return obj

        try:
            if self.run_dir is not None:
                ws_root = str((Path(self.run_dir) / "workspaces").resolve())
                mcp_servers_config = _deep_replace(mcp_servers_config, PLACEHOLDER, ws_root)
                self.logger.info(f"[MCP] Replaced {PLACEHOLDER} -> {ws_root}")
            else:
                self.logger.debug(f"[MCP] run_dir is None, skip placeholder replace: {PLACEHOLDER}")
        except Exception as e:
            self.logger.warning(f"[MCP] Failed to replace placeholder paths: {e}")
        
        # 6. Parse server configuration
        servers = self._parse_mcp_servers(mcp_servers_config)
        if not servers:
            self.logger.warning("No valid MCP servers found in config")
            return None

        # 7. Initialize MCP manager
        self.logger.info("Setting up MCP tools...")
        manager = MCPToolManager()

        # Subclasses can override this method to inject custom logic (e.g., path adaptor, tool_include_only)
        mcp_config = getattr(self.config, 'mcp', {}) or {}
        self._configure_mcp_manager(manager, mcp_config)

        # 8. Async initialization of MCP servers
        async def init_mcp_servers():
            for server_config in servers:
                try:
                    await manager.add_server(**server_config)
                except Exception as e:
                    self.logger.error(f"Failed to add MCP server {server_config.get('name')}: {e}")

        # Create and save a long-lived event loop dedicated to MCP
        if self._mcp_loop is None or self._mcp_loop.is_closed():
            self._mcp_loop = asyncio.new_event_loop()
            self._mcp_thread = self._start_loop_in_thread()

        manager.loop = self._mcp_loop

        # Submit coroutine to mcp_loop
        future = asyncio.run_coroutine_threadsafe(init_mcp_servers(), self._mcp_loop)
        future.result()  # Block and wait for initialization to complete (only option in sync code)

        # 9. Register MCP tools to the main tool registry
        manager.register_tools(self.tools)

        tool_count = len(manager.get_tool_names())
        server_count = len(manager.get_server_names())
        self.logger.info(f"MCP tools setup complete: {tool_count} tools from {server_count} servers")

        return manager

    def _configure_mcp_manager(self, manager: MCPToolManager, mcp_config: Dict[str, Any]) -> None:
        """Hook method for configuring the MCP manager.

        Subclasses can override this method to inject custom logic, such as:
        - Path adaptor
        - tool_include_only (tool filtering)
        - Other custom configurations

        Args:
            manager: MCP tool manager instance.
            mcp_config: MCP configuration dictionary.

        Example:
            class MyPlayground(BasePlayground):
                def _configure_mcp_manager(self, manager, mcp_config):
                    # Inject a custom adaptor
                    manager.path_adaptor_factory = lambda: MyAdaptor()
        """
        # Base class does nothing by default
        # Subclasses can override to add custom logic
        pass

    def _parse_mcp_servers(self, mcp_config: dict) -> list[dict]:
        """Parse MCP server configuration.

        Supports standard MCP format and extended format.

        Args:
            mcp_config: MCP configuration dictionary.

        Returns:
            List of server configurations.
        """
        servers = []
        mcp_servers = mcp_config.get('mcpServers', {})

        for name, config in mcp_servers.items():
            if 'command' in config:
                # Standard format (stdio)
                servers.append({
                    'name': name,
                    'transport': 'stdio',
                    'command': config['command'],
                    'args': config.get('args', []),
                    'env': config.get('env', {})
                })
            elif 'transport' in config:
                # Extended format (http/sse)
                transport = config['transport'].lower()
                if transport in ['http', 'sse', 'streamable_http', 'streamable-http']:
                    servers.append({
                        'name': name,
                        'transport': transport,
                        'url': config['url'],
                        'headers': config.get('headers', {})
                    })
                else:
                    self.logger.warning(f"Unsupported transport for server {name}: {transport}")
            else:
                self.logger.warning(f"Invalid config for server {name}: missing 'command' or 'transport'")

        return servers

    def _create_exp(self):
        """Create an Exp instance.

        Subclasses can override this method to use a custom Exp class.

        Returns:
            BaseExp instance.
        """
        exp = BaseExp(self.agent, self.config)
        # Pass run_dir to Exp
        if self.run_dir:
            exp.set_run_dir(self.run_dir)
        return exp

    def _setup_trajectory_file(self, output_file: str | Path | None = None) -> Path | None:
        """Set up the trajectory file path.

        Determines the trajectory file path and sets it on BaseAgent. Priority:
        1. If output_file is provided, use that path
        2. If run_dir is set, auto-save to trajectories/
           - Batch task mode: trajectories/{task_id}/trajectory.json
           - Single task mode: trajectories/trajectory.json

        Args:
            output_file: Result save file path (optional).

        Returns:
            Trajectory file path, or None if not set.
        """
        trajectory_file = None
        if output_file:
            trajectory_file = Path(output_file)
        elif self.run_dir:
            # If run_dir is set, auto-save to trajectories/
            if hasattr(self, 'task_id') and self.task_id:
                # Batch task mode: save to trajectories/{task_id}/trajectory.json
                trajectory_dir = self.run_dir / "trajectories" / self.task_id
                trajectory_dir.mkdir(parents=True, exist_ok=True)
                trajectory_file = trajectory_dir / "trajectory.json"
            else:
                # Single task mode: save to trajectories/trajectory.json
                trajectory_file = self.run_dir / "trajectories" / "trajectory.json"
        
        # Set trajectory file path to BaseAgent
        if trajectory_file:
            from evomaster.agent import BaseAgent
            BaseAgent.set_trajectory_file_path(trajectory_file)
            self.logger.info(f"Trajectory file set to: {trajectory_file}")
        else:
            self.logger.warning("No trajectory file set")
        return trajectory_file

    def run(self, task_description: str, output_file: str | None = None, images: list[str] | None = None, on_step=None) -> dict:
        """Run the workflow.

        Args:
            task_description: Task description.
            output_file: Result save file (optional; if run_dir is set, auto-saves to trajectories/).
            images: List of image file paths (optional, for multimodal tasks).
            on_step: Step callback, signature (StepRecord, step_number, max_steps) -> None.

        Returns:
            Run result.
        """
        try:
            # Register current thread (for log filtering)
            self.register_thread()

            self.setup()

            # Set up trajectory file path
            self._setup_trajectory_file(output_file)

            # Create and run experiment
            exp = self._create_exp()

            self.logger.info("Running experiment...")
            result = exp.run(task_description, images=images, on_step=on_step)

            return result

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources.

        For DockerSession, if auto_remove=False, the container is kept alive and the
        session is not closed, allowing reuse in subsequent runs.
        """
        # Clean up Openclaw bridge
        if self.openclaw_bridge is not None:
            try:
                self.openclaw_bridge.stop()
                self.logger.debug("Openclaw bridge stopped")
            except Exception as e:
                self.logger.warning(f"Error stopping Openclaw bridge: {e}")
            self.openclaw_bridge = None

        if self._mcp_managers:
            try:
                loop = self._mcp_loop
                t = self._mcp_thread

                if loop is not None and not loop.is_closed():
                    # 1) Perform async cleanup for all MCP managers
                    async def _cleanup_all():
                        for mgr in self._mcp_managers.values():
                            await mgr.cleanup()

                    fut = asyncio.run_coroutine_threadsafe(_cleanup_all(), loop)
                    fut.result()

                    # 2) Stop the loop
                    if loop.is_running():
                        loop.call_soon_threadsafe(loop.stop)

                    # 3) Wait for the thread to exit run_forever
                    if t is not None and t.is_alive():
                        t.join(timeout=5)

                    # 4) Confirm the loop has stopped before closing
                    if not loop.is_closed():
                        loop.close()

                self._mcp_managers.clear()
                self._mcp_loop = None
                self._mcp_thread = None

            except Exception as e:
                self.logger.warning(f"Error cleaning up MCP: {e}")


        # # Clean up MCP connections
        # if self.mcp_manager:
        #     try:
        #         import asyncio
        #         asyncio.run(self.mcp_manager.cleanup())
        #         self.logger.debug("MCP connections cleaned up")
        #     except Exception as e:
        #         self.logger.warning(f"Error cleaning up MCP: {e}")

        if self.session:
            # Check if this is a DockerSession configured to keep the container
            should_keep_session = False
            if isinstance(self.session, DockerSession):
                if not self.session.config.auto_remove:
                    should_keep_session = True
                    self.logger.info("Keeping Docker session and container for reuse (auto_remove=False)")
            
            if not should_keep_session:
                try:
                    self.session.close()
                    self.logger.debug("Session closed")
                except Exception as e:
                    self.logger.warning(f"Error closing session: {e}")
            else:
                # Only mark as closed, but don't actually close the session (container keeps running)
                self.logger.debug("Session marked as closed but container kept running")

    def execute_parallel_tasks(self, tasks: List[Callable], max_workers: int = 3) -> List[Any]:
            """General-purpose parallel task executor

            Args:
                tasks: Each element should be a callable object.
                    If the function requires arguments, wrap it with functools.partial.
                    Example: [partial(exp1.run, task="A"), partial(exp2.run, task="B")]
                max_workers: Maximum number of parallel worker threads

            Returns:
                List[Any]: A list of results in the same order as the input tasks.
                        If a task raises an exception, the corresponding position in the results list will contain that Exception object.
            """
            self.logger.info(f"Starting parallel execution of {len(tasks)} tasks with {max_workers} workers.")
            
            results = [None] * len(tasks)
            
            # Check if parallel resource allocation is enabled
            session_config = self.config.session.get("local", {})
            parallel_config = session_config.get("parallel", {})
            parallel_enabled = parallel_config.get("enabled", False)
            
            # Check if split_workspace_for_exp is enabled
            split_workspace = parallel_config.get("split_workspace_for_exp", False)
            
            # Wrap task functions to set parallel index and independent workspace
            def wrap_task(task_func, parallel_index):
                def wrapped():
                    try:
                        # If parallel resource allocation is enabled, set the session's parallel index
                        if parallel_enabled and self.session is not None:
                            from evomaster.agent.session.local import LocalSession
                            if isinstance(self.session, LocalSession):
                                self.session.set_parallel_index(parallel_index)
                                self.logger.debug(f"Set parallel index: {parallel_index}")
                                
                                # If split_workspace_for_exp is enabled, create an independent workspace for the current exp
                                if split_workspace:
                                    import os
                                    main_workspace = self.session.config.workspace_path
                                    exp_workspace = os.path.join(main_workspace, f"exp_{parallel_index}")
                                    # Create exp workspace via env (with symlinks)
                                    self.session._env.setup_exp_workspace(exp_workspace)
                                    # Set thread-local workspace path
                                    self.session.set_workspace_path(exp_workspace)
                                    self.logger.info(
                                        f"Exp {parallel_index} using independent workspace: {exp_workspace}"
                                    )
                        return task_func()
                    finally:
                        # Clean up thread-local state
                        if parallel_enabled and self.session is not None:
                            from evomaster.agent.session.local import LocalSession
                            if isinstance(self.session, LocalSession):
                                self.session.set_parallel_index(None)
                                if split_workspace:
                                    self.session.set_workspace_path(None)
                return wrapped
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks, build future-to-index mapping to preserve return order
                wrapped_tasks = [wrap_task(task, i) for i, task in enumerate(tasks)]
                future_to_index = {executor.submit(wrapped_task): i for i, wrapped_task in enumerate(wrapped_tasks)}

                # Process completed tasks
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        # Get the return value
                        result = future.result()
                        results[index] = result
                    except Exception as exc:
                        self.logger.error(f"Task {index} generated an exception: {exc}")
                        # Store exception object as result to avoid interrupting other tasks
                        results[index] = exc

            self.logger.info("Parallel execution completed.")
            return results