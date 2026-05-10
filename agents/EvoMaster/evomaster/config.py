"""EvoMaster Configuration Management

Provides unified configuration loading and management functionality.
All configuration classes inherit from BaseConfig.
Supports loading environment variables from .env and substituting ${VAR}
with values from os.environ in configuration files.
"""

from __future__ import annotations

import os
import re
from abc import ABC
from pathlib import Path
from typing import Any
import warnings

import yaml
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

# Match ${VAR_NAME}, where VAR_NAME consists of letters, digits, and underscores
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)\}")


def _substitute_env(value: Any) -> Any:
    """Recursively substitute ${VAR} in configuration values with os.environ.get("VAR", "").

    Args:
        value: The configuration value to process. Can be a string, dict, list, or other type.

    Returns:
        The value with all ${VAR} patterns replaced by their environment variable values.
    """
    if isinstance(value, str):
        return _ENV_PATTERN.sub(
            lambda m: os.environ.get(m.group(1), ""),
            value,
        )
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(item) for item in value]
    return value


# ============================================
# Base Configuration Class
# ============================================

class BaseConfig(BaseModel, ABC):
    """Configuration base class.

    All configuration classes should inherit from this class.
    Uses Pydantic for validation.
    """

    class Config:
        extra = "allow"  # Allow extra fields
        arbitrary_types_allowed = True


# ============================================
# Env Configuration
# ============================================

class ClusterPoolConfig(BaseConfig):
    """Cluster resource pool configuration."""
    type: str = Field(description="Resource type: cpu, gpu")
    max_concurrent: int = Field(default=5, description="Maximum concurrency")
    resource_limits: dict[str, Any] = Field(default_factory=dict, description="Resource limits")


class ClusterConfig(BaseConfig):
    """Cluster configuration."""
    debug_pool: ClusterPoolConfig = Field(
        default_factory=lambda: ClusterPoolConfig(type="cpu"), description="Debug pool configuration"
    )
    train_pool: ClusterPoolConfig = Field(
        default_factory=lambda: ClusterPoolConfig(type="cpu"), description="Training pool configuration"
    )


class DockerEnvConfig(BaseConfig):
    """Docker environment configuration."""
    base_image: str = Field(default="evomaster/base:latest", description="Base image")
    registry: str = Field(default="docker.io", description="Image registry")
    pull_policy: str = Field(default="if_not_present", description="Pull policy")


class SchedulerConfig(BaseConfig):
    """Scheduler configuration."""
    type: str = Field(default="local", description="Scheduler type: local, slurm, kubernetes")
    queue_timeout: int = Field(default=3600, description="Queue timeout in seconds")
    retry_failed: bool = Field(default=True, description="Whether to retry failed tasks")
    max_retries: int = Field(default=3, description="Maximum retry attempts")


# class EnvConfig(BaseConfig):
#     """Environment configuration (cluster / Docker / scheduler).
#     Bohrium authentication (BOHRIUM_ACCESS_KEY, BOHRIUM_PROJECT_ID, etc.) is provided
#     by .env, injected into executor/storage by the MCP calculation path adaptor."""
#     cluster: ClusterConfig = Field(description="Cluster configuration")
#     docker: DockerEnvConfig = Field(description="Docker configuration")
#     scheduler: SchedulerConfig = Field(description="Scheduler configuration")

class ToolConfig(BaseConfig):
    """Tools configuration."""
    builtin: list[str] = Field(default_factory=list, description="Builtin Tools")
    mcp: list[str] = Field(default_factory=list, description="MCP Tools")

# ============================================
# Logging Configuration
# ============================================

class LoggingConfig(BaseConfig):
    """Logging configuration."""
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file: str | None = Field(default=None, description="Log file path")
    console: bool = Field(default=True, description="Whether to output to console")
    log_path: str | None = Field(default=None, description="Log file save path (saved after program execution completes)")


# ============================================
# Top-level Configuration
# ============================================

class EvoMasterConfig(BaseConfig):
    """EvoMaster top-level configuration.

    Contains configuration for all sub-modules.
    """

    # LLM configuration (stored as dict, converted to LLMConfig on demand)
    llm: dict[str, Any] = Field(default_factory=dict, description="LLM configuration")

    # Agent configuration (stored as dict, converted to AgentConfig on demand)
    agents: dict[str, Any] = Field(default_factory=dict, description="Agent configuration")

    # Session configuration (stored as dict, converted to SessionConfig on demand)
    session: dict[str, Any] = Field(default_factory=dict, description="Session configuration")

    # Env configuration
    # env: EnvConfig = Field(default_factory=EnvConfig, description="Environment configuration")

    # Tools configuration
    tools: ToolConfig = Field(default_factory=ToolConfig, description="Tools configuration")
    


    # Skills loading (for Playground: when enabled=true, loads SkillRegistry; skills_root is the skills directory)
    skills: dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False, "skills_root": "evomaster/skills"},
        description="Skills enablement and root directory",
    )

    # Logging configuration
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")

    # LLM output display configuration
    llm_output: dict[str, Any] = Field(
        default_factory=lambda: {
            "show_in_console": False,
            "log_to_file": False,
        },
        description="LLM output display configuration"
    )

    # Other configuration
    project_root: str = Field(default=".", description="Project root directory")
    workspace: str = Field(default="./workspace", description="Working directory")
    results_dir: str = Field(default="./results", description="Results save directory")
    debug: bool = Field(default=False, description="Whether to enable debug mode")


# ============================================
# Configuration Manager
# ============================================

class ConfigManager:
    """Configuration Manager.

    Loads configuration from YAML files and constructs configuration objects.
    """

    DEFAULT_CONFIG_FILE = "config.yaml"

    def __init__(self, config_dir: str | Path | None = None, config_file: str | None = None):
        """Initialize the configuration manager.

        Args:
            config_dir: Configuration file directory, defaults to configs/ under the project root.
            config_file: Configuration file name, defaults to config.yaml.
        """
        if config_dir is None:
            # Default config directory: project_root/configs
            project_root = Path(__file__).parent.parent
            config_dir = project_root / "configs"

        self.config_dir = Path(config_dir)
        self.config_file = config_file or self.DEFAULT_CONFIG_FILE
        self._config: EvoMasterConfig | None = None

    def load(self) -> EvoMasterConfig:
        """Load the configuration file.

        Attempts to load .env from the project root directory
        and substitutes ${VAR} in the configuration with environment variable values.

        Returns:
            EvoMaster configuration object.
        """
        if self._config is not None:
            return self._config

        if load_dotenv is not None:
            # Search for .env from config_dir upward (e.g., configs/mat_master -> project root)
            for parent in [self.config_dir] + list(self.config_dir.parents):
                env_file = parent / ".env"
                if env_file.exists():
                    load_dotenv(env_file)
                    break
            else:
                load_dotenv()  # Fall back to cwd and parent directories

        config_path = self.config_dir / self.config_file

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        config_dict = _substitute_env(config_dict)

        # Construct the configuration object
        self._config = EvoMasterConfig(**config_dict)
        return self._config

    @staticmethod
    def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
        """Ensure a configuration field is a dict type, providing a unified error message.

        Args:
            value: The value to validate.
            field_name: The name of the configuration field (for error messages).

        Returns:
            The value itself if it is a dict.

        Raises:
            TypeError: If the value is not a dict.
        """
        if not isinstance(value, dict):
            raise TypeError(f"Config field '{field_name}' must be a dict, got {type(value).__name__}")
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key (supports dot-separated nested keys, e.g., "agent.max_turns").
            default: Default value.

        Returns:
            The configuration value.
        """
        config = self.load()

        # Support nested keys
        keys = key.split(".")
        value = config.model_dump()
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    def get_llm_config(self, name: str | None = None) -> dict[str, Any]:
        """Get LLM configuration.

        Args:
            name: LLM configuration name. None uses the default configuration.

        Returns:
            LLM configuration dictionary.
        """
        config = self.load()

        llm_root = self._require_dict(config.llm, "llm")
        if name is None:
            name = llm_root.get("default", "openai")

        llm_config = llm_root.get(name)
        if llm_config is None:
            raise ValueError(f"LLM config '{name}' not found")

        return self._require_dict(llm_config, f"llm.{name}")

    def get_agent_config(self, name: str | None = None) -> dict[str, Any]:
        """Get Agent configuration.

        Args:
            name: Agent configuration name.

        Returns:
            Agent configuration dictionary.
        """
        config = self.load()
        agents = self._require_dict(config.agents, "agents")
        if name is None:
            raise ValueError(f"No {name} configuration found. Add {name} in agents section in config.yaml")
        if name not in agents:
            raise ValueError(f"Agent config '{name}' not found")
        return self._require_dict(agents[name], f"agents.{name}")

    def get_agents_config(self) -> dict[str, Any]:
        """Get all Agents configuration.

        Returns:
            Agents configuration dictionary.
        """
        config = self.load()
        agents = self._require_dict(config.agents, "agents")
        if not agents:
            raise ValueError("No agents configuration found. Add 'agents' section to config.yaml")
        return agents
    
    def get_agent_llm_config(self, name: str) -> dict[str, Any]:
        """Get Agent LLM configuration.

        Args:
            name: Agent name.

        Returns:
            Agent LLM configuration dictionary.
        """
        config = self.load()
        agents = self._require_dict(config.agents, "agents")
        if name not in agents:
            raise ValueError(f"Agent config '{name}' not found")
        agent_cfg = self._require_dict(agents[name], f"agents.{name}")
        if "llm" not in agent_cfg:
            warnings.warn(f"Agent '{name}' does not have LLM configuration, trying to use default LLM configuration")
            return self.get_llm_config()
        else:
            return self.get_llm_config(agent_cfg["llm"])

    def get_agent_tools_config(self, name: str) -> dict[str, Any]:
        """Get Agent Tools configuration (per-agent).

        YAML supports the following formats:
          1) No tools configured (key missing) -> {"builtin": ["*"], "mcp": ""}  (default: all builtin, no mcp)
          2) tools: "default"          -> same as above
          3) tools:                    -> {"builtin": [], "mcp": ""}  (null value = disable all tools)
          4) tools: []                 -> {"builtin": [], "mcp": ""}  (empty list = disable all tools)
          5) tools:
               builtin: ["*"]         -> all builtin
               mcp: ["*"]             -> use default mcp_config.json
          6) tools:
               builtin: ["execute_bash", "finish"]   -> only specified builtin tools
          7) tools:
               builtin: []            -> disable all builtin tools
          8) tools:
               mcp: "custom_mcp.json" -> use specified MCP config file
          9) tools:
               search: "google_search"       -> custom tool configuration (any field name)

        Args:
            name: Agent name.

        Returns:
            Normalized dict of the form {"builtin": list[str], "mcp": str, "custom": dict},
            where mcp is the MCP config file path (relative to config_dir), empty string means MCP is disabled,
            and custom contains all custom tool configurations other than builtin and mcp.
        """
        _DEFAULT = {"builtin": ["*"], "mcp": "", "custom": {}}
        _EMPTY = {"builtin": [], "mcp": "", "custom": {}}

        config = self.load()
        agents = self._require_dict(config.agents, "agents")
        if name not in agents:
            raise ValueError(f"Agent config '{name}' not found")
        agent_cfg = self._require_dict(agents[name], f"agents.{name}")

        # Key missing -> default (all builtin)
        if "tools" not in agent_cfg:
            return _DEFAULT.copy()

        raw_tools = agent_cfg["tools"]

        # tools:  (null value, YAML parses as None) -> disable all tools
        if raw_tools is None:
            return _EMPTY.copy()

        # tools: [] (empty list) -> disable all tools
        if isinstance(raw_tools, list) and len(raw_tools) == 0:
            return _EMPTY.copy()

        # tools: "default" -> default
        if isinstance(raw_tools, str):
            if raw_tools == "default":
                return _DEFAULT.copy()
            raise ValueError(
                f"Config field 'agents.{name}.tools' string value must be 'default', got '{raw_tools}'"
            )

        if not isinstance(raw_tools, dict):
            raise TypeError(
                f"Config field 'agents.{name}.tools' must be dict, 'default', [], or omitted, "
                f"got {type(raw_tools).__name__}"
            )

        # Parse builtin
        raw_builtin = raw_tools.get("builtin")
        if raw_builtin is None:
            builtin = ["*"]  # builtin not explicitly configured -> all
        elif isinstance(raw_builtin, str) and raw_builtin == "*":
            builtin = ["*"]
        elif isinstance(raw_builtin, list) and all(isinstance(s, str) for s in raw_builtin):
            builtin = raw_builtin
        else:
            raise TypeError(
                f"Config field 'agents.{name}.tools.builtin' must be list[str] or '*', "
                f"got {type(raw_builtin).__name__}"
            )

        # Parse mcp
        raw_mcp = raw_tools.get("mcp")
        if raw_mcp is None:
            mcp = ""  # mcp not explicitly configured -> disabled
        elif isinstance(raw_mcp, str):
            if raw_mcp == "*":
                mcp = "mcp_config.json"  # "*" -> default config file
            else:
                mcp = raw_mcp  # Use the specified config file path directly
        elif isinstance(raw_mcp, list):
            if len(raw_mcp) == 0:
                mcp = ""  # Empty list -> disabled
            elif raw_mcp == ["*"]:
                mcp = "mcp_config.json"  # ["*"] -> default config file
            else:
                raise ValueError(
                    f"Config field 'agents.{name}.tools.mcp' list value must be ['*'] or [], "
                    f"got {raw_mcp}"
                )
        else:
            raise TypeError(
                f"Config field 'agents.{name}.tools.mcp' must be str, ['*'], [], or omitted, "
                f"got {type(raw_mcp).__name__}"
            )

        # Extract custom tool configurations (all fields except builtin and mcp)
        custom = {k: v for k, v in raw_tools.items() if k not in ("builtin", "mcp")}

        return {"builtin": builtin, "mcp": mcp, "custom": custom}
    
    def get_agent_skills_config(self, name: str) -> dict[str, Any]:
        """Get Agent Skills configuration.

        Args:
            name: Agent name.

        Returns:
            Normalized Agent Skills configuration dictionary, of the form {"skills": list[str]}.
        """
        config = self.load()
        agents = self._require_dict(config.agents, "agents")
        if name not in agents:
            raise ValueError(f"Agent config '{name}' not found")
        agent_cfg = self._require_dict(agents[name], f"agents.{name}")
        raw_skills = agent_cfg.get("skills")
        if raw_skills is None:
            return {"skills": []}

        # Compatible with skills: "*" syntax
        if raw_skills == "*":
            raw_skills = ["*"]

        if not isinstance(raw_skills, list) or not all(isinstance(skill, str) for skill in raw_skills):
            raise TypeError(
                f"Config field 'agents.{name}.skills' must be list[str], '*' or omitted"
            )

        # '*' can only appear alone
        if "*" in raw_skills and raw_skills != ["*"]:
            raise ValueError(
                f"Config field 'agents.{name}.skills' cannot mix '*' with specific skill names"
            )
        result: dict[str, Any] = {"skills": raw_skills}
        skill_dir = agent_cfg.get("skill_dir", "./evomaster/skills_ts")
        result["skill_dir"] = str(skill_dir)
        return result


    def get_session_config(self, session_type: str = "docker") -> dict[str, Any]:
        """Get Session configuration.

        Args:
            session_type: Session type (docker, local).

        Returns:
            Session configuration dictionary.
        """
        config = self.load()
        sessions = self._require_dict(config.session, "session")
        session_config = sessions.get(session_type)
        if session_config is None:
            raise ValueError(f"Session config '{session_type}' not found")
        return self._require_dict(session_config, f"session.{session_type}")

    def get_env_config(self) -> EnvConfig:
        """Get Env configuration.

        Returns:
            Env configuration object.
        """
        config = self.load()
        return config.env

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration.

        Returns:
            Logging configuration object.
        """
        config = self.load()
        return config.logging

    def create_llm_from_config(self, name: str | None = None):
        """Create an LLM instance from configuration.

        Args:
            name: LLM configuration name.

        Returns:
            LLM instance.
        """
        from evomaster.utils import LLMConfig, create_llm

        config_dict = self.get_llm_config(name)
        llm_config = LLMConfig(**config_dict)
        return create_llm(llm_config)


# ============================================
# Global Configuration Manager
# ============================================

_config_manager: ConfigManager | None = None


def get_config_manager(config_dir: str | Path | None = None) -> ConfigManager:
    """Get the global configuration manager.

    Args:
        config_dir: Configuration directory (set on first call).

    Returns:
        Configuration manager instance.
    """
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager(config_dir)

    return _config_manager


def load_config() -> EvoMasterConfig:
    """Shortcut function: Load the configuration file.

    Returns:
        Configuration object.
    """
    return get_config_manager().load()


def get_config(key: str, default: Any = None) -> Any:
    """Shortcut function: Get a configuration value.

    Args:
        key: Configuration key.
        default: Default value.

    Returns:
        Configuration value.
    """
    return get_config_manager().get(key, default)
