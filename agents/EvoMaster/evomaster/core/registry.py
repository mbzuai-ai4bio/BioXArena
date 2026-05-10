"""Playground Registry

Provides a decorator mechanism for registering custom Playground classes for each agent.

Usage examples:
    from evomaster.core import BasePlayground, register_playground

    @register_playground("my-agent")
    class MyAgentPlayground(BasePlayground):
        pass

    # In run.py
    from evomaster.core import get_playground_class
    playground = get_playground_class("my-agent", config_dir=...)
"""

import logging
from typing import Dict, Type, Optional
from pathlib import Path

# Global registry: stores agent_name -> Playground class mapping
_PLAYGROUND_REGISTRY: Dict[str, Type] = {}

logger = logging.getLogger(__name__)


def register_playground(agent_name: str):
    """Decorator: Register a Playground class.

    Usage example:
        @register_playground("agent-builder")
        class AgentBuilderPlayground(BasePlayground):
            pass

    Args:
        agent_name: Agent name (e.g., "minimal", "agent-builder").
                   Must match the playground directory name (use hyphens).

    Returns:
        Decorator function.
    """
    def decorator(cls):
        if agent_name in _PLAYGROUND_REGISTRY:
            logger.warning(
                f"Playground '{agent_name}' is already registered, will be overridden: "
                f"{_PLAYGROUND_REGISTRY[agent_name].__name__} -> {cls.__name__}"
            )

        _PLAYGROUND_REGISTRY[agent_name] = cls
        logger.debug(f"Registered playground: {agent_name} -> {cls.__name__}")
        return cls

    return decorator


def get_playground_class(agent_name: str, config_dir: Optional[Path] = None, config_path: Optional[Path] = None):
    """Get a registered Playground class instance.

    If the agent has a registered custom Playground class, uses the custom class;
    otherwise falls back to BasePlayground.

    Args:
        agent_name: Agent name.
        config_dir: Configuration directory path (optional).
        config_path: Full configuration file path (recommended).

    Returns:
        Playground instance.

    Raises:
        ImportError: If BasePlayground cannot be imported (internal error).
    """
    from .playground import BasePlayground

    playground_class = _PLAYGROUND_REGISTRY.get(agent_name)

    if playground_class:
        # Use registered custom class
        logger.info(f"Using custom Playground: {agent_name} -> {playground_class.__name__}")
        return playground_class(config_dir=config_dir, config_path=config_path)
    else:
        # Fall back to BasePlayground
        logger.info(f"Using BasePlayground for agent '{agent_name}' (no custom implementation registered)")
        return BasePlayground(config_dir=config_dir, config_path=config_path)


def list_registered_playgrounds():
    """List all registered Playgrounds.

    Returns:
        List of registered agent names.
    """
    return list(_PLAYGROUND_REGISTRY.keys())


def get_registry_info():
    """Get detailed information about the registry.

    Returns:
        Dictionary of the form {agent_name: class_name}.
    """
    return {
        agent_name: cls.__name__
        for agent_name, cls in _PLAYGROUND_REGISTRY.items()
    }
