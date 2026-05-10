"""EvoMaster Env module.

Env is EvoMaster's environment component, responsible for:
- Executable sandboxes (Docker)
- Cluster scheduling (k8s, ray, skypilot)
- Resource management
- Bohrium authentication (MCP calculation storage/executor, see .bohrium)
"""

from .base import BaseEnv, EnvConfig
from .local import LocalEnv, LocalEnvConfig
from .docker import DockerEnv, DockerEnvConfig
from .bohrium import (
    get_bohrium_credentials,
    get_bohrium_storage_config,
    inject_bohrium_executor,
)

# Resolve Pydantic circular dependency: rebuild EnvConfig models
# Ensure SessionConfig subclasses are fully defined
def _rebuild_env_configs():
    """Lazily rebuild EnvConfig models to resolve circular dependencies."""
    try:
        # Ensure SessionConfig subclasses are imported
        from evomaster.agent.session import DockerSessionConfig, LocalSessionConfig
        # Rebuild EnvConfig models
        DockerEnvConfig.model_rebuild()
        LocalEnvConfig.model_rebuild()
    except Exception:
        # If rebuild fails, ignore (may already have been rebuilt or not yet imported)
        pass

# Execute rebuild lazily to ensure all modules are loaded
_rebuild_env_configs()

__all__ = [
    "BaseEnv",
    "EnvConfig",
    "LocalEnv",
    "LocalEnvConfig",
    "DockerEnv",
    "DockerEnvConfig",
    "get_bohrium_credentials",
    "get_bohrium_storage_config",
    "inject_bohrium_executor",
]

