"""EvoMaster Agent Session module.

A Session serves as the medium for an Agent to interact with the cluster Env.
"""

from .base import BaseSession, SessionConfig
from .docker import DockerSession, DockerSessionConfig
from .local import LocalSession, LocalSessionConfig

__all__ = [
    "BaseSession",
    "SessionConfig",
    "DockerSession",
    "DockerSessionConfig",
    "LocalSession",
    "LocalSessionConfig",
]

