"""EvoMaster Env base class.

Env is the environment component, responsible for managing execution environments and job scheduling.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession, SessionConfig


class EnvConfig(BaseModel):
    """Base Env configuration."""
    name: str = Field(default="default_env", description="Environment name")
    session_config: SessionConfig | None = Field(default=None, description="Session configuration")


class BaseEnv(ABC):
    """Abstract base class for Env.

    Defines the standard interface for the environment component:
    - Session management
    - Job execution
    - Resource management
    """

    def __init__(self, config: EnvConfig | None = None):
        """Initialize Env.

        Args:
            config: Env configuration.
        """
        self.config = config or EnvConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        """Whether the environment is ready."""
        return self._is_ready

    @abstractmethod
    def setup(self) -> None:
        """Initialize the environment."""
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Clean up environment resources."""
        pass

    @abstractmethod
    def get_session(self) -> BaseSession:
        """Get a Session for executing commands.

        Returns:
            BaseSession instance.
        """
        pass

    @abstractmethod
    def submit_job(
        self,
        command: str,
        job_type: str = "debug",
        **kwargs: Any,
    ) -> str:
        """Submit a job to the environment.

        Args:
            command: Command to execute.
            job_type: Job type ("debug" or "train").
            **kwargs: Additional arguments.

        Returns:
            Job ID.
        """
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Query job status.

        Args:
            job_id: Job ID.

        Returns:
            Status information dictionary.
        """
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> None:
        """Cancel a job.

        Args:
            job_id: Job ID.
        """
        pass

    def __enter__(self) -> BaseEnv:
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.teardown()
