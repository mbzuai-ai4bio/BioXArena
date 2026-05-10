"""EvoMaster Session base class.

A Session serves as the medium for an Agent to interact with the cluster Env,
providing basic capabilities such as command execution and file operations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class SessionConfig(BaseModel):
    """Base configuration for a Session."""
    timeout: int = Field(default=300, description="Default execution timeout in seconds")
    workspace_path: str = Field(default="/workspace", description="Workspace path")


class BaseSession(ABC):
    """Abstract base class for Session.

    Defines the standard interface for Agent-environment interaction:
    - Command execution
    - File upload/download
    - Session lifecycle management
    """

    def __init__(self, config: SessionConfig | None = None):
        self.config = config or SessionConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """Whether the session is open."""
        return self._is_open

    @abstractmethod
    def open(self) -> None:
        """Open the session and establish a connection to the environment."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the session and release resources."""
        pass

    @abstractmethod
    def exec_bash(
        self,
        command: str,
        timeout: int | None = None,
        is_input: bool = False,
    ) -> dict[str, Any]:
        """Execute a Bash command.

        Args:
            command: The command to execute.
            timeout: Timeout in seconds; None uses the default value.
            is_input: Whether this is input sent to a running process.

        Returns:
            A result dictionary containing:
            - stdout: Standard output
            - stderr: Standard error
            - exit_code: Exit code
            - working_dir: Current working directory
            - Other enviroinformation
        """
        pass

    @abstractmethod
    def upload(self, local_path: str, remote_path: str) -> None:
        """Upload a file to the remote environment.

        Args:
            local_path: Local file path.
            remote_path: Remote file path.
        """
        pass

    @abstractmethod
    def download(self, remote_path: str, timeout: int | None = None) -> bytes:
        """Download a file from the remote environment.

        Args:
            remote_path: Remote file path.
            timeout: Timeout in seconds.

        Returns:
            File content as bytes.
        """
        pass

    def read_file(self, remote_path: str, encoding: str = "utf-8") -> str:
        """Read remote file content as text.

        Args:
            remote_path: Remote file path.
            encoding: File encoding.

        Returns:
            File content as a string.
        """
        content = self.download(remote_path)
        return content.decode(encoding)

    def write_file(self, remote_path: str, content: str, encoding: str = "utf-8") -> None:
        """Write content to a remote file.

        Args:
            remote_path: Remote file path.
            content: File content.
            encoding: File encoding.
        """
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(content.encode(encoding))
            temp_path = f.name
        
        try:
            self.upload(temp_path, remote_path)
        finally:
            os.unlink(temp_path)

    def path_exists(self, remote_path: str) -> bool:
        """Check whether a remote path exists.

        Args:
            remote_path: Remote path.

        Returns:
            True if the path exists, False otherwise.
        """
        result = self.exec_bash(f'test -e "{remote_path}" && echo "exists" || echo "not_exists"')
        stdout = result.get("stdout", "").strip()
        # Exact match to avoid false positives (e.g., "exists" in "not_exists")
        return stdout == "exists"

    def is_file(self, remote_path: str) -> bool:
        """Check whether a remote path is a file.

        Args:
            remote_path: Remote path.

        Returns:
            True if the path is a file, False otherwise.
        """
        result = self.exec_bash(f'test -f "{remote_path}" && echo "file" || echo "not_file"')
        stdout = result.get("stdout", "").strip()
        # Exact match to avoid false positives (e.g., "file" in "not_file")
        return stdout == "file"

    def is_directory(self, remote_path: str) -> bool:
        """Check whether a remote path is a directory.

        Args:
            remote_path: Remote path.

        Returns:
            True if the path is a directory, False otherwise.
        """
        result = self.exec_bash(f'test -d "{remote_path}" && echo "dir" || echo "not_dir"')
        stdout = result.get("stdout", "").strip()
        # Exact match to avoid false positives (e.g., "dir" in "not_dir")
        return stdout == "dir"

    def __enter__(self) -> BaseSession:
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

