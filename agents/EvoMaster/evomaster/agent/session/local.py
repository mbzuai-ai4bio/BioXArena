"""EvoMaster Local Session implementation.

Executes commands directly on the local machine without containers.
"""

from __future__ import annotations

import threading
from typing import Any

from pydantic import Field

from evomaster.env.local import LocalEnv, LocalEnvConfig

from .base import BaseSession, SessionConfig


class LocalSessionConfig(SessionConfig):
    """Local Session configuration."""
    encoding: str = Field(default="utf-8", description="File encoding")
    symlinks: dict[str, str] = Field(
        default_factory=dict,
        description="Symlink configuration, format: {source_directory_path: target_path_within_workspace}"
    )
    config_dir: str | None = Field(
        default=None,
        description="Directory where config files reside, used for resolving relative paths in symlinks"
    )
    gpu_devices: str | list[str] | None = Field(
        default=None,
        description="GPU devices, e.g. '2' or ['0', '1']; None means no GPU restriction"
    )
    cpu_devices: str | list[int] | None = Field(
        default=None,
        description="CPU devices, e.g. '0-15' or [0, 1, 2, 3]; None means no CPU restriction"
    )
    parallel: dict[str, Any] | None = Field(
        default=None,
        description="Parallel execution configuration, containing 'enabled' and 'max_parallel' fields"
    )


class LocalSession(BaseSession):
    """Local Session implementation.

    Executes bash commands directly on the local machine without containers.
    Internally uses LocalEnv for underlying operations.
    """

    # Thread-local storage for tracking parallel index per thread
    _thread_local = threading.local()

    def __init__(self, config: LocalSessionConfig | None = None):
        """Initialize the local session.

        Args:
            config: Local session configuration.
        """
        super().__init__(config)
        self.config: LocalSessionConfig = config or LocalSessionConfig()
        # Create LocalEnv instance
        env_config = LocalEnvConfig(session_config=self.config)
        self._env = LocalEnv(env_config)

    def set_parallel_index(self, parallel_index: int | None) -> None:
        """Set the parallel index for the current thread.

        Args:
            parallel_index: Parallel index (starting from 0); None means no parallel resource allocation.
        """
        self._thread_local.parallel_index = parallel_index
    
    def get_parallel_index(self) -> int | None:
        """Get the parallel index for the current thread.

        Returns:
            Parallel index, or None if not set.
        """
        return getattr(self._thread_local, 'parallel_index', None)
    
    def set_workspace_path(self, workspace_path: str | None) -> None:
        """Set the workspace path for the current thread (used by split_workspace_for_exp).

        Args:
            workspace_path: Workspace path; None means use the default workspace.
        """
        self._thread_local.workspace_path = workspace_path
    
    def get_workspace_path(self) -> str | None:
        """Get the workspace path for the current thread.

        Returns:
            Workspace path, or None if not set (uses the default workspace).
        """
        return getattr(self._thread_local, 'workspace_path', None)
        
    def open(self) -> None:
        """Open the local session."""
        if self._is_open:
            self.logger.warning("Session already open")
            return
        
        # Use LocalEnv to set up the environment
        if not self._env.is_ready:
            self._env.setup()
        
        self._is_open = True
        self.logger.info("Local session opened")

    def close(self) -> None:
        """Close the local session."""
        if not self._is_open:
            return
        
        # Use LocalEnv to clean up the environment
        if self._env.is_ready:
            self._env.teardown()
        
        self._is_open = False
        self.logger.info("Session closed")

    def exec_bash(
        self,
        command: str,
        timeout: int | None = None,
        is_input: bool = False,
        parallel_index: int | None = None,
    ) -> dict[str, Any]:
        """Execute a bash command.

        Provides local command execution capability.

        Args:
            command: The command to execute.
            timeout: Timeout in seconds.
            is_input: Whether this is input sent to a running process (not supported locally).
            parallel_index: Parallel index (optional; if not provided, obtained from thread-local storage).
        """
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        timeout = timeout or self.config.timeout
        command = command.strip()
        
        # Local environment does not support is_input mode
        if is_input:
            return {
                "stdout": "ERROR: Local session does not support is_input mode.",
                "stderr": "",
                "exit_code": 1,
            }
        
        # Get parallel index (prefer argument; otherwise retrieve from thread-local storage)
        if parallel_index is None:
            parallel_index = self.get_parallel_index()
        
        # Get thread-local workspace path (used by split_workspace_for_exp)
        workspace_override = self.get_workspace_path()
        
        # Execute command using LocalEnv
        result = self._env.local_exec(
            command, timeout=timeout,
            workdir=workspace_override,
            parallel_index=parallel_index,
        )
        
        # Get working directory (prefer thread-local workspace path)
        workspace = workspace_override or self.config.workspace_path
        
        # Build result
        return {
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "exit_code": result.get("exit_code", -1),
            "working_dir": workspace,
            "output": result.get("output", ""),
        }

    def upload(self, local_path: str, remote_path: str) -> None:
        """Upload a file to the local environment."""
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        self._env.upload_file(local_path, remote_path)

    def read_file(self, remote_path: str, encoding: str = "utf-8") -> str:
        """Read remote file content as text."""
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.read_file_content(remote_path, encoding)
    
    def write_file(self, remote_path: str, content: str, encoding: str = "utf-8") -> None:
        """Write content to a remote file."""
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        self._env.write_file_content(remote_path, content, encoding)
    
    def download(self, remote_path: str, timeout: int | None = None) -> bytes:
        """Download a file from the local environment."""
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.download_file(remote_path, timeout)
    
    def path_exists(self, remote_path: str) -> bool:
        """Check whether a remote path exists."""
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.path_exists(remote_path)
    
    def is_file(self, remote_path: str) -> bool:
        """Check whether a remote path is a file."""
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.is_file(remote_path)
    
    def is_directory(self, remote_path: str) -> bool:
        """Check whether a remote path is a directory."""
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.is_directory(remote_path)
