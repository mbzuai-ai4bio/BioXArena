"""EvoMaster Docker Session implementation.

Docker container-based Session implementation providing an isolated execution environment.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import Field

from evomaster.env.docker import DockerEnv, DockerEnvConfig, PS1_PATTERN, BashMetadata

from .base import BaseSession, SessionConfig


class DockerSessionConfig(SessionConfig):
    """Docker Session configuration."""
    image: str = Field(default="python:3.11-slim", description="Docker image")
    container_name: str | None = Field(default=None, description="Container name; None for auto-generated")
    working_dir: str = Field(default="/workspace", description="Working directory")
    memory_limit: str = Field(default="4g", description="Memory limit")
    cpu_limit: float = Field(default=2.0, description="CPU limit")
    gpu_devices: str | list[str] | None = Field(default=None, description="GPU devices, e.g. 'all' or ['0', '1']; None means no GPU")
    network_mode: str = Field(default="bridge", description="Network mode")
    volumes: dict[str, str] = Field(default_factory=dict, description="Mount volumes {host_path: container_path}")
    env_vars: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    auto_remove: bool = Field(default=True, description="Automatically remove container when it exits")
    use_existing_container: str | None = Field(default=None, description="Use an existing container by name; if set, no new container is created")


class DockerSession(BaseSession):
    """Docker Session implementation.

    Uses a Docker container to provide an isolated execution environment.
    Internally uses DockerEnv for underlying operations.
    """

    def __init__(self, config: DockerSessionConfig | None = None):
        """Initialize the Docker session.

        Args:
            config: Docker session configuration.
        """
        super().__init__(config)
        self.config: DockerSessionConfig = config or DockerSessionConfig()
        # Create DockerEnv instance
        env_config = DockerEnvConfig(session_config=self.config)
        self._env = DockerEnv(env_config)
        # Session state management
        self._last_ps1_count: int = 0
        self._prev_command_status: Literal["completed", "timeout"] = "completed"
        self._prev_command_output: str = ""
        
    def open(self) -> None:
        """Start the Docker container."""
        if self._is_open:
            self.logger.warning("Session already open")
            return
        
        # Use DockerEnv to set up the environment
        if not self._env.is_ready:
            self._env.setup()
        
        # Get initial PS1 count
        logs = self._env.get_tmux_logs()
        matches = list(PS1_PATTERN.finditer(logs))
        self._last_ps1_count = len(matches)
        
        self._is_open = True
        self.logger.info("Docker session opened")

    def close(self) -> None:
        """Close the session.

        If auto_remove=True, stops and removes the container.
        If auto_remove=False, only marks the session as closed; the container keeps running for reuse.
        """
        if not self._is_open:
            return
        
        # Use DockerEnv to clean up the environment
        if self._env.is_ready:
            self._env.teardown()
        
        self._is_open = False
        self.logger.info("Session closed")

    def exec_bash(
        self,
        command: str,
        timeout: int | None = None,
        is_input: bool = False,
    ) -> dict[str, Any]:
        """Execute a bash command via tmux.

        Provides a persistent bash environment with preserved environment variables,
        working directory, and other state.
        """
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        timeout = timeout or self.config.timeout
        command = command.strip()
        
        # Handle input mode
        if is_input:
            if self._prev_command_status == "completed":
                if command == "":
                    return {
                        "stdout": "ERROR: No previous running command to retrieve logs from.",
                        "stderr": "",
                        "exit_code": 1,
                    }
                else:
                    return {
                        "stdout": "ERROR: No previous running command to interact with.",
                        "stderr": "",
                        "exit_code": 1,
                    }
            
            # Send control signal or input
            if command.startswith("C-") and len(command) == 3:
                self._env.tmux_send_keys(command, enter=False)
            elif command == "":
                pass  # Only retrieve logs
            else:
                self._env.tmux_send_keys(command, enter=True)
        else:
            # Normal command execution
            if self._prev_command_status != "completed" and command != "":
                return {
                    "stdout": f"[Previous command is still running. Use is_input=true to interact.]",
                    "stderr": "",
                    "exit_code": 1,
                }
            
            if command != "":
                self._env.tmux_send_keys(command, enter=True)
        
        # Wait for command completion
        start_time = time.time()
        poll_interval = 0.5
        self._prev_command_status = "timeout"
        
        while time.time() - start_time < timeout:
            logs = self._env.get_tmux_logs()
            matches = list(PS1_PATTERN.finditer(logs))
            ps1_count = len(matches)
            
            if ps1_count > self._last_ps1_count:
                # Command completed
                self._prev_command_status = "completed"
                break
            
            time.sleep(poll_interval)
        
        # Parse output
        logs = self._env.get_tmux_logs()
        matches = list(PS1_PATTERN.finditer(logs))
        ps1_count = len(matches)
        
        output = ""
        exit_code = -1
        working_dir = ""
        
        if ps1_count > self._last_ps1_count:
            # Extract output from the last command
            if self._last_ps1_count > 0:
                prev_match = matches[self._last_ps1_count - 1]
                curr_match = matches[ps1_count - 1]
                output = logs[prev_match.end():curr_match.start()]
            else:
                curr_match = matches[ps1_count - 1]
                output = logs[:curr_match.start()]
            
            # Parse metadata
            try:
                metadata = BashMetadata.from_json(matches[-1].group(1))
                exit_code = metadata.exit_code
                working_dir = metadata.working_dir
            except Exception:
                pass
            
            self._last_ps1_count = ps1_count
        else:
            # Timeout: get partial output
            if self._last_ps1_count > 0 and matches:
                prev_match = matches[self._last_ps1_count - 1]
                output = logs[prev_match.end():]
        
        # Clean output
        output = output.strip()
        if command and output.startswith(command):
            output = output[len(command):].strip()
        
        # Build result
        result = {
            "stdout": output,
            "stderr": "",
            "exit_code": exit_code,
            "working_dir": working_dir,
            "output": output,
        }
        
        if self._prev_command_status == "timeout":
            result["stdout"] += f"\n[Command timed out after {timeout}s]"
            result["exit_code"] = -1
        
        return result

    def upload(self, local_path: str, remote_path: str) -> None:
        """Upload a file to the container.

        If the target path is within a mounted volume, copies the file directly on the host.
        """
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        self._env.upload_file(local_path, remote_path)

    def read_file(self, remote_path: str, encoding: str = "utf-8") -> str:
        """Read remote file content as text.

        If the path is within a mounted volume, reads directly on the host.
        """
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.read_file_content(remote_path, encoding)
    
    def write_file(self, remote_path: str, content: str, encoding: str = "utf-8") -> None:
        """Write content to a remote file.

        If the path is within a mounted volume, writes directly on the host.
        """
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        self._env.write_file_content(remote_path, content, encoding)
    
    def download(self, remote_path: str, timeout: int | None = None) -> bytes:
        """Download a file from the container.

        If the path is within a mounted volume, reads directly on the host.
        """
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.download_file(remote_path, timeout)
    
    def path_exists(self, remote_path: str) -> bool:
        """Check whether a remote path exists.

        If the path is within a mounted volume, checks directly on the host.
        """
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.path_exists(remote_path)
    
    def is_file(self, remote_path: str) -> bool:
        """Check whether a remote path is a file.

        If the path is within a mounted volume, checks directly on the host.
        """
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.is_file(remote_path)
    
    def is_directory(self, remote_path: str) -> bool:
        """Check whether a remote path is a directory.

        If the path is within a mounted volume, checks directly on the host.
        """
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.is_directory(remote_path)
