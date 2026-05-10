"""Docker environment implementation.

Provides the low-level operations interface for Docker containers.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from .base import BaseEnv, EnvConfig
from evomaster.agent.session.base import SessionConfig


class DockerEnvConfig(EnvConfig):
    """Docker environment configuration."""
    session_config: SessionConfig = Field(
        ...,
        description="Session configuration"
    )


# PS1 Prompt configuration used for parsing bash output
PS1_BEGIN = "\n===PS1JSONBEGIN===\n"
PS1_END = "\n===PS1JSONEND===\n"
PS1_PATTERN = re.compile(
    f"{PS1_BEGIN.strip()}(.*?){PS1_END.strip()}",
    re.DOTALL | re.MULTILINE,
)


class BashMetadata:
    """Bash execution metadata."""
    
    def __init__(
        self,
        exit_code: int = -1,
        working_dir: str = "",
        pid: int = -1,
    ):
        self.exit_code = exit_code
        self.working_dir = working_dir
        self.pid = pid

    @classmethod
    def to_ps1_prompt(cls) -> str:
        """Generate the PS1 prompt configuration."""
        prompt = "===PS1JSONBEGIN==="
        json_str = json.dumps({
            "pid": "$!",
            "exit_code": "$?",
            "working_dir": r"$(pwd)",
        }, indent=2)
        prompt += json_str.replace('"', r'\"')
        prompt += "===PS1JSONEND===\n"
        return prompt

    @classmethod
    def from_json(cls, json_str: str) -> BashMetadata:
        """Parse metadata from a JSON string."""
        try:
            data = json.loads(json_str)
            return cls(
                exit_code=int(data.get("exit_code", -1)),
                working_dir=data.get("working_dir", ""),
                pid=int(data.get("pid", -1)) if data.get("pid") else -1,
            )
        except (json.JSONDecodeError, ValueError):
            return cls()


class DockerEnv(BaseEnv):
    """Docker environment implementation.

    Provides the low-level operations interface for Docker containers:
    - Container lifecycle management
    - Command execution
    - File operations
    - Tmux session management
    """

    def __init__(self, config: DockerEnvConfig | None = None):
        """Initialize the Docker environment.

        Args:
            config: Docker environment configuration.
        """
        if config is None:
            raise ValueError("DockerEnv requires DockerEnvConfig with session_config")
        super().__init__(config)
        self.config: DockerEnvConfig = config
        self._container_id: str | None = None
        self._tmux_session: str | None = None
        self._tmux_log_path: str | None = None

    def setup(self) -> None:
        """Initialize the Docker environment."""
        if self._is_ready:
            self.logger.warning("Environment already setup")
            return

        self.logger.info("Setting up Docker environment")
        self._create_or_get_container()
        self._setup_tmux()
        self._is_ready = True
        self.logger.info("Docker environment setup complete")

    def teardown(self) -> None:
        """Clean up Docker environment resources."""
        if not self._is_ready:
            return

        self.logger.info("Tearing down Docker environment")

        if self._container_id:
            session_config = self.config.session_config
            if session_config.auto_remove:
                # Auto-remove mode: stop and remove the container
                self.logger.info(f"Stopping and removing container: {self._container_id[:12]}")
                try:
                    subprocess.run(
                        ["docker", "stop", self._container_id],
                        capture_output=True,
                        timeout=30,
                    )
                    subprocess.run(
                        ["docker", "rm", "-f", self._container_id],
                        capture_output=True,
                        timeout=30,
                    )
                except Exception as e:
                    self.logger.warning(f"Error stopping/removing container: {e}")
                self._container_id = None
            else:
                # Keep-container mode: only mark as closed; container keeps running
                self.logger.info(f"Environment closed (container {self._container_id[:12]} kept running for reuse)")

        self._is_ready = False
        self.logger.info("Docker environment teardown complete")

    def get_session(self) -> Any:
        """Get a Session (DockerEnv does not provide Sessions directly; managed by the caller)."""
        raise NotImplementedError("DockerEnv does not provide session directly")

    def submit_job(
        self,
        command: str,
        job_type: str = "debug",
        **kwargs: Any,
    ) -> str:
        """Submit a job (DockerEnv does not support job scheduling directly)."""
        raise NotImplementedError("DockerEnv does not support job submission")

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Query job status (DockerEnv does not support job scheduling directly)."""
        raise NotImplementedError("DockerEnv does not support job status")

    def cancel_job(self, job_id: str) -> None:
        """Cancel a job (DockerEnv does not support job scheduling directly)."""
        raise NotImplementedError("DockerEnv does not support job cancellation")

    @property
    def container_id(self) -> str | None:
        """Get the container ID."""
        return self._container_id

    def _create_or_get_container(self) -> None:
        """Create or obtain a Docker container."""
        session_config = self.config.session_config

        # If container ID already exists (previously opened then closed), check container status
        if self._container_id:
            # Check if the container is still running
            result = subprocess.run(
                ["docker", "ps", "--filter", f"id={self._container_id}", "--format", "{{.ID}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Container is running; reuse it
                self.logger.info(f"Reusing existing running container: {self._container_id[:12]}")
                return
            else:
                # Container is stopped; try to start it
                self.logger.info(f"Starting existing stopped container: {self._container_id[:12]}")
                try:
                    result = subprocess.run(
                        ["docker", "start", self._container_id],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        # Wait for the container to fully start
                        time.sleep(1)
                        return
                    else:
                        self.logger.warning(f"Failed to start container: {result.stderr}")
                        # Continue to create a new container
                        self._container_id = None
                except Exception as e:
                    self.logger.warning(f"Error starting container: {e}")
                    # Continue to create a new container
                    self._container_id = None

        # If configured to use an existing container
        if session_config.use_existing_container:
            self.logger.info(f"Using existing container: {session_config.use_existing_container}")
            # Check if the container exists and is running
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={session_config.use_existing_container}", "--format", "{{.ID}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                self._container_id = result.stdout.strip()
                self.logger.info(f"Found running container: {self._container_id[:12]}")
            else:
                # Try to find a stopped container
                result = subprocess.run(
                    ["docker", "ps", "-a", "--filter", f"name={session_config.use_existing_container}", "--format", "{{.ID}}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    container_id = result.stdout.strip()
                    # Start the container
                    subprocess.run(
                        ["docker", "start", container_id],
                        capture_output=True,
                        timeout=30,
                    )
                    self._container_id = container_id
                    self.logger.info(f"Started existing container: {self._container_id[:12]}")
                else:
                    raise RuntimeError(f"Container '{session_config.use_existing_container}' not found")
            return

        self.logger.info(f"Starting Docker container with image: {session_config.image}")

        # Container name
        container_name = session_config.container_name or f"evomaster-{os.getpid()}-{int(time.time())}"

        # If a container name is specified, check if the container already exists
        if session_config.container_name:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.ID}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                container_id = result.stdout.strip()
                # Container already exists; check if it is running
                result_running = subprocess.run(
                    ["docker", "ps", "--filter", f"id={container_id}", "--format", "{{.ID}}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result_running.returncode == 0 and result_running.stdout.strip():
                    # Container is running; reuse it
                    self.logger.info(f"Reusing existing running container: {container_id[:12]}")
                    self._container_id = container_id
                    return
                else:
                    # Container is stopped; start it
                    self.logger.info(f"Starting existing stopped container: {container_id[:12]}")
                    subprocess.run(
                        ["docker", "start", container_id],
                        capture_output=True,
                        timeout=30,
                    )
                    time.sleep(1)
                    self._container_id = container_id
                    return

        # Build docker run command
        cmd = ["docker", "run", "-d"]
        cmd.extend(["--name", container_name])

        # Resource limits
        cmd.extend(["--memory", session_config.memory_limit])
        cmd.extend(["--cpus", str(session_config.cpu_limit)])

        # GPU devices
        if session_config.gpu_devices is not None:
            if isinstance(session_config.gpu_devices, str):
                if session_config.gpu_devices.lower() == "all":
                    cmd.extend(["--gpus", "all"])
                else:
                    cmd.extend(["--gpus", f"device={session_config.gpu_devices}"])
            elif isinstance(session_config.gpu_devices, list):
                devices_str = ",".join(session_config.gpu_devices)
                cmd.extend(["--gpus", f"device={devices_str}"])

        # Network
        cmd.extend(["--network", session_config.network_mode])

        # Working directory
        cmd.extend(["-w", session_config.working_dir])

        # Volume mounts
        for host_path, container_path in session_config.volumes.items():
            cmd.extend(["-v", f"{host_path}:{container_path}"])

        # Environment variables
        for key, value in session_config.env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Auto-remove
        if session_config.auto_remove:
            cmd.append("--rm")

        # Image and command (use tail -f to keep the container running)
        cmd.extend([session_config.image, "tail", "-f", "/dev/null"])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Failed to start container: {result.stderr}")

            self._container_id = result.stdout.strip()
            self.logger.info(f"Container started: {self._container_id[:12]}")

            # Initialize workspace directory permissions to ensure files can be written
            try:
                self.docker_exec(f"mkdir -p {session_config.working_dir} && chmod 777 {session_config.working_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize workspace directory: {e}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Timeout starting Docker container")
        except Exception as e:
            self.logger.error(f"Failed to start Docker container: {e}")
            raise

    def _setup_tmux(self) -> None:
        """Set up the tmux session."""
        if not self._container_id:
            raise RuntimeError("Container not started")

        session_name = f"evo-{self._container_id[:8]}"
        log_path = f"/tmp/evo-{self._container_id[:8]}.log"

        self._tmux_session = session_name
        self._tmux_log_path = log_path

        # Install tmux (if needed)
        self.docker_exec("apt-get update && apt-get install -y tmux || true", timeout=120)

        # Create tmux session
        self.docker_exec(f"tmux new-session -d -s {session_name} 'bash -i'")

        # Set up pipe logging
        self.docker_exec(f"tmux pipe-pane -o -t {session_name} 'cat >> {log_path}'")

        # Set PS1 prompt
        ps1 = BashMetadata.to_ps1_prompt()
        init_cmd = f"PROMPT_COMMAND='PS1=\"{ps1}\"'"
        self.tmux_send_keys(init_cmd, enter=True)

        # Trigger the first prompt
        self.tmux_send_keys("", enter=True)
        time.sleep(0.5)

        self.logger.debug(f"Tmux session {session_name} initialized")

    def docker_exec(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
    ) -> dict[str, Any]:
        """Execute a command in the container (direct execution, not via tmux).

        Args:
            command: Command to execute.
            timeout: Timeout in seconds.
            workdir: Working directory.

        Returns:
            Result dictionary containing:
            - stdout: Standard output
            - stderr: Standard error
            - exit_code: Exit code
            - output: Combined stdout + stderr
        """
        if not self._container_id:
            raise RuntimeError("Container not started")

        timeout = timeout or self.config.session_config.timeout

        cmd = ["docker", "exec"]
        if workdir:
            cmd.extend(["-w", workdir])
        cmd.extend([self._container_id, "bash", "-c", command])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "output": result.stdout + result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "exit_code": -1,
                "output": f"Command timed out after {timeout}s",
            }

    def tmux_send_keys(self, keys: str, enter: bool = False) -> None:
        """Send keys to the tmux session.

        Args:
            keys: Keys to send.
            enter: Whether to press Enter.
        """
        if not self._tmux_session:
            raise RuntimeError("Tmux session not initialized")

        # Escape single quotes
        escaped = keys.replace("'", "'\\''")
        cmd = f"tmux send-keys -t {self._tmux_session} '{escaped}'"
        if enter:
            cmd += " C-m"

        self.docker_exec(cmd)

    def get_tmux_logs(self) -> str:
        """Get the tmux session logs.

        Returns:
            Tmux log content.
        """
        if not self._tmux_log_path:
            return ""

        result = self.docker_exec(f"cat {self._tmux_log_path} 2>/dev/null || echo ''")
        return result.get("stdout", "")

    def is_mounted_path(self, container_path: str) -> tuple[bool, str | None]:
        """Check if the path is within a mounted volume.

        Args:
            container_path: Path inside the container (should be an absolute path).

        Returns:
            (is_mounted, host_path): Whether the path is in a mounted volume,
            and the corresponding host path (if it exists).
        """
        session_config = self.config.session_config
        if not session_config.volumes:
            return False, None

        # Normalize the container path (ensure it is absolute, remove trailing slash)
        container_path = str(Path(container_path).as_posix())
        if not container_path.startswith("/"):
            # If not an absolute path, it may need resolution first, but here we assume it is absolute
            # If it is not, return False
            return False, None

        # Check each mounted volume
        for host_path, mount_point in session_config.volumes.items():
            # Normalize the mount point path
            mount_point_norm = str(Path(mount_point).as_posix())

            # Check if the container path starts with the mount point
            # Ensure exact matching (avoid /workspace matching /workspace2)
            if container_path == mount_point_norm:
                # Exact match with the mount point itself
                return True, str(Path(host_path))
            elif container_path.startswith(mount_point_norm + "/"):
                # Is a sub-path of the mount point
                # Compute the relative path
                relative_path = container_path[len(mount_point_norm):].lstrip("/")
                # Build the host path
                host_path_obj = Path(host_path) / relative_path
                return True, str(host_path_obj)

        return False, None

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a file to the container.

        If the target path is within a mounted volume, copies the file directly on the host.

        Args:
            local_path: Local file path.
            remote_path: Remote file path (path inside the container).
        """
        if not self._container_id:
            raise RuntimeError("Container not started")

        # Check if the path is within a mounted volume
        is_mounted, host_path = self.is_mounted_path(remote_path)

        if is_mounted and host_path:
            # Copy the file directly on the host
            try:
                import shutil

                # Ensure the target directory exists
                host_path_obj = Path(host_path)
                host_path_obj.parent.mkdir(parents=True, exist_ok=True)

                # Copy the file
                shutil.copy2(local_path, host_path)
            except Exception as e:
                raise RuntimeError(f"Failed to upload file {local_path} to host path {host_path}: {e}")
            return

        # Not in a mounted volume; use docker cp
        # Ensure the remote directory exists and has correct permissions
        remote_dir = str(Path(remote_path).parent)
        # Create the directory and set permissions (777 ensures all users can write)
        self.docker_exec(f"mkdir -p {remote_dir} && chmod 777 {remote_dir}")

        cmd = ["docker", "cp", local_path, f"{self._container_id}:{remote_path}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to upload file: {result.stderr}")

        # Set file permissions after upload to ensure read/write/execute access
        self.docker_exec(f"chmod 777 {remote_path}")

    def download_file(self, remote_path: str, timeout: int | None = None) -> bytes:
        """Download a file from the container.

        If the path is within a mounted volume, reads directly from the host.

        Args:
            remote_path: Remote file path (path inside the container).
            timeout: Timeout in seconds.

        Returns:
            File content (bytes).
        """
        if not self._container_id:
            raise RuntimeError("Container not started")

        # Check if the path is within a mounted volume
        is_mounted, host_path = self.is_mounted_path(remote_path)

        if is_mounted and host_path:
            # Read directly from the host
            try:
                # Check if it is a directory
                if os.path.isdir(host_path):
                    raise RuntimeError(f"Cannot download directory: {remote_path}. Use exec_bash to list directory contents instead.")

                with open(host_path, "rb") as f:
                    return f.read()
            except FileNotFoundError:
                raise RuntimeError(f"File not found: {remote_path} (host path: {host_path})")
            except Exception as e:
                raise RuntimeError(f"Failed to download file {remote_path} from host: {e}")

        # Not in a mounted volume; use docker cp
        # Check if the path is a directory; docker cp cannot copy directories
        if self.is_directory(remote_path):
            raise RuntimeError(f"Cannot download directory: {remote_path}. Use exec_bash to list directory contents instead.")

        timeout = timeout or 60

        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            cmd = ["docker", "cp", f"{self._container_id}:{remote_path}", temp_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

            if result.returncode != 0:
                # Check if the error message contains a directory-related error
                error_msg = result.stderr.strip()
                if "cannot copy directory" in error_msg.lower() or "is a directory" in error_msg.lower():
                    raise RuntimeError(f"Cannot download directory: {remote_path}. Use exec_bash to list directory contents instead.")
                raise RuntimeError(f"Failed to download file: {error_msg}")

            with open(temp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def read_file_content(self, remote_path: str, encoding: str = "utf-8") -> str:
        """Read remote file content (text).

        If the path is within a mounted volume, reads directly from the host.

        Args:
            remote_path: Remote file path (path inside the container).
            encoding: File encoding.

        Returns:
            File content (string).
        """
        is_mounted, host_path = self.is_mounted_path(remote_path)

        if is_mounted and host_path:
            # Read directly from the host
            try:
                with open(host_path, "r", encoding=encoding) as f:
                    return f.read()
            except FileNotFoundError:
                raise RuntimeError(f"File not found: {remote_path} (host path: {host_path})")
            except Exception as e:
                raise RuntimeError(f"Failed to read file {remote_path} from host: {e}")

        # Not in a mounted volume; use download_file
        content = self.download_file(remote_path)
        return content.decode(encoding)

    def write_file_content(self, remote_path: str, content: str, encoding: str = "utf-8") -> None:
        """Write content to a remote file.

        If the path is within a mounted volume, writes directly on the host.

        Args:
            remote_path: Remote file path (path inside the container).
            content: File content.
            encoding: File encoding.
        """
        is_mounted, host_path = self.is_mounted_path(remote_path)

        if is_mounted and host_path:
            # Write directly on the host
            try:
                # Ensure the directory exists
                host_path_obj = Path(host_path)
                host_path_obj.parent.mkdir(parents=True, exist_ok=True)

                # Write the file
                with open(host_path, "w", encoding=encoding) as f:
                    f.write(content)
            except Exception as e:
                raise RuntimeError(f"Failed to write file {remote_path} to host: {e}")
            return

        # Not in a mounted volume; use upload_file
        import tempfile
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(content.encode(encoding))
            temp_path = f.name

        try:
            self.upload_file(temp_path, remote_path)
        finally:
            os.unlink(temp_path)

    def path_exists(self, remote_path: str) -> bool:
        """Check if a remote path exists.

        If the path is within a mounted volume, checks directly on the host.

        Args:
            remote_path: Remote path (path inside the container).

        Returns:
            Whether the path exists.
        """
        is_mounted, host_path = self.is_mounted_path(remote_path)

        if is_mounted and host_path:
            return os.path.exists(host_path)

        # Not in a mounted volume; use docker exec
        result = self.docker_exec(f'test -e "{remote_path}" && echo "exists" || echo "not_exists"')
        stdout = result.get("stdout", "").strip()
        return stdout == "exists"

    def is_file(self, remote_path: str) -> bool:
        """Check if a remote path is a file.

        If the path is within a mounted volume, checks directly on the host.

        Args:
            remote_path: Remote path (path inside the container).

        Returns:
            Whether the path is a file.
        """
        is_mounted, host_path = self.is_mounted_path(remote_path)

        if is_mounted and host_path:
            return os.path.isfile(host_path)

        # Not in a mounted volume; use docker exec
        result = self.docker_exec(f'test -f "{remote_path}" && echo "file" || echo "not_file"')
        stdout = result.get("stdout", "").strip()
        return stdout == "file"

    def is_directory(self, remote_path: str) -> bool:
        """Check if a remote path is a directory.

        If the path is within a mounted volume, checks directly on the host.

        Args:
            remote_path: Remote path (path inside the container).

        Returns:
            Whether the path is a directory.
        """
        is_mounted, host_path = self.is_mounted_path(remote_path)

        if is_mounted and host_path:
            return os.path.isdir(host_path)

        # Not in a mounted volume; use docker exec
        result = self.docker_exec(f'test -d "{remote_path}" && echo "dir" || echo "not_dir"')
        stdout = result.get("stdout", "").strip()
        return stdout == "dir"

