# Session - The Interface Between Agent and Env

Session is the intermediate layer for Agent interaction with remote cluster environments, providing a unified interface for executing commands, transferring files, etc.

## Directory Structure

- `base.py` - Session abstract base class, defining the standard interface
- `local.py` - Local Session implementation, executing commands locally
- `docker.py` - Docker Session implementation, using Docker containers for isolated execution environments

## Core Classes

### BaseSession (base.py)
The abstract base class for Session, defining interfaces that all Session implementations must provide:

- `open()` / `close()` - Session lifecycle management
- `exec_bash(command)` - Execute Bash commands
- `upload(local, remote)` - Upload files
- `download(remote)` - Download files
- `read_file()` / `write_file()` - Text file read/write
- `path_exists()` / `is_file()` / `is_directory()` - Path checks

### LocalSession (local.py)
Local Session implementation, executing commands directly on the local machine:

- Uses subprocess to execute bash commands directly
- File operations are local copy/read/write
- Suitable for development and testing
- No external dependencies required (Docker, clusters, etc.)

### DockerSession (docker.py)
Docker-based Session implementation providing an isolated execution environment:

- Uses Docker containers as the execution environment
- Maintains persistent bash sessions via tmux
- Supports environment variables, working directory, and other state persistence
- Supports resource limits (memory, CPU) and volume mounts

## Usage Examples

### Local Session

```python
from evomaster.agent.session import LocalSession, LocalSessionConfig

# Create configuration
config = LocalSessionConfig(timeout=30)

# Use Session
with LocalSession(config) as session:
    # Execute command
    result = session.exec_bash("python --version")
    print(result["stdout"])

    # Upload file (local copy)
    session.upload("/local/path", "/tmp/remote.py")

    # Download file (local read)
    content = session.download("/tmp/file.txt")
```

### Docker Session

```python
from evomaster.agent.session import DockerSession, DockerSessionConfig

# Create configuration
config = DockerSessionConfig(
    image="python:3.11-slim",
    memory_limit="4g",
    cpu_limit=2.0,
)

# Use Session
with DockerSession(config) as session:
    # Execute command
    result = session.exec_bash("python --version")
    print(result["stdout"])

    # Upload file
    session.upload("/local/path", "/workspace/remote.py")

    # Download file
    content = session.download("/workspace/output.txt")
```

## Design Features

1. **Abstract Interface** - BaseSession defines a standard interface, enabling multiple implementations (local, remote, Kubernetes, etc.)
2. **Multiple Implementations** - Supports local, Docker, and future environments
3. **Isolated Environment** - Docker containers provide fully isolated execution environments
4. **Persistent Sessions** - Uses tmux to maintain bash state for long-running experiments
5. **Resource Management** - Supports memory, CPU, and other resource limits
6. **Context Manager** - Implements Python's context manager interface

## Configuration Parameters

### SessionConfig (Base Configuration)
- `timeout` - Command execution timeout in seconds, default 300
- `workspace_path` - Workspace path, default `/workspace`

### LocalSessionConfig (Local Session Configuration)
Inherits from `SessionConfig`, additional parameters:
- `encoding` - File encoding, default `utf-8`

### DockerSessionConfig (Docker Session Configuration)
Inherits from `SessionConfig`, additional parameters:
- `image` - Docker image name, default `python:3.11-slim`
- `container_name` - Container name, auto-generated if None
- `memory_limit` - Memory limit, default `4g`
- `cpu_limit` - CPU limit, default 2.0
- `volumes` - Volume mounts {host_path: container_path}
- `env_vars` - Environment variables
- `auto_remove` - Auto-remove container when finished, default True

## Future Extensions

The following can be implemented on this foundation:
- `RemoteSession` - SSH connection to remote servers
- `KubernetesSession` - Kubernetes cluster execution
- `RaySession` - Ray distributed framework
