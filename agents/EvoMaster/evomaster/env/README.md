# Env - Execution Environment Management

Env is EvoMaster's environment component, responsible for managing execution environments and job scheduling.

## Directory Structure

- `base.py` - Env abstract base class, defining the standard interface
- `local.py` - LocalEnv implementation, executing commands locally

## Core Classes

### BaseEnv (base.py)
The abstract base class for Env, defining interfaces that all Env implementations must provide:

- `setup()` / `teardown()` - Environment lifecycle management
- `get_session()` - Get a Session for executing commands
- `submit_job(command, job_type)` - Submit a job
- `get_job_status(job_id)` - Query job status
- `cancel_job(job_id)` - Cancel a job

### LocalEnv (local.py)
Local environment implementation, no Docker or cluster required:

- Executes commands directly on the local machine
- Synchronous job execution
- Supports job status queries
- Suitable for development and testing

### LocalSession (local.py)
Local Session implementation:

- Uses subprocess to execute commands directly
- Supports file upload/download (actually local copy operations)
- Works on the local file system

## Usage Examples

### Basic Usage

```python
from evomaster.env import LocalEnv, LocalEnvConfig

# Create local environment
config = LocalEnvConfig(name="my_env")
env = LocalEnv(config)

# Setup environment
env.setup()

try:
    # Get Session to execute commands directly
    session = env.get_session()
    result = session.exec_bash("python --version")
    print(result["stdout"])

    # Submit job
    job_id = env.submit_job("python -c 'print(123)'", job_type="debug")

    # Query job status
    status = env.get_job_status(job_id)
    print(status)

finally:
    env.teardown()
```

### Using Context Manager

```python
with LocalEnv() as env:
    session = env.get_session()
    result = session.exec_bash("ls -la")
    print(result["stdout"])
```

## Design Features

1. **Simple Implementation** - Executes locally with no complex dependencies
2. **Standard Interface** - BaseEnv defines a unified environment interface
3. **Forward Compatible** - Easily replaceable with Docker, Kubernetes, or other implementations
4. **Job Management** - Supports job submission, status queries, and cancellation
5. **Context Manager** - Implements Python's context manager interface

## Configuration Parameters

### EnvConfig (Base Configuration)
- `name` - Environment name
- `session_config` - Session configuration

### LocalEnvConfig (Local Environment Configuration)
- Inherits all configuration from EnvConfig
- Uses local Session by default

## Future Extensions

The following can be implemented on this foundation:
- `DockerEnv` - Using Docker containers
- `KubernetesEnv` - Using Kubernetes clusters
- `RemoteEnv` - Connecting to remote servers
- `RayEnv` - Using Ray distributed framework
