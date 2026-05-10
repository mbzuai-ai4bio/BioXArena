"""EvoMaster Bash tool.

Provides the ability to execute Bash commands in the environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field

from ..base import BaseTool, BaseToolParams, ToolError

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession


class BashToolParams(BaseToolParams):
    """Execute a bash command in the terminal within a persistent shell session.

    ### Command Execution
    * One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.
    * Persistent session: Commands execute in a persistent shell session where environment variables, virtual environments, and working directory persist between commands.
    * Soft timeout: Commands have a soft timeout of 10 seconds, once that's reached, you have the option to continue or interrupt the command.

    ### Long-running Commands
    * For commands that may run indefinitely, run them in the background and redirect output to a file, e.g. `python3 app.py > server.log 2>&1 &`.
    * For commands that may run for a long time, you should set the "timeout" parameter to an appropriate value.
    * If a bash command returns exit code `-1`, this means the process hit the soft timeout and is not yet finished. By setting `is_input` to `true`, you can:
      - Send empty `command` to retrieve additional logs
      - Send text (set `command` to the text) to STDIN of the running process
      - Send control commands like `C-c` (Ctrl+C), `C-z` (Ctrl+Z) to interrupt the process

    ### Best Practices
    * Directory verification: Before creating new directories or files, first verify the parent directory exists and is the correct location.
    * Directory management: Try to maintain working directory by using absolute paths and avoiding excessive use of `cd`.

    ### Output Handling
    * Output truncation: If the output exceeds a maximum length, it will be truncated.
    """
    
    name: ClassVar[str] = "execute_bash"

    command: str = Field(
        description="The bash command to execute. Can be empty string to view additional logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently running process."
    )
    is_input: Literal["true", "false"] = Field(
        default="false",
        description="If True, the command is an input to the running process. If False, the command is a bash command to be executed in the terminal. Default is False.",
    )
    timeout: float = Field(
        default=-1,
        description="Optional. Sets a hard timeout in seconds for the command execution. If not provided, the command will use the default soft timeout behavior.",
    )


class BashTool(BaseTool):
    """Bash command execution tool."""
    
    name: ClassVar[str] = "execute_bash"
    params_class: ClassVar[type[BaseToolParams]] = BashToolParams

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """Execute a Bash command."""
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {str(e)}", {"error": str(e)}
        
        assert isinstance(params, BashToolParams)
        
        # Execute command
        timeout = int(params.timeout) if params.timeout > 0 else None
        is_input = params.is_input == "true"
        
        result = session.exec_bash(
            command=params.command,
            timeout=timeout,
            is_input=is_input,
        )
        
        # Build output
        # Prefer merged output (stdout+stderr) to ensure errors are visible
        output = result.get("output", "") or result.get("stdout", "")
        exit_code = result.get("exit_code", -1)
        working_dir = result.get("working_dir", "")
        
        # Convert relative path to absolute path
        working_dir_abs = str(Path(working_dir).absolute()) if working_dir else ""
        
        # Add status information
        obs = output
        if working_dir_abs:
            obs += f"\n[Current working directory: {working_dir_abs}]"
        if exit_code != -1:
            obs += f"\n[Command finished with exit code {exit_code}]"
        
        info = {
            "exit_code": exit_code,
            "working_dir": working_dir_abs,
        }
        
        return obs, info

