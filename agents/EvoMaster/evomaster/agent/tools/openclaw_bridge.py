"""Openclaw Tool Bridge -- Manages communication with the Node.js bridge subprocess and Openclaw plugin tools.

Communicates with a Node.js bridge subprocess via stdin/stdout JSON-RPC protocol,
loading Openclaw plugins and executing their registered tools.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OpenclawBridge:
    """Manages the Node.js bridge subprocess lifecycle and tool execution."""

    def __init__(self, skills_ts_dir: Path):
        """Initialize OpenclawBridge.

        Args:
            skills_ts_dir: Path to the skills_ts directory (containing bridge/ and plugins/).
        """
        self.skills_ts_dir = skills_ts_dir
        self.process: subprocess.Popen | None = None
        self._tools_info: dict[str, dict[str, Any]] = {}
        self._request_id = 0
        self._lock = threading.Lock()
        self._stderr_thread: threading.Thread | None = None

    def start(self, plugins: list[str]) -> None:
        """Start the bridge subprocess, send init, and receive the tool list.

        Args:
            plugins: List of plugin names to load (e.g. ["feishu"]).
        """
        env = {**os.environ}
        self.process = subprocess.Popen(
            ["npx", "tsx", "bridge/server.ts"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.skills_ts_dir),
            env=env,
            bufsize=0,
        )

        # Start stderr reading thread (for log output)
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, daemon=True
        )
        self._stderr_thread.start()

        # Send init message
        response = self._send_and_recv(
            "init", {"plugins": plugins}
        )

        if "error" in response:
            raise RuntimeError(
                f"Bridge init failed: {response['error'].get('message', 'unknown error')}"
            )

        # Parse tool information
        tools_list = response.get("result", {}).get("tools", [])
        self._tools_info = {t["name"]: t for t in tools_list}

        logger.info(
            "Openclaw bridge started with %d tools: %s",
            len(self._tools_info),
            ", ".join(self._tools_info.keys()),
        )

    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute a tool and return the text result.

        Args:
            tool_name: Tool name (e.g. "feishu_doc").
            args: Tool arguments.

        Returns:
            Tool execution result text.
        """
        response = self._send_and_recv(
            "execute", {"tool_name": tool_name, "args": args}
        )

        if "error" in response:
            return json.dumps(
                {"error": response["error"].get("message", "unknown error")}
            )

        content = response.get("result", {}).get("content", [])
        if content:
            return content[0].get("text", "")
        return ""

    def get_tools_info(self) -> dict[str, dict[str, Any]]:
        """Return name/description/parameters for all tools.

        Returns:
            Mapping from tool name to tool information.
        """
        return self._tools_info

    def stop(self) -> None:
        """Shut down the bridge subprocess."""
        if self.process and self.process.poll() is None:
            try:
                self._send({"id": self._next_id(), "method": "shutdown"})
                self.process.wait(timeout=5)
            except Exception:
                logger.warning("Bridge shutdown timed out, killing process")
                self.process.kill()
                self.process.wait(timeout=3)
            finally:
                self.process = None

    def _next_id(self) -> int:
        """Generate the next request ID."""
        self._request_id += 1
        return self._request_id

    def _send_and_recv(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a request and wait for the response.

        Args:
            method: JSON-RPC method name.
            params: Request parameters.

        Returns:
            Response dictionary.
        """
        with self._lock:
            req_id = self._next_id()
            msg = {"id": req_id, "method": method, "params": params}
            self._send(msg)
            return self._recv()

    def _send(self, msg: dict[str, Any]) -> None:
        """Send a JSON line to stdin."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("Bridge process not running")
        line = json.dumps(msg) + "\n"
        self.process.stdin.write(line.encode("utf-8"))
        self.process.stdin.flush()

    def _recv(self) -> dict[str, Any]:
        """Read a JSON line from stdout."""
        if not self.process or not self.process.stdout:
            raise RuntimeError("Bridge process not running")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Bridge process terminated unexpectedly")
        return json.loads(line.decode("utf-8"))

    def _read_stderr(self) -> None:
        """Continuously read stderr and output to the logger."""
        if not self.process or not self.process.stderr:
            return
        try:
            for line in self.process.stderr:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.debug("[bridge] %s", text)
        except Exception:
            pass
