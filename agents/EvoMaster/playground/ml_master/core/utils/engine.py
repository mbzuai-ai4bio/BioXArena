from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

from evomaster.agent.tools.builtin.bash import BashTool


def _extract_fenced_block(text: str, language_hint: str) -> str:
    """Execute extract fenced block.

    Args:
        text: Input text content.
        language_hint: Value for language hint.

    Returns:
        str: Result of this function.
    """
    if not text:
        return ""
    pattern = rf"```(?:{language_hint})?\s*(.*?)```"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def extract_python_code(text: str) -> str:
    """Extract the first Python code block from text."""
    return _extract_fenced_block(text, "python|py")


def extract_json_code(text: str) -> str:
    """Extract the first JSON code block from text."""
    return _extract_fenced_block(text, "json")


def parse_metric_content(text: str) -> dict[str, Any]:
    """Parse metric-agent output into normalized metric detail."""
    if not text:
        return {"metric": None, "is_bug": True, "has_submission": False, "error": "empty metric text"}

    try:
        data = json.loads(extract_json_code(text.strip()))
    except Exception as exc:  # noqa: BLE001
        return {
            "metric": None,
            "is_bug": True,
            "has_submission": False,
            "error": f"metric json parse failed: {exc}",
        }

    if not isinstance(data, dict):
        return {"metric": None, "is_bug": True, "has_submission": False, "error": "metric content must be json object"}

    metric_value = data.get("metric")
    if metric_value is not None:
        try:
            data["metric"] = float(metric_value)
        except (TypeError, ValueError):
            return {
                "metric": None,
                "is_bug": True,
                "has_submission": False,
                "error": f"invalid metric value: {metric_value!r}",
            }

    data.setdefault("has_submission", True)
    data.setdefault("is_bug", data.get("metric") is None)
    return data


def extract_text_up_to_code(text: str) -> str:
    """Extract natural language before the first markdown code block."""
    if not text:
        return ""
    if "```" not in text:
        return text.strip()
    return text.split("```", 1)[0].strip()


def extract_agent_response(trajectory: Any) -> str:
    """Extract the latest assistant message from a trajectory."""
    if not trajectory or not getattr(trajectory, "dialogs", None):
        return ""

    for dialog in reversed(trajectory.dialogs):
        for message in reversed(getattr(dialog, "messages", []) or []):
            role = getattr(message, "role", None)
            role_value = getattr(role, "value", role)
            if role_value == "assistant":
                content = getattr(message, "content", "")
                if content:
                    return str(content)
    return ""


def run_code_via_bash(agent: Any, workspace: Path, code: str, node_id: str) -> dict[str, Any]:
    """
    Execute generated code via agent tool chain to preserve session semantics.
    """
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    script_path = workspace / f"solution_{node_id}.py"
    script_path.write_text(code, encoding="utf-8")

    command = f"cd {shlex.quote(str(workspace))} && python {shlex.quote(script_path.name)}"
    arguments = json.dumps(
        {"command": command, "is_input": "false", "timeout": "-1"},
        ensure_ascii=False,
    )

    tool_call = ChatCompletionMessageToolCall(
        id=f"call_{node_id}",
        type="function",
        function=Function(name=BashTool.name, arguments=arguments),
    )

    observation, info = agent._execute_tool(tool_call)
    info = info or {}

    return {
        "stdout": observation if isinstance(observation, str) else str(observation),
        "exit_code": info.get("exit_code", -1),
        "working_dir": info.get("working_dir"),
        "script": str(script_path),
    }
