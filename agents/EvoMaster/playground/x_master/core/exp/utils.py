"""X-Master 实验工具函数"""

import json
from typing import Any


def strip_think_and_exec(text: str) -> str:
    """清理文本中的 </think> 和 </execution_results> 标签及其之前的内容

    保留可见的答案部分，移除思考过程和执行结果的尾部标记。
    这个函数用于在传递给下游 Agent（如 Critic/Rewriter/Selector）之前
    清理上游 Agent 的输出，确保下游只看到最终答案而非中间过程。

    Args:
        text: 原始文本

    Returns:
        清理后的文本
    """
    if text is None:
        return ""
    out = text
    if "</think>" in out:
        out = out.split("</think>")[-1]
    if "</execution_results>" in out:
        out = out.split("</execution_results>")[-1]
    return out.strip()



def extract_final_response(trajectory: Any) -> str:
    """从轨迹中提取Agent的最终回答（同时包含content + finish message）"""
    if not trajectory or not trajectory.dialogs:
        return ""

    last_dialog = trajectory.dialogs[-1]

    assistant_content = None
    finish_message = None

    for message in reversed(last_dialog.messages):
        if not (hasattr(message, "role") and getattr(message.role, "value", message.role) == "assistant"):
            continue

        if assistant_content is None and hasattr(message, "content") and message.content:
            assistant_content = message.content

        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                fn = getattr(tc, "function", tc) if hasattr(tc, "function") else tc
                name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None)

                if name == "finish":
                    args = getattr(fn, "arguments", None) or (
                        fn.get("arguments", "{}") if isinstance(fn, dict) else "{}"
                    )
                    try:
                        obj = json.loads(args) if isinstance(args, str) else args
                        if isinstance(obj, dict) and "message" in obj:
                            finish_message = obj["message"]
                    except (json.JSONDecodeError, TypeError):
                        pass

        if assistant_content and finish_message:
            break

    if assistant_content and finish_message:
        return f"{assistant_content}\n\n[FINAL]\n{finish_message}"
    elif finish_message:
        return finish_message
    elif assistant_content:
        return assistant_content

    return ""


def extract_agent_response(trajectory: Any) -> str:
    """从轨迹中提取Agent的最终回答

    Args:
        trajectory: 执行轨迹

    Returns:
        Agent的回答文本
    """
    if not trajectory or not trajectory.dialogs:
        return ""
    last_dialog = trajectory.dialogs[-1]
    for message in reversed(last_dialog.messages):
        if not (hasattr(message, "role") and getattr(message.role, "value", message.role) == "assistant"):
            continue
        # 若该条 assistant 调用了 finish，优先用 finish 的 message 作为回答
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                fn = getattr(tc, "function", tc) if hasattr(tc, "function") else tc
                name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None)
                if name == "finish":
                    args = getattr(fn, "arguments", None) or (fn.get("arguments", "{}") if isinstance(fn, dict) else "{}")
                    try:
                        obj = json.loads(args) if isinstance(args, str) else args
                        if isinstance(obj, dict) and "message" in obj:
                            return obj["message"]
                    except (json.JSONDecodeError, TypeError):
                        pass
        if hasattr(message, "content") and message.content:
            return message.content
    return ""
