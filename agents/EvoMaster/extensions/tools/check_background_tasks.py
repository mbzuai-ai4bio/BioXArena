"""后台任务查询工具

允许 magiclaw 查询后台子智能体任务的状态和进度。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession
    from evomaster.interface.feishu.background_task import BackgroundTaskRegistry

logger = logging.getLogger(__name__)


class CheckBackgroundTasksParams(BaseToolParams):
    """查询后台任务的状态和进度。

    无需参数，自动查询当前聊天的所有后台任务。
    当用户询问后台任务进度、状态时使用此工具。
    """

    name: ClassVar[str] = "check_background_tasks"


class CheckBackgroundTasksTool(BaseTool):
    """查询后台子智能体任务状态的工具"""

    name: ClassVar[str] = "check_background_tasks"
    params_class: ClassVar[type[BaseToolParams]] = CheckBackgroundTasksParams

    def __init__(
        self,
        task_registry: BackgroundTaskRegistry | None = None,
        chat_id: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._task_registry = task_registry
        self._chat_id = chat_id

    def set_context(self, task_registry: BackgroundTaskRegistry, chat_id: str) -> None:
        """由 dispatcher 调用，注入运行时上下文。"""
        self._task_registry = task_registry
        self._chat_id = chat_id

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        if not self._task_registry or not self._chat_id:
            return "后台任务系统未初始化。", {}

        from evomaster.interface.feishu.background_task import BackgroundTaskStatus

        tasks = self._task_registry.get_tasks_for_chat(self._chat_id)
        if not tasks:
            return "当前没有后台任务。", {"task_count": 0}

        # 按状态排序：活跃 > 已完成未审阅 > 已审阅
        def sort_key(t):
            if t.is_active:
                return 0
            if not t.reviewed:
                return 1
            return 2
        tasks.sort(key=sort_key)

        status_emoji = {
            BackgroundTaskStatus.PENDING: "⏳",
            BackgroundTaskStatus.RUNNING: "🔄",
            BackgroundTaskStatus.COMPLETED: "✅",
            BackgroundTaskStatus.FAILED: "❌",
        }

        lines = [f"共 {len(tasks)} 个后台任务:\n"]
        for t in tasks:
            emoji = status_emoji.get(t.status, "❓")
            elapsed = int(t.elapsed_seconds)
            elapsed_str = f"{elapsed // 60}m{elapsed % 60}s" if elapsed >= 60 else f"{elapsed}s"

            line = f"{emoji} [{t.agent_name}] {t.task_description[:80]}"
            line += f"\n   状态: {t.status.value} | 步数: {t.step_count} | 耗时: {elapsed_str}"

            if t.latest_tool_calls:
                line += f"\n   最近工具: {', '.join(t.latest_tool_calls)}"

            if t.status == BackgroundTaskStatus.COMPLETED and t.result:
                preview = t.result[:200] + ("..." if len(t.result) > 200 else "")
                line += f"\n   结果预览: {preview}"
            elif t.status == BackgroundTaskStatus.FAILED and t.error:
                line += f"\n   错误: {t.error[:200]}"

            if t.reviewed:
                line += "\n   (已审阅)"

            lines.append(line)

        summary = "\n\n".join(lines)
        active_count = sum(1 for t in tasks if t.is_active)
        return summary, {"task_count": len(tasks), "active_count": active_count}