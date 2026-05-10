"""Scheduler tools for MagiClaw agent

ScheduleTaskTool: create scheduled jobs (at/every/cron)
ManageSchedulesTool: list/cancel/history actions
Both use set_context() for runtime injection by dispatcher.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession
    from evomaster.scheduler.service import SchedulerService
    from evomaster.scheduler.store import ScheduledJobStore

logger = logging.getLogger(__name__)


def _format_time(epoch: float, tz_str: str = "Asia/Shanghai") -> str:
    """Format epoch → human-readable local time string."""
    try:
        from zoneinfo import ZoneInfo
        tzinfo = ZoneInfo(tz_str)
    except Exception:
        tzinfo = None
    dt = datetime.fromtimestamp(epoch, tz=tzinfo)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# ScheduleTaskTool
# ---------------------------------------------------------------------------


class ScheduleTaskParams(BaseToolParams):
    """创建定时/周期任务。

    支持三种调度类型：
    - at: 一次性任务，如 "30m"（30分钟后）或 "2026-03-21T15:00:00"
    - every: 固定间隔，如 "30m"（每30分钟）、"1h"（每小时）
    - cron: Cron 表达式，如 "0 9 * * 1-5"（工作日每天9点）
    """

    name: ClassVar[str] = "schedule_task"

    schedule_type: str = Field(
        description='调度类型: "at"（一次性）, "every"（固定间隔）, "cron"（cron 表达式）'
    )
    schedule_expr: str = Field(
        description='调度表达式。at: "30m"/"2h"/"2026-03-21T15:00:00"。every: "30m"/"1h"。cron: "0 9 * * 1-5"'
    )
    task_description: str = Field(
        description="任务描述（自然语言），将作为 agent 的输入"
    )
    agent_name: str = Field(
        default="magiclaw",
        description="执行任务的 agent 名称，默认 magiclaw",
    )


class ScheduleTaskTool(BaseTool):
    """创建定时/周期任务"""

    name: ClassVar[str] = "schedule_task"
    params_class: ClassVar[type[BaseToolParams]] = ScheduleTaskParams

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._store: ScheduledJobStore | None = None
        self._service: SchedulerService | None = None
        self._chat_id: str = ""
        self._creator_id: str = ""
        self._max_jobs_per_chat: int = 20
        self._max_jobs_total: int = 200
        self._default_timezone: str = "Asia/Shanghai"

    def set_context(
        self,
        store: ScheduledJobStore,
        service: SchedulerService,
        chat_id: str,
        creator_id: str,
        max_jobs_per_chat: int = 20,
        max_jobs_total: int = 200,
        default_timezone: str = "Asia/Shanghai",
    ) -> None:
        """Injected by dispatcher at runtime."""
        self._store = store
        self._service = service
        self._chat_id = chat_id
        self._creator_id = creator_id
        self._max_jobs_per_chat = max_jobs_per_chat
        self._max_jobs_total = max_jobs_total
        self._default_timezone = default_timezone

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        if not self._store or not self._service:
            return "定时任务系统未初始化。", {}

        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"参数错误: {e}", {}

        from evomaster.scheduler.models import JobStatus, ScheduleType
        from evomaster.scheduler.parser import (
            parse_at_expr,
            parse_every_expr,
            validate_cron_expr,
        )
        from evomaster.scheduler.store import ScheduledJobStore

        schedule_type_str = params.schedule_type.lower().strip()
        schedule_expr = params.schedule_expr.strip()
        task_desc = params.task_description.strip()
        agent_name = params.agent_name.strip() or "magiclaw"

        if not task_desc:
            return "任务描述不能为空。", {}

        # Validate schedule type
        try:
            stype = ScheduleType(schedule_type_str)
        except ValueError:
            return f"不支持的调度类型: {schedule_type_str!r}。支持: at, every, cron", {}

        # Validate expression and compute first fire time
        tz = self._default_timezone
        try:
            if stype == ScheduleType.AT:
                next_run = parse_at_expr(schedule_expr, tz)
                max_runs = 1
            elif stype == ScheduleType.EVERY:
                interval = parse_every_expr(schedule_expr)
                next_run = time.time() + interval
                max_runs = None
            elif stype == ScheduleType.CRON:
                validate_cron_expr(schedule_expr)
                # Compute first fire
                from evomaster.scheduler.parser import compute_next_run as _compute
                from evomaster.scheduler.models import ScheduledJob as _SJ
                # Temporary job just for computing next_run
                tmp = _SJ(
                    job_id="", chat_id="", creator_id="",
                    schedule_type=stype, schedule_expr=schedule_expr,
                    task_description="", agent_name="",
                    status=JobStatus.ACTIVE, created_at=0, next_run_at=0,
                    timezone=tz,
                )
                next_run = _compute(tmp)
                max_runs = None
            else:
                return f"不支持的调度类型: {stype}", {}
        except ValueError as e:
            return f"表达式错误: {e}", {}

        # Check limits
        chat_count = self._store.count_active_for_chat(self._chat_id)
        if chat_count >= self._max_jobs_per_chat:
            return f"当前会话已有 {chat_count} 个活跃任务，达到上限 ({self._max_jobs_per_chat})。", {}

        total_count = self._store.count_active_total()
        if total_count >= self._max_jobs_total:
            return f"系统活跃任务总数已达上限 ({self._max_jobs_total})。", {}

        # Create job
        from evomaster.scheduler.models import ScheduledJob

        job_id = ScheduledJobStore.generate_id()
        job = ScheduledJob(
            job_id=job_id,
            chat_id=self._chat_id,
            creator_id=self._creator_id,
            schedule_type=stype,
            schedule_expr=schedule_expr,
            timezone=tz,
            task_description=task_desc,
            agent_name=agent_name,
            status=JobStatus.ACTIVE,
            created_at=time.time(),
            next_run_at=next_run,
            max_runs=max_runs,
        )

        self._store.add(job)
        self._service.wake()

        # Build confirmation
        type_labels = {"at": "一次性", "every": "定时循环", "cron": "Cron"}
        next_run_str = _format_time(next_run, tz)
        msg = (
            f"已创建{type_labels.get(stype.value, stype.value)}定时任务\n"
            f"ID: {job_id}\n"
            f"类型: {stype.value} ({schedule_expr})\n"
            f"Agent: {agent_name}\n"
            f"任务: {task_desc[:200]}\n"
            f"下次执行: {next_run_str}"
        )

        return msg, {"job_id": job_id, "next_run_at": next_run}


# ---------------------------------------------------------------------------
# ManageSchedulesTool
# ---------------------------------------------------------------------------


class ManageSchedulesParams(BaseToolParams):
    """管理定时任务：查看列表、取消任务、查看执行历史。

    action:
    - list: 列出当前聊天的所有活跃定时任务
    - cancel: 取消指定任务（需要 job_id）
    - history: 查看指定任务的最近执行记录（需要 job_id）
    """

    name: ClassVar[str] = "manage_schedules"

    action: str = Field(
        description='操作: "list"（列出任务）, "cancel"（取消任务）, "history"（执行历史）'
    )
    job_id: str = Field(
        default="",
        description="任务 ID（cancel 和 history 操作需要）",
    )


class ManageSchedulesTool(BaseTool):
    """管理定时任务"""

    name: ClassVar[str] = "manage_schedules"
    params_class: ClassVar[type[BaseToolParams]] = ManageSchedulesParams

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._store: ScheduledJobStore | None = None
        self._chat_id: str = ""

    def set_context(self, store: ScheduledJobStore, chat_id: str) -> None:
        """Injected by dispatcher at runtime."""
        self._store = store
        self._chat_id = chat_id

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        if not self._store:
            return "定时任务系统未初始化。", {}

        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"参数错误: {e}", {}

        action = params.action.lower().strip()

        if action == "list":
            return self._list_jobs()
        elif action == "cancel":
            return self._cancel_job(params.job_id.strip())
        elif action == "history":
            return self._show_history(params.job_id.strip())
        else:
            return f"不支持的操作: {action!r}。支持: list, cancel, history", {}

    def _list_jobs(self) -> tuple[str, dict[str, Any]]:
        jobs = self._store.get_active_for_chat(self._chat_id)
        if not jobs:
            return "当前没有活跃的定时任务。", {"count": 0}

        lines = [f"共 {len(jobs)} 个活跃定时任务:\n"]
        for j in jobs:
            next_str = _format_time(j.next_run_at, j.timezone)
            line = (
                f"[{j.job_id}] {j.schedule_type.value}({j.schedule_expr}) "
                f"→ {j.agent_name}\n"
                f"  任务: {j.task_description[:80]}\n"
                f"  下次执行: {next_str} | 已执行: {j.run_count}次"
            )
            if j.consecutive_failures > 0:
                line += f" | 连续失败: {j.consecutive_failures}"
            lines.append(line)

        return "\n\n".join(lines), {"count": len(jobs)}

    def _cancel_job(self, job_id: str) -> tuple[str, dict[str, Any]]:
        if not job_id:
            return "请提供要取消的任务 ID。", {}

        job = self._store.get(job_id)
        if not job:
            return f"未找到任务: {job_id}", {}
        if job.chat_id != self._chat_id:
            return "无权取消其他会话的任务。", {}
        if job.status != JobStatus.ACTIVE:
            return f"任务 {job_id} 状态为 {job.status.value}，无需取消。", {}

        from evomaster.scheduler.models import JobStatus

        self._store.mark_cancelled(job_id)
        return f"已取消任务 {job_id}: {job.task_description[:80]}", {"cancelled": job_id}

    def _show_history(self, job_id: str) -> tuple[str, dict[str, Any]]:
        if not job_id:
            return "请提供要查询的任务 ID。", {}

        job = self._store.get(job_id)
        if not job:
            return f"未找到任务: {job_id}", {}
        if job.chat_id != self._chat_id:
            return "无权查看其他会话的任务。", {}

        runs = self._store.get_runs(job_id, limit=10)
        if not runs:
            return f"任务 {job_id} 暂无执行记录。", {"count": 0}

        lines = [f"任务 {job_id} 最近 {len(runs)} 次执行:\n"]
        for r in runs:
            started = _format_time(r.started_at, job.timezone)
            status = "✅" if r.success else "❌"
            line = f"{status} {started}"
            if r.finished_at:
                duration = r.finished_at - r.started_at
                line += f" ({duration:.1f}s)"
            if r.success and r.result:
                line += f"\n  结果: {r.result[:150]}"
            elif not r.success and r.error:
                line += f"\n  错误 ({r.error_kind or '?'}): {r.error[:150]}"
            lines.append(line)

        return "\n\n".join(lines), {"count": len(runs)}