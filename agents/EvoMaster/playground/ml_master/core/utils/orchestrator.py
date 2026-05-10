from __future__ import annotations

import logging
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from evomaster.agent import Agent

from .grading import shutdown_embedded_grading_server, validate_submission
from .artifacts import (
    append_trajectory,
    build_review,
    copy_submission,
    save_best,
    save_node_snapshot,
)
from .uct import UCTDecayConfig, UCTNode, UCTSearchConfig, UCTSearchManager


@dataclass
class BestState:
    code: Optional[str] = None
    metric: Optional[float] = None
    node_id: Optional[str] = None
    dispatch_id: int = 0
    active_jobs: int = 0


@dataclass
class WorkerTask:
    stage: str
    node: UCTNode
    prev_code: str
    term_out: str
    issue: str
    memory: str
    best_code: Optional[str]
    best_metric: Optional[float]
    dispatch_id: int


def parallel_config(config: Any) -> dict[str, Any]:
    """Execute parallel config.

    Args:
        config: Configuration object.

    Returns:
        dict[str, Any]: Result of this function.
    """
    session_cfg = config.session.get("local", {})
    return session_cfg.get("parallel", {})


def link_or_copy(source: Path, destination: Path) -> None:
    """Execute link or copy.

    Args:
        source: Value for source.
        destination: Value for destination.

    Returns:
        None.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy(source, destination)


def ensure_prepared_links(config: Any, workspace: Path) -> None:
    """Ensure prepared links.

    Args:
        config: Configuration object.
        workspace: Workspace path.

    Returns:
        None.
    """
    exp_id = getattr(config, "exp_id", None)
    data_root = getattr(config, "data_root", None)
    if not (exp_id and data_root):
        return

    prepared = Path(data_root) / exp_id / "prepared"
    baseline_src = prepared / "baseline.json"
    baseline_dst = workspace / "input" / "baseline.json"
    if baseline_src.exists():
        link_or_copy(baseline_src, baseline_dst)

    grade_src = prepared / "grade.py"
    grade_dst = workspace / "grade.py"
    if grade_src.exists():
        link_or_copy(grade_src, grade_dst)


def resolve_worker_workspace(config: Any, worker_index: int, main_workspace: Path) -> Path:
    """Resolve worker workspace.

    Args:
        config: Configuration object.
        worker_index: Numeric control parameter.
        main_workspace: Workspace-related path or name.

    Returns:
        Path: Result of this function.
    """
    split_workspace = parallel_config(config).get("split_workspace_for_exp", False)
    return main_workspace / f"exp_{worker_index}" if split_workspace else main_workspace


def create_worker_agents(playground: Any, worker_index: int) -> dict[str, Agent]:
    """Create worker agents.

    Args:
        playground: Playground instance.
        worker_index: Numeric control parameter.

    Returns:
        dict[str, Agent]: Result of this function.
    """
    return {
        "draft": playground.copy_agent(playground.agents.draft_agent, new_agent_name=f"draft_worker_{worker_index}"),
        "debug": playground.copy_agent(playground.agents.debug_agent, new_agent_name=f"debug_worker_{worker_index}"),
        "improve": playground.copy_agent(playground.agents.improve_agent, new_agent_name=f"improve_worker_{worker_index}"),
        "metric": playground.copy_agent(playground.agents.metric_agent, new_agent_name=f"metric_worker_{worker_index}"),
    }


def prepare_workspace(workspace: Path) -> Path:
    """Execute prepare workspace.

    Args:
        workspace: Workspace path.

    Returns:
        Path: Result of this function.
    """
    (workspace / "working").mkdir(parents=True, exist_ok=True)
    (workspace / "best_solution").mkdir(parents=True, exist_ok=True)
    (workspace / "best_submission").mkdir(parents=True, exist_ok=True)
    submission_dir = workspace / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    return submission_dir


def create_search_manager(
    *,
    config: Any,
    session: Any,
    run_dir: str | Path | None,
    submission_dir: Path,
) -> UCTSearchManager:
    """Create search manager.

    Args:
        config: Configuration object.
        session: Execution session object.
        run_dir: Run directory path.
        submission_dir: Directory path.

    Returns:
        UCTSearchManager: Result of this function.
    """
    servers = getattr(config, "grading_servers", []) or []
    search_mgr = UCTSearchManager(
        search_cfg=UCTSearchConfig(),
        decay_cfg=UCTDecayConfig(),
        grader=lambda exp_id, submission_path: validate_submission(
            exp_id,
            submission_path,
            server_urls=servers,
            dataset_root=getattr(config, "data_root", None),
        ),
        exp_id=getattr(config, "exp_id", "unknown"),
        submission_dir=submission_dir,
    )

    search_mgr.set_snapshot_fn(
        lambda node, submission, review, reward: save_node_snapshot(
            run_dir,
            Path(session.config.workspace_path),
            node,
            submission,
            review,
            reward,
            search_mgr,
        )
    )
    return search_mgr


def reset_working_dir(workspace: Path, logger: logging.Logger) -> None:
    """Execute reset working dir.

    Args:
        workspace: Workspace path.
        logger: Logger instance.

    Returns:
        None.
    """
    working_dir = workspace / "working"
    try:
        if working_dir.exists():
            shutil.rmtree(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to reset working dir %s: %s", working_dir, exc)


def select_stage_and_inputs(target: UCTNode) -> tuple[str, str, str, str]:
    """Select stage and inputs.

    Args:
        target: Target UCT node.

    Returns:
        tuple[str, str, str, str]: Result of this function.
    """
    if target.stage == "root":
        return "draft", "", "", ""
    issue = getattr(target, "grading_issue", "") or ""
    if target.is_buggy or target.metric.value is None:
        return "debug", getattr(target, "code", ""), getattr(target, "stdout", ""), issue
    return "improve", getattr(target, "code", ""), getattr(target, "stdout", ""), issue


def run_one_node(
    *,
    config: Any,
    session: Any,
    worker_agents: dict[str, Agent],
    worker_workspace: Path,
    data_preview: str,
    task_description: str,
    stage: str,
    node: UCTNode,
    prev_code: str,
    term_out: str,
    issue: str,
    best_code: str | None,
    best_metric: float | None,
    memory: str,
    exp_index: int,
) -> dict[str, Any]:
    # Local import avoids circular dependency: exp -> runtime utilities -> exp
    """Run one node.

    Args:
        config: Configuration object.
        session: Execution session object.
        worker_agents: Worker-specific agent map.
        worker_workspace: Workspace-related path or name.
        data_preview: Value for data preview.
        task_description: Natural language task description.
        stage: Value for stage.
        node: UCT node object.
        prev_code: Previous Python code string.
        term_out: Terminal output text.
        issue: Issue description used for debugging.
        best_code: Current best Python code string.
        best_metric: Metric value or metric-related input.
        memory: Context memory text.
        exp_index: Numeric control parameter.

    Returns:
        dict[str, Any]: Result of this function.
    """
    from ..exp.debug_exp import DebugExp
    from ..exp.draft_exp import DraftExp
    from ..exp.improve_exp import ImproveExp

    common_args = (
        session,
        worker_workspace,
        getattr(config, "exp_id", None),
        data_preview,
        node,
    )

    if stage == "draft":
        exp = DraftExp(worker_agents["draft"], worker_agents["metric"], *common_args, exp_index=exp_index)
        return exp.run(task_description, memory=memory)

    if stage == "debug":
        exp = DebugExp(worker_agents["debug"], worker_agents["metric"], *common_args, exp_index=exp_index)
        return exp.run(task_description, prev_code=prev_code, term_out=term_out, issue=issue)

    exp = ImproveExp(worker_agents["improve"], worker_agents["metric"], *common_args, exp_index=exp_index)
    return exp.run(
        task_description,
        best_code=best_code or prev_code,
        best_metric=best_metric,
        memory=memory,
        term_out=term_out,
    )


def acquire_work_item(
    *,
    search_mgr: UCTSearchManager,
    best_state: BestState,
    lock: threading.Lock,
    max_steps: int,
) -> tuple[bool, WorkerTask | None]:
    """Execute acquire work item.

    Args:
        search_mgr: UCT search manager.
        best_state: Shared best-state tracker.
        lock: Value for lock.
        max_steps: Numeric control parameter.

    Returns:
        tuple[bool, WorkerTask | None]: Result of this function.
    """
    with lock:
        if search_mgr.current_step >= max_steps:
            return False, None

        target = search_mgr.select_next()
        if target is None:
            should_wait = best_state.active_jobs > 0
            return should_wait, None

        if target.stage != "root" and target.is_buggy is None:
            return True, None

        stage, prev_code, term_out, issue = select_stage_and_inputs(target)
        node = search_mgr.create_child(target, stage=stage, plan="", code="")
        best_state.active_jobs += 1
        dispatch_id = best_state.dispatch_id
        best_state.dispatch_id += 1

        memory = search_mgr.root.fetch_child_memory() if stage == "draft" else target.fetch_child_memory()
        return (
            False,
            WorkerTask(
                stage=stage,
                node=node,
                prev_code=prev_code,
                term_out=term_out,
                issue=issue,
                memory=memory,
                best_code=best_state.code,
                best_metric=best_state.metric,
                dispatch_id=dispatch_id,
            ),
        )


def finalize_work_item(
    *,
    playground: Any,
    task: WorkerTask,
    result: dict[str, Any],
    worker_index: int,
    worker_submission_dir: Path,
    submission_dir: Path,
    search_mgr: UCTSearchManager,
    best_state: BestState,
    results: dict[str, list[dict[str, Any]]],
    lock: threading.Lock,
    workspace: Path,
) -> None:
    """Execute finalize work item.

    Args:
        playground: Playground instance.
        task: Value for task.
        result: Value for result.
        worker_index: Numeric control parameter.
        worker_submission_dir: Directory path.
        submission_dir: Directory path.
        search_mgr: UCT search manager.
        best_state: Shared best-state tracker.
        results: Value for results.
        lock: Value for lock.
        workspace: Workspace path.

    Returns:
        None.
    """
    with lock:
        best_state.active_jobs = max(best_state.active_jobs - 1, 0)

        task.node.code = result.get("code", "")
        task.node.plan = result.get("plan", "")
        task.node.stdout = result.get("exec", {}).get("stdout", "")
        task.node.exit_code = result.get("exec", {}).get("exit_code", None)

        copied_submission = copy_submission(
            submission_dir,
            task.node.id,
            source_submission_dir=worker_submission_dir,
        )

        review = build_review(result, has_submission=copied_submission is not None)
        reward = search_mgr.ingest_result(task.node, review)

        save_node_snapshot(
            playground.run_dir,
            Path(playground.session.config.workspace_path),
            task.node,
            copied_submission,
            review,
            reward,
            search_mgr,
        )

        trail = {
            "ts": datetime.utcnow().isoformat(),
            "step": search_mgr.current_step,
            "stage": task.stage,
            "node_id": task.node.id,
            "parent": getattr(task.node.parent, "id", None),
            "is_buggy": task.node.is_buggy,
            "metric": getattr(task.node.metric, "value", None),
            "has_submission": copied_submission is not None,
            "submission_file": str(copied_submission) if copied_submission else None,
            "worker_index": worker_index,
        }
        append_trajectory(playground, trail, logger=playground.logger)
        results[task.stage].append(result)

        if (
            search_mgr.best_node
            and search_mgr.best_node.id != best_state.node_id
            and search_mgr.best_node.metric.value is not None
        ):
            best_state.node_id = search_mgr.best_node.id
            best_state.metric = search_mgr.best_node.metric.value
            best_state.code = search_mgr.best_node.code

            best_submission = submission_dir / f"submission_{best_state.node_id}.csv"
            save_best(
                playground.logger,
                workspace,
                str(best_state.code or ""),
                best_submission if best_submission.exists() else copied_submission,
            )


def shutdown_grading_server(logger: logging.Logger) -> None:
    """Execute shutdown grading server.

    Args:
        logger: Logger instance.

    Returns:
        None.
    """
    try:
        shutdown_embedded_grading_server(timeout=5)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to shutdown embedded grading server: %s", exc)

