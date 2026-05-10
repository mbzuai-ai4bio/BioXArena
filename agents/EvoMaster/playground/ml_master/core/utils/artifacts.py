from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from .uct import MetricReview, UCTSearchManager


def copy_submission(
    submission_dir: Path,
    node_id: str,
    source_submission_dir: Path | None = None,
) -> Path | None:
    """Copy submission.csv into node-specific snapshot file."""
    submission_dir.mkdir(parents=True, exist_ok=True)
    src_dir = source_submission_dir or submission_dir
    src = src_dir / "submission.csv"
    if not src.exists():
        return None

    dst = submission_dir / f"submission_{node_id}.csv"
    shutil.copy(src, dst)

    try:
        src.unlink()
    except Exception:
        pass

    return dst


def build_review(result: dict[str, Any], has_submission: bool) -> MetricReview:
    """Convert stage output to a UCT MetricReview."""
    metric_detail = result.get("metric_detail", {}) or {}
    metric = result.get("metric")
    return MetricReview(
        metric=metric,
        lower_is_better=metric_detail.get("lower_is_better"),
        is_bug=metric_detail.get("is_bug", False) or metric is None,
        has_submission=has_submission,
        summary=(result.get("exec", {}).get("stdout", "") or "")[-500:],
        raw_output=result.get("raw_response"),
    )


def append_trajectory(playground: Any, record: dict[str, Any], logger: logging.Logger | None = None) -> None:
    """Append trajectory record and persist when run_dir is available."""
    logger = logger or logging.getLogger(__name__)
    if not hasattr(playground, "trajectories") or playground.trajectories is None:
        playground.trajectories = []

    playground.trajectories.append(record)

    run_dir = getattr(playground, "run_dir", None)
    if not run_dir:
        return

    trajectory_dir = Path(run_dir) / "trajectories"
    task_id = getattr(playground, "task_id", None)
    if task_id:
        trajectory_dir = trajectory_dir / task_id
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    trajectory_file = trajectory_dir / "trajectory.jsonl"
    with trajectory_file.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    logger.debug("trajectory appended: %s", trajectory_file)


def save_node_snapshot(
    run_dir: str | Path | None,
    workspace_path: Path,
    node: Any,
    submission_path: Path | None,
    review: MetricReview,
    reward: float,
    search_mgr: UCTSearchManager,
) -> None:
    """Persist key node state for later inspection and visualization."""
    base_dir = Path(run_dir) / "logs" / "uct_nodes" if run_dir else workspace_path / "logs" / "uct_nodes"
    base_dir.mkdir(parents=True, exist_ok=True)

    parent_visits = node.parent.visits if node.parent else 1
    try:
        uct_val = node.uct_value(search_mgr._exploration_constant(), parent_visits)
    except Exception:
        uct_val = None

    snapshot = {
        "id": node.id,
        "stage": node.stage,
        "parent": getattr(node.parent, "id", None),
        "metric": getattr(node.metric, "value", None),
        "maximize": getattr(node.metric, "maximize", True) if getattr(node, "metric", None) else None,
        "is_buggy": node.is_buggy,
        "has_submission": review.has_submission,
        "reward": reward,
        "visits": node.visits,
        "total_reward": node.total_reward,
        "uct_value": uct_val,
        "submission_file": str(submission_path) if submission_path else None,
        "code": getattr(node, "code", "") or "",
        "stdout": getattr(node, "stdout", "") or "",
        "initial_reward": getattr(node, "initial_reward", None),
        "initial_total_reward": getattr(node, "initial_total_reward", None),
        "initial_visits": getattr(node, "initial_visits", None),
        "initial_uct": getattr(node, "initial_uct", None),
    }

    snapshot_path = base_dir / f"{node.id}.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def save_best(logger: logging.Logger, workspace: Path, best_code: str, submission_csv: Path | None) -> None:
    """Persist current best code and submission snapshot."""
    best_solution_path = workspace / "best_solution" / "best_solution.py"
    best_solution_path.write_text(best_code, encoding="utf-8")

    best_submission_path = workspace / "best_submission" / "best_submission.csv"
    if submission_csv is not None and submission_csv.exists():
        shutil.copy(submission_csv, best_submission_path)
    else:
        logger.debug("No submission csv to save as best (None or missing).")

    logger.info("Saved best solution: %s", best_solution_path)
    logger.info("Saved best submission: %s", best_submission_path)

