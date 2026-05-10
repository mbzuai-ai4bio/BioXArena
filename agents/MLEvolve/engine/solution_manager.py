"""Top-K candidate management and result persistence (update_top_candidates, save_top_candidates, get_branch_top_nodes, save_best_solution, update_best_solution, write_metric_file)."""

import shutil
import logging
from collections import defaultdict
from typing import List

from engine.search_node import SearchNode
from utils.submission_artifacts import export_submission_bundle

logger = logging.getLogger("MLEvolve")


_STAGE_LABELS = {
    'fusion_draft': 'fusion_draft (multi-branch aggregation)',
    'draft': 'draft (initial solution)',
    'improve': 'improve (refinement)',
    'evolution': 'evolution (intra-branch evolution)',
    'fusion': 'fusion (cross-branch fusion)',
    'debug': 'debug (bug fixing)',
}


def format_stage_display(stage: str) -> str:
    """Map stage value to human-readable label."""
    return _STAGE_LABELS.get(stage, stage)


def write_metric_file(filepath, node, metric_maximize: bool) -> None:
    """Write metric.txt with metric value, maximize, branch_id, stage, from_topk, exec/created time."""
    with open(filepath, "w") as f:
        f.write(f"Metric: {node.metric.value}\n")
        f.write(f"Maximize: {metric_maximize}\n")

        if hasattr(node, 'branch_id') and node.branch_id is not None:
            f.write(f"Branch ID: {node.branch_id}\n")
        else:
            f.write(f"Branch ID: N/A\n")

        if hasattr(node, 'stage') and node.stage:
            f.write(f"Stage: {format_stage_display(node.stage)}\n")
        else:
            f.write(f"Stage: N/A\n")

        if hasattr(node, 'from_topk'):
            f.write(f"From Top-K: {node.from_topk}\n")
        else:
            f.write(f"From Top-K: False\n")

        if node.exec_time is not None:
            f.write(f"Execution Time(s): {node.exec_time:.2f}\n")
        else:
            f.write(f"Execution Time(s): N/A\n")

        if hasattr(node, 'created_time') and node.created_time:
            f.write(f"Created Time: {node.created_time}\n")
        else:
            f.write(f"Created Time: N/A\n")


def save_best_solution(agent, result_node, submission_file_path) -> None:
    """Save best solution code, submission, and meta to disk (thread-safe via agent.save_node_lock)."""
    best_solution_dir = agent.cfg.workspace_dir / "best_solution"
    best_submission_dir = agent.cfg.workspace_dir / "best_submission"

    with agent.save_node_lock:
        best_solution_dir.mkdir(exist_ok=True, parents=True)
        if best_submission_dir.exists():
            shutil.rmtree(best_submission_dir)
        best_submission_dir.mkdir(exist_ok=True, parents=True)

        export_submission_bundle(
            source_submission_csv=submission_file_path,
            destination_dir=best_submission_dir,
            submission_filename="submission.csv",
            node_id=result_node.id,
        )

        with open(best_solution_dir / "solution.py", "w") as f:
            f.write(result_node.code)

        with open(best_solution_dir / "node_id.txt", "w") as f:
            f.write(str(result_node.id))

        write_metric_file(
            best_solution_dir / "metric.txt",
            result_node,
            agent.metric_maximize,
        )


def update_top_candidates(agent, new_node: SearchNode) -> None:
    """Maintain a top-N list of best candidates by metric (higher is better if maximize else lower).
    Only consider nodes that are not buggy and have a valid metric value.
    Each branch contributes at most 5 candidates to ensure diversity.
    """
    if not new_node or new_node.is_buggy or not new_node.metric or new_node.metric.value is None or new_node.is_valid is False:
        return

    # Avoid duplicates (by node id)
    existing_ids = {n.id for n in agent.top_candidates}
    if new_node.id not in existing_ids:
        agent.top_candidates.append(new_node)

    if agent.metric_maximize is None:
        logger.warning("metric_maximize not initialized, using default value True")
        maximize = True
    else:
        maximize = agent.metric_maximize

    branch_nodes = defaultdict(list)

    for node in agent.top_candidates:
        branch_id = getattr(node, 'branch_id', None)
        if branch_id is None:
            branch_id = -1
        branch_nodes[branch_id].append(node)

    branch_top_nodes = []
    max_per_branch = 5

    for branch_id, nodes in branch_nodes.items():
        nodes.sort(
            key=lambda n: (
                n.metric.value
                if (n.metric and n.metric.value is not None)
                else (float('-inf') if maximize else float('inf'))
            ),
            reverse=maximize
        )
        branch_top_nodes.extend(nodes[:max_per_branch])

    branch_top_nodes.sort(
        key=lambda n: (
            n.metric.value
            if (n.metric and n.metric.value is not None)
            else (float('-inf') if maximize else float('inf'))
        ),
        reverse=maximize
    )

    agent.top_candidates = branch_top_nodes[:agent.top_k]


def save_top_candidates(agent) -> None:
    """Persist top-N candidates' code and submissions into workspace directories for offline inspection.
    All top-N files are organized under a single 'top_solution/' directory for better organization.
    Does not change best_node logic. Thread-safe with save_node_lock.
    """
    with agent.save_node_lock:
        top_solution_dir = agent.cfg.workspace_dir / "top_solution"
        top_solution_dir.mkdir(exist_ok=True, parents=True)

        for rank, node in enumerate(agent.top_candidates, start=1):
            rank_dir = top_solution_dir / f"top{rank}"
            if rank_dir.exists():
                shutil.rmtree(rank_dir)
            rank_dir.mkdir(exist_ok=True, parents=True)

            # Save code and meta
            try:
                with open(rank_dir / "solution.py", "w") as f:
                    f.write(node.code)
                with open(rank_dir / "node_id.txt", "w") as f:
                    f.write(str(node.id))
                write_metric_file(
                    rank_dir / "metric.txt",
                    node,
                    agent.metric_maximize,
                )
            except Exception as e:
                logger.error(f"Failed to save top{rank} solution files for node {node.id}: {e}")

            # Copy submission to the same directory
            submission_file_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"
            target_submission_path = rank_dir / "submission.csv"

            if submission_file_path.exists():
                try:
                    export_submission_bundle(
                        source_submission_csv=submission_file_path,
                        destination_dir=rank_dir,
                        submission_filename="submission.csv",
                        node_id=node.id,
                    )
                    logger.info(f"Saved top{rank} submission for node {node.id}")
                except Exception as e:
                    logger.error(f"Failed to copy top{rank} submission for node {node.id}: {e}")
            else:
                # Best-effort search for alternative matching file
                submission_dir = agent.cfg.workspace_dir / "submission"
                if submission_dir.exists():
                    for file in submission_dir.iterdir():
                        if node.id in file.name and file.name.endswith('.csv'):
                            try:
                                export_submission_bundle(
                                    source_submission_csv=file,
                                    destination_dir=rank_dir,
                                    submission_filename="submission.csv",
                                    node_id=node.id,
                                )
                                logger.info(
                                    f"Found alternative submission for top{rank} node {node.id}: {file.name}")
                            except Exception as e:
                                logger.error(f"Failed to copy alternative submission for node {node.id}: {e}")
                            break


def get_branch_top_nodes(agent, branch_id: int, top_k: int = 3) -> List[SearchNode]:
    """Return top-k nodes for a branch, sorted by metric."""
    if branch_id not in agent.branch_successful_nodes:
        logger.info(f"Branch {branch_id} has no successful nodes")
        return []

    successful_nodes = agent.branch_successful_nodes[branch_id]

    if not successful_nodes:
        logger.info(f"Branch {branch_id} has no successful nodes")
        return []

    maximize = agent.metric_maximize if agent.metric_maximize is not None else True

    sorted_nodes = sorted(
        successful_nodes,
        key=lambda n: n.metric.value if n.metric and n.metric.value is not None else (
            float('-inf') if maximize else float('inf')),
        reverse=maximize
    )

    result = sorted_nodes[:top_k]

    logger.info(f"Branch {branch_id}: found {len(successful_nodes)} successful nodes, returning top {len(result)}")
    for i, node in enumerate(result):
        logger.debug(f"  Top {i + 1}: Node {node.id}, Metric: {node.metric.value}")

    return result


def update_best_solution(agent, node):
    """Update top-K candidates and global best node."""
    if not node.metric or node.metric.value is None:
        return

    submission_file_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"

    update_top_candidates(agent, node)
    save_top_candidates(agent)

    if agent.best_node is None or agent.best_node.metric < node.metric:
        if agent.best_node is None or node.is_valid is True:
            agent.best_node = node
            save_best_solution(agent, node, submission_file_path)
            logger.info(f"[best] updated: node {node.id}, metric={node.metric.value}")
        else:
            logger.debug(f"Node {node.id} is invalid, skipped")
    else:
        if agent.best_node.is_valid is False:
            agent.best_node = node
            save_best_solution(agent, node, submission_file_path)
            logger.info(f"[best] updated: node {node.id}, metric={node.metric.value}")
        else:
            logger.debug(f"Node {node.id} not the best (current best: {agent.best_node.id})")
