"""Search conditions: should_trigger_branch_fusion, is_branch_stagnant, is_globally_stagnant."""

import logging
import time

logger = logging.getLogger("MLEvolve")


def should_trigger_branch_fusion(agent) -> bool:
    """Whether to trigger multi-branch aggregation: time window, min branches with success, global stagnation, under max attempts."""
    if agent.fusion_draft_count >= agent.max_fusion_drafts:
        return False

    if not agent.search_start_time:
        return False

    scfg = agent.scfg
    elapsed_time = time.time() - agent.search_start_time
    if elapsed_time < scfg.fusion_min_time_hours * 3600 or elapsed_time > scfg.fusion_max_time_hours * 3600:
        return False

    successful_branches = [
        bid for bid, nodes in agent.branch_successful_nodes.items()
        if len(nodes) >= scfg.fusion_min_successful_nodes
    ]
    if len(successful_branches) < scfg.fusion_min_branches:
        return False

    if not is_globally_stagnant(agent):
        return False

    logger.info(
        f"Branch fusion conditions met at {elapsed_time/3600:.1f}h "
        f"with {len(successful_branches)} successful branches"
    )
    return True


def is_branch_stagnant(agent, branch_id: int, threshold: int = 3) -> bool:
    """True if branch has no improvement over branch best for the last threshold attempts."""
    if branch_id not in agent.branch_successful_nodes:
        return False

    successful_nodes = agent.branch_successful_nodes[branch_id]
    if len(successful_nodes) < 1:
        return False

    maximize = agent.metric_maximize if agent.metric_maximize is not None else True

    sorted_nodes = sorted(
        successful_nodes,
        key=lambda n: n.metric.value if n.metric and n.metric.value is not None else (
            float('-inf') if maximize else float('inf')),
        reverse=maximize
    )

    branch_best_metric = sorted_nodes[0].metric.value
    if branch_best_metric is None:
        return False

    consecutive_no_improvement = 0
    max_consecutive = threshold

    recent_nodes = successful_nodes[-max_consecutive:] if len(
        successful_nodes) >= max_consecutive else successful_nodes

    for node in recent_nodes:
        if node.metric and node.metric.value is not None:
            if maximize:
                if node.metric.value >= branch_best_metric:
                    break
            else:
                if node.metric.value <= branch_best_metric:
                    break
            consecutive_no_improvement += 1

    if consecutive_no_improvement >= len(recent_nodes) and len(recent_nodes) >= 2:
        logger.info(
            f"Branch {branch_id} stagnant: {consecutive_no_improvement} consecutive attempts "
            f"didn't exceed branch best {branch_best_metric}")
        return True

    return False


def is_globally_stagnant(agent) -> bool:
    """True if no significant improvement in the last window_size nodes."""
    if not agent.best_node or not agent.best_node.metric:
        return False

    window_size = agent.stagnation_threshold

    if len(agent.journal.nodes) < window_size:
        return False

    recent_nodes = agent.journal.nodes[-window_size:]
    current_best_metric = agent.best_node.metric

    for node in recent_nodes:
        if node.is_buggy is False and node.metric and node.metric.value is not None:
            if agent.metric_maximize:
                improvement = node.metric.value - current_best_metric.value
            else:
                improvement = current_best_metric.value - node.metric.value

            if improvement > agent.scfg.metric_improvement_threshold:
                return False

    logger.info(f"Global stagnation detected: no improvement beyond threshold in last {window_size} nodes")
    return True
