"""Post-execution validation: validate_executed_node (csv existence, metric=0.0, register success)."""

import logging

from engine.search_node import SearchNode
from utils.metric import WorstMetricValue

logger = logging.getLogger("MLEvolve")

_ZERO_METRIC_ANALYSIS = (
    "Performance is 0.0 (complete failure). This indicates fundamental issues that need debugging:\n"
    "1. Model architecture may be incorrect or not learning\n"
    "2. Data preprocessing might be broken (wrong format, normalization issues)\n"
    "3. Loss function or evaluation metric calculation may be faulty\n"
    "4. Training loop might not be updating weights properly\n"
    "5. Input data might not be loaded correctly\n\n"
    "Please review the code carefully to identify the root cause."
)


def validate_executed_node(agent, node: SearchNode):
    """Check submission.csv exists, metric=0.0 anomaly; register successful node to branch."""
    if node.is_buggy:
        return

    submission_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"
    if not submission_path.exists():
        node.is_buggy = True
        node.metric = WorstMetricValue()
        logger.info(f"Node {node.id} did not produce a submission.csv")
        return

    if node.metric.maximize and node.metric.value == 0.0:
        node.is_buggy = True
        node.metric = WorstMetricValue()
        node.analysis = _ZERO_METRIC_ANALYSIS
        logger.warning(
            f"Node {node.id} has metric=0.0 (maximize=True), marking as buggy for debugging."
        )
        return

    if hasattr(node, 'branch_id') and node.branch_id:
        if node.branch_id not in agent.branch_successful_nodes:
            agent.branch_successful_nodes[node.branch_id] = []
        agent.branch_successful_nodes[node.branch_id].append(node)
