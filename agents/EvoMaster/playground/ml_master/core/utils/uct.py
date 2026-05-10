from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

StageLiteral = Literal["root", "draft", "debug", "improve"]


@dataclass
class MetricReview:
    metric: Optional[float]
    lower_is_better: Optional[bool] = None
    maximize: bool = True
    is_bug: bool = False
    has_submission: bool = True
    summary: str = ""
    raw_output: Optional[str] = None

    def __post_init__(self) -> None:
        """Normalize fields after initialization.

        Returns:
            None.
        """
        if self.lower_is_better is not None:
            self.maximize = not self.lower_is_better
        if self.metric is not None:
            self.metric = float(self.metric)


@dataclass
class MetricValue:
    value: Optional[float]
    maximize: bool = True

    def __post_init__(self) -> None:
        """Normalize fields after initialization.

        Returns:
            None.
        """
        if self.value is not None:
            self.value = float(self.value)

    def __gt__(self, other: "MetricValue") -> bool:  # type: ignore[override]
        """Compare this value with another value.

        Args:
            other: Value for other.

        Returns:
            bool: Result of this function.
        """
        if self.value is None:
            return False
        if other.value is None:
            return True
        if self.value == other.value:
            return False
        comparison = self.value > other.value
        return comparison if self.maximize else not comparison


class WorstMetricValue(MetricValue):
    def __init__(self) -> None:
        """Initialize WorstMetricValue.

        Returns:
            None.
        """
        super().__init__(value=None, maximize=True)


@dataclass
class UCTSearchConfig:
    back_debug_depth: int = 3
    max_debug_depth: int = 20
    num_drafts: int = 5
    num_bugs: int = 1
    num_improves: int = 3
    invalid_metric_upper_bound: int = 100
    metric_improvement_threshold: float = 0.0001
    max_improve_failure: int = 3


@dataclass
class UCTDecayConfig:
    decay_type: Literal[
        "constant",
        "linear",
        "exponential",
        "piecewise",
        "dynamic_piecewise",
    ] = "piecewise"
    exploration_constant: float = 1.414
    lower_bound: float = 0.5

    linear_alpha: float = 0.01
    exponential_gamma: float = 0.99
    piecewise_alpha: float = 0.01
    piecewise_phase_ratios: tuple[float, float] = (0.3, 0.7)
    dynamic_alpha: float = 0.01
    dynamic_phase_ratios: tuple[float, float] = (0.85, 1.0)


def _linear_decay(t: int, initial_c: float, alpha: float, lower_bound: float) -> float:
    """Execute linear decay.

    Args:
        t: Value for t.
        initial_c: Value for initial c.
        alpha: Value for alpha.
        lower_bound: Value for lower bound.

    Returns:
        float: Result of this function.
    """
    return max(initial_c - alpha * t, lower_bound)


def _exponential_decay(t: int, initial_c: float, gamma: float, lower_bound: float) -> float:
    """Execute exponential decay.

    Args:
        t: Value for t.
        initial_c: Value for initial c.
        gamma: Value for gamma.
        lower_bound: Value for lower bound.

    Returns:
        float: Result of this function.
    """
    return max(initial_c * (gamma**t), lower_bound)


def _piecewise_decay(
    t: int,
    initial_c: float,
    t1: int,
    t2: int,
    alpha: float,
    lower_bound: float,
) -> float:
    """Execute piecewise decay.

    Args:
        t: Value for t.
        initial_c: Value for initial c.
        t1: Value for t1.
        t2: Value for t2.
        alpha: Value for alpha.
        lower_bound: Value for lower bound.

    Returns:
        float: Result of this function.
    """
    if t < t1:
        return initial_c
    if t <= t2:
        return max(initial_c - alpha * (t - t1), lower_bound)
    return lower_bound


def _dynamic_piecewise_decay(
    *,
    steps_limit: int,
    n_nodes: int,
    initial_c: float,
    start_time: float,
    time_limit: float,
    alpha: float,
    lower_bound: float,
    phase_ratios: tuple[float, float],
) -> float:
    """Execute dynamic piecewise decay.

    Args:
        steps_limit: Value for steps limit.
        n_nodes: Node-related object.
        initial_c: Value for initial c.
        start_time: Value for start time.
        time_limit: Value for time limit.
        alpha: Value for alpha.
        lower_bound: Value for lower bound.
        phase_ratios: Value for phase ratios.

    Returns:
        float: Result of this function.
    """
    now = time.time()
    elapsed = max(now - start_time, 1e-6)
    remaining = max(time_limit - elapsed, 1e-6)

    speed = n_nodes / elapsed
    n_remaining = round(speed * remaining)
    estimated_total = min(n_nodes + n_remaining, steps_limit)
    progress = n_nodes / estimated_total if estimated_total > 0 else 0.0

    p1, p2 = phase_ratios
    if progress < p1:
        return initial_c
    if progress < p2:
        decay_len = p2 - p1
        decay_progress = (progress - p1) / decay_len if decay_len > 0 else 0.0
        c_value = initial_c - alpha * decay_progress * estimated_total
        return max(c_value, lower_bound)
    return lower_bound


@dataclass(eq=False)
class UCTNode:
    stage: StageLiteral
    plan: str = ""
    code: str = ""
    stdout: Optional[str] = None
    exit_code: Optional[int] = None
    parent: Optional["UCTNode"] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)

    analysis: Optional[str] = None
    metric: MetricValue = field(default_factory=WorstMetricValue)
    is_buggy: Optional[bool] = None
    is_valid: Optional[bool] = None
    is_terminal: bool = False
    finish_time: Optional[float] = None
    is_debug_success: bool = False
    continue_improve: bool = False
    improve_failure_depth: int = 0
    local_best_node: Optional["UCTNode"] = None

    visits: int = 0
    total_reward: float = 0.0
    children: set["UCTNode"] = field(default_factory=set)
    expected_child_count: int = 0
    locked: bool = False

    initial_reward: float | None = None
    initial_total_reward: float | None = None
    initial_visits: int | None = None
    initial_uct: float | None = None

    def __post_init__(self) -> None:
        """Normalize fields after initialization.

        Returns:
            None.
        """
        if self.parent is not None:
            self.parent.children.add(self)

    def __hash__(self) -> int:
        """Return the hash value for this instance.

        Returns:
            int: Result of this function.
        """
        return hash(self.id)

    @property
    def num_children(self) -> int:
        """Execute num children.

        Returns:
            int: Result of this function.
        """
        return len(self.children)

    @property
    def debug_depth(self) -> int:
        """Execute debug depth.

        Returns:
            int: Result of this function.
        """
        if self.stage != "debug" or self.parent is None:
            return 0
        return 1 + self.parent.debug_depth

    def expect_child(self) -> None:
        """Execute expect child.

        Returns:
            None.
        """
        self.expected_child_count += 1

    def complete_child(self) -> None:
        """Execute complete child.

        Returns:
            None.
        """
        self.expected_child_count = max(self.expected_child_count - 1, 0)

    def is_fully_expanded(self, cfg: UCTSearchConfig) -> bool:
        """Check whether fully expanded.

        Args:
            cfg: Configuration dictionary.

        Returns:
            bool: Result of this function.
        """
        if self.stage == "root":
            return self.expected_child_count >= cfg.num_drafts
        if self.is_buggy:
            if any(child.is_buggy is False for child in self.children):
                return True
            return self.expected_child_count >= cfg.num_bugs
        return self.expected_child_count >= cfg.num_improves

    def uct_value(self, exploration_constant: float, parent_visits: int) -> float:
        """Execute uct value.

        Args:
            exploration_constant: Value for exploration constant.
            parent_visits: Value for parent visits.

        Returns:
            float: Result of this function.
        """
        if self.visits == 0:
            return float("inf")
        parent_total = max(parent_visits, 1)
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(math.log(parent_total) / self.visits)
        return exploitation + exploration

    def update_reward(self, reward: float) -> None:
        """Execute update reward.

        Args:
            reward: Value for reward.

        Returns:
            None.
        """
        self.visits += 1
        self.total_reward += reward

    def fetch_child_memory(self, include_code: bool = False) -> str:
        """Execute fetch child memory.

        Args:
            include_code: Value for include code.

        Returns:
            str: Result of this function.
        """
        summary: list[str] = []
        children = sorted(self.children, key=lambda child: child.created_at)
        for child in children:
            if child.is_buggy is None:
                continue
            part = f"Design: {child.plan}\n"
            if include_code:
                part += f"Code: {child.code}\n"
            if child.is_buggy:
                part += "Results: The implementation of this design has bugs.\n"
                part += "Insight: Using a different approach may not result in the same bugs as the above approach.\n"
            else:
                if child.analysis:
                    part += f"Results: {child.analysis}\n"
                if child.metric:
                    part += f"Validation Metric: {child.metric.value}\n"
            summary.append(part)

        if not summary:
            return "There is no previous memory"
        return "\n-------------------------------\n".join(summary)

    def fetch_parent_memory(self, include_code: bool = False) -> str:
        """Execute fetch parent memory.

        Args:
            include_code: Value for include code.

        Returns:
            str: Result of this function.
        """
        if self.parent and self.parent.is_buggy is False:
            part = f"Design: {self.parent.plan}\n"
            if include_code:
                part += f"Code: {self.parent.code}\n"
            if self.parent.analysis:
                part += f"Results: {self.parent.analysis}\n"
            if self.parent.metric:
                part += f"Validation Metric: {self.parent.metric.value}\n"
            return part
        return ""


MetricParser = Callable[[str, str, Optional[str]], MetricReview]


class UCTSearchManager:
    def __init__(
        self,
        search_cfg: UCTSearchConfig,
        decay_cfg: UCTDecayConfig,
        *,
        time_limit: float = 0,
        grader: Optional[Callable[[str, Path], Tuple[bool, dict | str]]] = None,
        exp_id: Optional[str] = None,
        submission_dir: Optional[Path | str] = None,
    ) -> None:
        """Initialize UCTSearchManager.

        Args:
            search_cfg: Configuration value.
            decay_cfg: Configuration value.
            time_limit: Value for time limit.
            grader: Value for grader.
            exp_id: Identifier string.
            submission_dir: Directory path.

        Returns:
            None.
        """
        self.search_cfg = search_cfg
        self.decay_cfg = decay_cfg
        self.time_limit = time_limit
        self.grader = grader
        self.exp_id = exp_id
        self.submission_dir = Path(submission_dir) if submission_dir else None

        self.root = UCTNode(stage="root", plan="virtual root", code="")
        self.best_node: Optional[UCTNode] = None
        self.best_metric: Optional[float] = None

        self.current_step = 0
        self.search_start_time = time.time()
        self.snapshot_fn: Optional[Callable[[UCTNode, Optional[Path], MetricReview, float], None]] = None

    def set_snapshot_fn(
        self,
        fn: Callable[[UCTNode, Optional[Path], MetricReview, float], None],
    ) -> None:
        """Set snapshot fn.

        Args:
            fn: Value for fn.

        Returns:
            None.
        """
        self.snapshot_fn = fn

    def create_child(
        self,
        parent: UCTNode,
        stage: StageLiteral,
        plan: str = "",
        code: str = "",
    ) -> UCTNode:
        """Create child.

        Args:
            parent: Parent UCT node.
            stage: Value for stage.
            plan: Value for plan.
            code: Generated Python code string.

        Returns:
            UCTNode: Result of this function.
        """
        parent.expect_child()
        child = UCTNode(stage=stage, plan=plan, code=code, parent=parent)
        logger.info("Created child node %s stage=%s parent=%s", child.id, stage, parent.id)
        return child

    def select_next(self, node: Optional[UCTNode] = None) -> Optional[UCTNode]:
        """Select next.

        Args:
            node: UCT node object.

        Returns:
            Optional[UCTNode]: Result of this function.
        """
        selected = node or self.root
        while selected and not selected.is_terminal:
            if not selected.is_fully_expanded(self.search_cfg):
                if selected.is_buggy and selected.is_debug_success:
                    selected = self._uct_select(selected)
                    continue
                if selected.continue_improve and selected.children:
                    selected = self._uct_select(selected)
                    continue
                return selected
            selected = self._uct_select(selected)
        return selected

    def ingest_result(
        self,
        node: UCTNode,
        review: MetricReview,
        *,
        debug_budget_exhausted: bool = False,
    ) -> float:
        """Execute ingest result.

        Args:
            node: UCT node object.
            review: Value for review.
            debug_budget_exhausted: Value for debug budget exhausted.

        Returns:
            float: Result of this function.
        """
        node.finish_time = time.time()
        node.analysis = review.summary
        node.is_buggy = review.is_bug or review.metric is None or not review.has_submission
        node.is_valid = not node.is_buggy
        node.metric = WorstMetricValue() if node.is_buggy else MetricValue(review.metric, maximize=review.maximize)
        node.continue_improve = not node.is_buggy and node.metric.value is not None

        self._apply_metric_direction_guard(node)
        self._apply_debug_success(node)

        if debug_budget_exhausted and node.stage == "debug":
            node.is_terminal = True

        self._apply_grading_guard(node)

        if not node.is_buggy and not self._check_metric_valid(node):
            node.is_buggy = True
            node.is_valid = False
            node.metric = WorstMetricValue()
            node.analysis = f"{node.analysis or ''}\n[metric] invalid metric detected".strip()

        self._check_improvement(node)

        reward = self._get_node_reward(node)
        self._backpropagate(node, reward)
        self._record_initial_stats(node, reward)
        self._emit_snapshot(node, review, reward)

        self.current_step += 1
        logger.info(
            "Ingested node=%s stage=%s buggy=%s metric=%s reward=%.3f",
            node.id,
            node.stage,
            node.is_buggy,
            node.metric.value,
            reward,
        )
        return reward

    def _apply_metric_direction_guard(self, node: UCTNode) -> None:
        """Execute apply metric direction guard.

        Args:
            node: UCT node object.

        Returns:
            None.
        """
        if node.is_buggy:
            return
        if not self.best_node or not self.best_node.metric:
            return
        if node.metric.maximize == self.best_node.metric.maximize:
            return

        node.metric = WorstMetricValue()
        node.is_buggy = True
        node.is_valid = False
        node.analysis = f"{node.analysis or ''}\n[metric] direction mismatch with best node".strip()

    def _apply_debug_success(self, node: UCTNode) -> None:
        """Execute apply debug success.

        Args:
            node: UCT node object.

        Returns:
            None.
        """
        if node.parent and node.parent.is_buggy and node.is_buggy is False:
            node.parent.is_debug_success = True
            node.parent.is_buggy = False

        if node.parent and node.parent.stage != "root":
            node.parent.continue_improve = node.continue_improve

    def _apply_grading_guard(self, node: UCTNode) -> None:
        """Execute apply grading guard.

        Args:
            node: UCT node object.

        Returns:
            None.
        """
        if not (self.grader and self.exp_id and self.submission_dir):
            return
        if node.is_buggy:
            return

        submission_path = self.submission_dir / f"submission_{node.id}.csv"
        if not submission_path.exists():
            return

        ok, result = self.grader(self.exp_id, submission_path)
        if ok:
            if isinstance(result, dict) and not result.get("is_valid", True):
                detail = result.get("result") or result.get("details") or "submission format invalid"
                node.is_buggy = True
                node.is_valid = False
                node.metric = WorstMetricValue()
                node.analysis = f"{node.analysis or ''}\n[grading] {detail}".strip()
            return

        node.is_buggy = True
        node.is_valid = False
        node.metric = WorstMetricValue()
        node.analysis = f"{node.analysis or ''}\n[grading] grading server call failed".strip()

    def _record_initial_stats(self, node: UCTNode, reward: float) -> None:
        """Execute record initial stats.

        Args:
            node: UCT node object.
            reward: Value for reward.

        Returns:
            None.
        """
        if node.initial_reward is not None:
            return
        node.initial_reward = reward
        node.initial_total_reward = node.total_reward
        node.initial_visits = node.visits
        try:
            parent_visits = node.parent.visits if node.parent else 1
            node.initial_uct = node.uct_value(self._exploration_constant(), parent_visits)
        except Exception:
            node.initial_uct = None

    def _emit_snapshot(self, node: UCTNode, review: MetricReview, reward: float) -> None:
        """Execute emit snapshot.

        Args:
            node: UCT node object.
            review: Value for review.
            reward: Value for reward.

        Returns:
            None.
        """
        if not self.snapshot_fn:
            return

        submission_path = None
        if self.submission_dir:
            candidate = self.submission_dir / f"submission_{node.id}.csv"
            if candidate.exists():
                submission_path = candidate

        current: UCTNode | None = node
        while current:
            sub = submission_path if current is node else None
            try:
                self.snapshot_fn(current, sub, review, reward)
            except Exception as exc:
                logger.warning("Snapshot callback failed for node %s: %s", current.id, exc)
            current = current.parent

    def _backpropagate(self, node: UCTNode, reward: float) -> None:
        """Execute backpropagate.

        Args:
            node: UCT node object.
            reward: Value for reward.

        Returns:
            None.
        """
        current: UCTNode | None = node
        while current is not None:
            if current.stage == "draft" and current.locked:
                current.locked = False
            current.update_reward(reward)
            current = current.parent

    def _get_node_reward(self, node: UCTNode) -> float:
        """Execute get node reward.

        Args:
            node: UCT node object.

        Returns:
            float: Result of this function.
        """
        if node.is_buggy or node.metric.value is None:
            return -1.0

        reward = 1.0

        parent = node.parent
        if parent and parent.is_buggy:
            reward += 1.0

        if (
            self.best_node
            and self.best_node.metric
            and self.best_metric is not None
            and self.best_node.metric.maximize == node.metric.maximize
        ):
            delta = node.metric.value - self.best_metric if node.metric.maximize else self.best_metric - node.metric.value
            if delta > 0:
                reward += 1.0

        if self._check_metric_valid(node):
            if self.best_metric is None:
                self.best_metric = node.metric.value
                self.best_node = node
            elif self.best_node and node.metric > self.best_node.metric:
                self.best_metric = node.metric.value
                self.best_node = node

        return reward

    def _check_metric_valid(self, node: UCTNode, upper_bound: int | None = None) -> bool:
        """Execute check metric valid.

        Args:
            node: UCT node object.
            upper_bound: Value for upper bound.

        Returns:
            bool: Result of this function.
        """
        bound = upper_bound or self.search_cfg.invalid_metric_upper_bound
        best = self.best_metric
        current = node.metric.value
        if best is None or current is None:
            return True
        if best == 0 or current == 0:
            return abs(best - current) <= bound
        ratio = max(abs(best), abs(current)) / min(abs(best), abs(current))
        return ratio <= bound

    def _check_improvement(self, node: UCTNode) -> None:
        """Execute check improvement.

        Args:
            node: UCT node object.

        Returns:
            None.
        """
        parent = node.parent
        local_best = node.local_best_node or (parent.local_best_node if parent else None) or parent
        cfg = self.search_cfg

        if node.is_buggy is False:
            new_metric = node.metric.value
            if parent and parent.is_buggy:
                node.continue_improve = False
                node.is_terminal = False
                return

            if new_metric is not None and local_best and local_best.metric.value is not None:
                improvement = (
                    new_metric - local_best.metric.value
                    if node.metric.maximize
                    else local_best.metric.value - new_metric
                )
                if improvement < cfg.metric_improvement_threshold:
                    if local_best.improve_failure_depth < cfg.max_improve_failure:
                        local_best.improve_failure_depth += 1
                        node.continue_improve = True
                    else:
                        node.continue_improve = False
                        node.is_terminal = True
                else:
                    node.local_best_node = node
                    node.continue_improve = True
                    local_best.improve_failure_depth = 0
            elif new_metric is not None:
                node.local_best_node = node
                node.continue_improve = True
            else:
                node.continue_improve = False
            return

        if node.is_buggy is True:
            if node.debug_depth >= cfg.back_debug_depth and node.debug_depth >= cfg.max_debug_depth:
                node.is_terminal = True
            return

        node.continue_improve = False

    def _uct_select(self, node: UCTNode) -> UCTNode:
        """Execute uct select.

        Args:
            node: UCT node object.

        Returns:
            UCTNode: Result of this function.
        """
        c_value = self._exploration_constant()

        if node.stage == "root":
            unlocked = [child for child in node.children if not child.locked]
            if not unlocked:
                return node
            picked = max(unlocked, key=lambda child: child.uct_value(c_value, node.visits))
            if picked.stage == "draft":
                picked.locked = True
            return picked

        if not node.children:
            return node
        return max(node.children, key=lambda child: child.uct_value(c_value, node.visits))

    def _exploration_constant(self) -> float:
        """Execute exploration constant.

        Returns:
            float: Result of this function.
        """
        cfg = self.decay_cfg
        t = self.current_step

        if cfg.decay_type == "linear":
            value = _linear_decay(t, cfg.exploration_constant, cfg.linear_alpha, cfg.lower_bound)
        elif cfg.decay_type == "exponential":
            value = _exponential_decay(t, cfg.exploration_constant, cfg.exponential_gamma, cfg.lower_bound)
        elif cfg.decay_type == "piecewise":
            t1 = round(cfg.piecewise_phase_ratios[0] * max(self.current_step, 1))
            t2 = round(cfg.piecewise_phase_ratios[1] * max(self.current_step, 1))
            value = _piecewise_decay(t, cfg.exploration_constant, t1, t2, cfg.piecewise_alpha, cfg.lower_bound)
        elif cfg.decay_type == "dynamic_piecewise":
            value = _dynamic_piecewise_decay(
                steps_limit=max(self.current_step, 1),
                n_nodes=self.current_step,
                initial_c=cfg.exploration_constant,
                start_time=self.search_start_time,
                time_limit=self.time_limit or 1e6,
                alpha=cfg.dynamic_alpha,
                lower_bound=cfg.lower_bound,
                phase_ratios=cfg.dynamic_phase_ratios,
            )
        else:
            value = cfg.exploration_constant

        return value
