"""SearchExp: includes the plan and search agents; runs at least two Plan → Search rounds; if both rounds are empty, relax the threshold and run one more search."""

import logging
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance

from ..utils.rag_utils import (
    parse_plan_output,
    extract_agent_response,
    update_agent_format_kwargs,
)

DEFAULT_QUERY = "Summarize the following machine learning task in one complete English sentence."
RELAXED_THRESHOLD = 2.0  # When results from multiple rounds are all empty, retry with a relaxed threshold.


def _is_result_empty(text: str) -> bool:
    """Determine whether a search result should be treated as empty (no valid content). Use node_id, similarity, and prepare_code as validity markers."""
    if not text or not text.strip():
        return True
    stripped = text.strip().lower()
    if len(stripped) < 30:
        return True
    has_valid_marker = (
        "node_id" in stripped
        and ("similarity" in stripped or "prepare_code" in stripped)
    )
    if not has_valid_marker:
        return True
    return False


class SearchExp(BaseExp):
    def __init__(self, plan_agent, search_agent, config):
        super().__init__(plan_agent, config)
        self.plan_agent = plan_agent
        self.search_agent = search_agent
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(
        self,
        task_description: str,
        analyze_output: str,
        db: dict,
        task_id: str = "exp_001",
    ) -> tuple[str, list]:
        """Run two Plan → Search rounds and return (combined_search_results, [trajectories])."""
        self.logger.info("Starting SearchExp (plan + search, 2 rounds)")
        trajectories = []

        # ---------- Round 1: Plan (initial) ----------
        stage_input = analyze_output or "(no Analyze output)"
        update_agent_format_kwargs(
            self.plan_agent,
            task_description=task_description,
            stage_input=stage_input,
            **db,
        )
        plan_task_1 = TaskInstance(
            task_id=f"{task_id}_plan1",
            task_type="plan",
            description=task_description,
            input_data={},
        )
        plan_traj_1 = self.plan_agent.run(plan_task_1)
        trajectories.append(plan_traj_1)
        plan_output_1 = extract_agent_response(plan_traj_1)

        # ---------- Round 1: Search ----------
        params1 = parse_plan_output(plan_output_1)
        if not params1.get("query"):
            params1["query"] = DEFAULT_QUERY
        # Also pass the full Plan output to Search so it can follow the multi-round retrieval strategy.
        update_agent_format_kwargs(self.search_agent, plan_output=plan_output_1, **params1, **db)
        search_task_1 = TaskInstance(
            task_id=f"{task_id}_search1",
            task_type="search",
            description=task_description,
            input_data={},
        )
        search_traj_1 = self.search_agent.run(search_task_1)
        trajectories.append(search_traj_1)
        search_results_1 = extract_agent_response(search_traj_1)

        # ---------- Round 2: Plan (second params) ----------
        first_round_empty = _is_result_empty(search_results_1 or "")
        # For the second-round plan, provide both:
        # (1) the Analyze output (query-writing guidelines, DB description), and
        # (2) the first-round search results, so the planner can refine queries with full context.
        stage_input_2_parts = [
            "Analyze output:\n",
            (analyze_output or "(none)"),
            "\n\nFirst-round search results:\n",
            (search_results_1 or "(none)"),
        ]
        stage_input_2 = "".join(stage_input_2_parts)
        update_agent_format_kwargs(
            self.plan_agent,
            task_description=task_description,
            stage_input=stage_input_2,
            **db,
        )
        plan_task_2 = TaskInstance(
            task_id=f"{task_id}_plan2",
            task_type="plan",
            description=task_description,
            input_data={},
        )
        plan_traj_2 = self.plan_agent.run(plan_task_2)
        trajectories.append(plan_traj_2)
        plan_output_2 = extract_agent_response(plan_traj_2)

        # ---------- Round 2: Search ----------
        params2 = parse_plan_output(plan_output_2)
        if not params2.get("query"):
            params2 = params1
        # For the second round, also pass the corresponding Plan output so Search can use it as strategy and context.
        update_agent_format_kwargs(self.search_agent, plan_output=plan_output_2, **params2, **db)
        search_task_2 = TaskInstance(
            task_id=f"{task_id}_search2",
            task_type="search",
            description=task_description,
            input_data={},
        )
        search_traj_2 = self.search_agent.run(search_task_2)
        trajectories.append(search_traj_2)
        search_results_2 = extract_agent_response(search_traj_2)

        # Hard requirement: if results from multiple rounds are all empty, relax the threshold and retry one more round.
        search_results_3 = None
        if _is_result_empty(search_results_1 or "") and _is_result_empty(search_results_2 or ""):
            self.logger.info("Both rounds empty; retrying with relaxed threshold")
            params_retry = {**params2, "threshold": max(RELAXED_THRESHOLD, params2.get("threshold", 1.5) * 1.2)}
            # During retry, keep using the second-round Plan output as the strategy reference.
            update_agent_format_kwargs(self.search_agent, plan_output=plan_output_2, **params_retry, **db)
            search_task_3 = TaskInstance(
                task_id=f"{task_id}_search_retry",
                task_type="search",
                description=task_description,
                input_data={},
            )
            search_traj_3 = self.search_agent.run(search_task_3)
            trajectories.append(search_traj_3)
            search_results_3 = extract_agent_response(search_traj_3)

        combined = (search_results_1 or "") + "\n\n--- Second round ---\n\n" + (search_results_2 or "")
        if search_results_3:
            combined += "\n\n--- Retry with relaxed threshold ---\n\n" + (search_results_3 or "")
        self.logger.info("SearchExp completed")
        return combined, trajectories
