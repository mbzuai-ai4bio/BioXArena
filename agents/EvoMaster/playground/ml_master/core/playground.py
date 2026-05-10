from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from evomaster.core import BasePlayground, register_playground

from .utils.data_preview import generate as generate_data_preview
from .utils.mlebench_grade import grade_best_submission_and_save
from .utils.orchestrator import (
    BestState,
    acquire_work_item,
    create_search_manager,
    create_worker_agents,
    ensure_prepared_links,
    finalize_work_item,
    parallel_config,
    prepare_workspace,
    reset_working_dir,
    resolve_worker_workspace,
    run_one_node,
    shutdown_grading_server,
)


@register_playground("ml_master")
class MLMasterPlayground(BasePlayground):
    """UCT-driven orchestrator for draft/debug/improve workflow."""

    def __init__(self, config_dir: Path | None = None, config_path: Path | None = None):
        """Initialize MLMasterPlayground.

        Args:
            config_dir: Directory path.
            config_path: Filesystem path.

        Returns:
            None.
        """
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "ml_master"

        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("draft_agent", "debug_agent", "improve_agent", "metric_agent")
        self.trajectories: list[dict] = []
        self.mcp_manager = None

        cfg = parallel_config(self.config)
        self.max_workers = int(cfg.get("max_parallel", 1)) if cfg.get("enabled", False) else 1

    def setup(self) -> None:
        """Execute setup.

        Returns:
            None.
        """
        self.logger.info("Setting up MLMasterPlayground...")
        self._setup_session()
        self._setup_agents()

        required_slots = ["draft_agent", "debug_agent", "improve_agent", "metric_agent"]
        missing = [slot for slot in required_slots if self.agents.get(slot) is None]
        if missing:
            raise ValueError(f"config.agents missing required slots: {missing}")

        ensure_prepared_links(self.config, Path(self.session.config.workspace_path))

    def run(self, task_description: str, output_file: str | None = None) -> dict:
        """
        Simplest workflow:
        1) prepare workspace + UCT manager
        2) run parallel workers (select node -> run exp -> ingest result)
        3) aggregate and return stage results
        """
        try:
            self.setup()
            self._setup_trajectory_file(output_file)

            # Step 1: prepare search context.
            workspace = Path(self.session.config.workspace_path)
            submission_dir = prepare_workspace(workspace)
            search_mgr = create_search_manager(
                config=self.config,
                session=self.session,
                run_dir=self.run_dir,
                submission_dir=submission_dir,
            )

            # Shared state across workers.
            results: dict[str, list[dict[str, Any]]] = {"draft": [], "debug": [], "improve": []}
            best_state = BestState()
            state_lock = threading.Lock()

            max_steps = int(getattr(self.config, "max_steps", 400))
            worker_agents_map = {idx: create_worker_agents(self, idx) for idx in range(self.max_workers)}

            # Step 2: worker execution loop.
            def worker_loop(worker_index: int) -> dict[str, Any]:
                """Execute worker loop.

                Args:
                    worker_index: Numeric control parameter.

                Returns:
                    dict[str, Any]: Result of this function.
                """
                worker_agents = worker_agents_map[worker_index]
                worker_workspace = resolve_worker_workspace(self.config, worker_index, workspace)
                worker_workspace.mkdir(parents=True, exist_ok=True)
                ensure_prepared_links(self.config, worker_workspace)

                (worker_workspace / "working").mkdir(parents=True, exist_ok=True)
                (worker_workspace / "submission").mkdir(parents=True, exist_ok=True)
                worker_submission_dir = worker_workspace / "submission"
                data_preview = generate_data_preview(worker_workspace)
                completed = 0

                while True:
                    should_wait, task = acquire_work_item(
                        search_mgr=search_mgr,
                        best_state=best_state,
                        lock=state_lock,
                        max_steps=max_steps,
                    )

                    if task is None:
                        if should_wait:
                            time.sleep(0.1)
                            continue
                        break

                    try:
                        try:
                            result = run_one_node(
                                config=self.config,
                                session=self.session,
                                worker_agents=worker_agents,
                                worker_workspace=worker_workspace,
                                data_preview=data_preview,
                                task_description=task_description,
                                stage=task.stage,
                                node=task.node,
                                prev_code=task.prev_code,
                                term_out=task.term_out,
                                issue=task.issue,
                                best_code=task.best_code,
                                best_metric=task.best_metric,
                                memory=task.memory,
                                exp_index=task.dispatch_id,
                            )
                        except Exception as exc:  # noqa: BLE001
                            self.logger.error(
                                "Worker %s failed on node %s: %s",
                                worker_index,
                                task.node.id,
                                exc,
                                exc_info=True,
                            )
                            result = {
                                "plan": "",
                                "code": "",
                                "raw_response": str(exc),
                                "exec": {"stdout": str(exc), "exit_code": -1},
                                "metric": None,
                                "metric_detail": {"is_bug": True, "has_submission": False},
                            }

                        finalize_work_item(
                            playground=self,
                            task=task,
                            result=result,
                            worker_index=worker_index,
                            worker_submission_dir=worker_submission_dir,
                            submission_dir=submission_dir,
                            search_mgr=search_mgr,
                            best_state=best_state,
                            results=results,
                            lock=state_lock,
                            workspace=workspace,
                        )
                        completed += 1
                    finally:
                        reset_working_dir(worker_workspace, self.logger)

                return {"worker_index": worker_index, "completed": completed}

            worker_tasks = [lambda idx=idx: worker_loop(idx) for idx in range(self.max_workers)]
            worker_results = self.execute_parallel_tasks(worker_tasks, max_workers=self.max_workers)

            # Step 3: summarize worker outputs and return.
            for idx, summary in enumerate(worker_results):
                if isinstance(summary, Exception):
                    self.logger.error("Worker %s returned exception: %s", idx, summary)
                else:
                    self.logger.info("Worker summary: %s", summary)

            # Step 4: mlebench grade
            competition_id = str(getattr(self.config, "exp_id", "")) or str(self.config.get("exp_id", ""))

            auto_grade = bool(getattr(getattr(self.config, "mlebench", {}), "auto_grade", True)) \
                if hasattr(self.config, "mlebench") else bool(self.config.get("mlebench", {}).get("auto_grade", True))

            if auto_grade and competition_id:
                try:
                    grade_path = grade_best_submission_and_save(
                        workspace_dir=self.run_dir / "workspaces" / "task_0",
                        competition_id=competition_id,
                        out_name="mlebench_grade.json",
                        overwrite=False,
                    )
                    self.logger.info("Saved mlebench grade to: %s", grade_path)
                except Exception as exc: 
                    self.logger.warning("mlebench grading failed (ignored): %s", exc, exc_info=True)

            return {"status": "completed", **results}
        
        finally:
            shutdown_grading_server(self.logger)
            self.cleanup()


