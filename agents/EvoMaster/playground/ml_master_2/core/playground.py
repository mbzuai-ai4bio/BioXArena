import math
import os
import logging
import sys
import json
from pathlib import Path
import shutil
import copy
import threading
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evomaster.core import BasePlayground, register_playground
from evomaster.agent.session import (
    LocalSessionConfig,
    DockerSession,
    DockerSessionConfig,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evomaster.agent import Agent

from ..agent.session.local import MLMaster2LocalSession
from evomaster.utils.types import TaskInstance
from .exp.draft_exp import DraftExp
from .exp.research_exp import ResearchExp
from .exp.improve_exp import ImproveExp
from .exp.prefetch_exp import PrefetchExp
from .exp.knowledge_promotion_exp import KnowledgePromotionExp
from .exp.wisdom_promotion_exp import WisdomPromotionExp
from .utils.data_preview import generate
from .utils.code import save_code_to_file
from .utils.watch_dog import (
    TimeoutWatchdog,
    GlobalTimeoutInterrupt,
    RUN_TIMEOUT_SECONDS,
    _async_raise,
)
from typing import List, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

@register_playground("ml_master_2")
class MLMaster2Playground(BasePlayground):
    """Main orchestrator for the ML Master 2 automated ML research system.

    Implements an iterative improvement workflow: prefetch -> draft -> (research -> improve)* -> knowledge/wisdom promotion.
    Supports parallel execution of improvement experiments with resource isolation.
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "agent" / "ml_master_2"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("draft_agent", "debug_agent", "improve_agent", "reseach_agent", "knowledge_promotion_agent", "metric_agent", "prefetch_agent","wisdom_promotion_agent")

        self.initial_code = None
        self.best_score = None
        self.best_solution = None
        self.real_time_best_solution = None
        self.research_plan_and_result = []
        self.prefetch_descriptor = None
        self.is_lower_better = self.config_manager.get("is_lower_better", False)
        self.competition_id = self.config_manager.get("competition_id", "detecting-insults-in-social-commentary")
        # for grading server
        self.ground_truth_dir = os.path.join(os.getcwd(), self.config_manager.get("data_root", "playground/ml_master_2/data"))
        self.mcp_manager = None

        self.exp_index = 0

    def setup(self) -> None:
        """Initialize the playground: session, agents, and workspace directories."""
        self.logger.info("Setting up ml master 2 playground...")

        self._setup_session()
        self._setup_agents()
        self._setup_workspace()

        self.logger.info("ML Master 2 playground setup complete")

    def _setup_session(self) -> None:
        """Create and open a session using MLMaster2LocalSession instead of the default LocalSession."""
        if self.session is None:
            session_type = self.config.session.get("type", "local")
            if session_type == "docker":
                raise ValueError("Docker session is not supported for ML Master 2")
            else:
                session_config_dict = self.config.session.get("local", {}).copy()
                if "working_dir" in session_config_dict and "workspace_path" not in session_config_dict:
                    session_config_dict["workspace_path"] = session_config_dict["working_dir"]
                elif "workspace_path" in session_config_dict and "working_dir" not in session_config_dict:
                    session_config_dict["working_dir"] = session_config_dict["workspace_path"]
                if "config_dir" not in session_config_dict:
                    session_config_dict["config_dir"] = str(self.config_dir)
                session_config = LocalSessionConfig(**session_config_dict)
                self.session = MLMaster2LocalSession(session_config)
                self.logger.info("Using ML Master 2 Local session")

        if not self.session.is_open:
            self.session.open()
        else:
            self.logger.debug("Session already open, reusing existing session")

    def _setup_workspace(self) -> None:
        """Create required workspace subdirectories (best_submission, best_solution, submission, working)."""
        os.makedirs(os.path.join(self.session.config.workspace_path, "best_submission"), exist_ok=True)
        os.makedirs(os.path.join(self.session.config.workspace_path, "best_solution"), exist_ok=True)
        os.makedirs(os.path.join(self.session.config.workspace_path, "submission"), exist_ok=True)
        os.makedirs(os.path.join(self.session.config.workspace_path, "working"), exist_ok=True)
        self.logger.info(f"working_dir: {self.session.config.workspace_path}")

    def _is_valid_score(self, score) -> bool:
        """Check whether a score is valid. NaN and None are treated as invalid."""
        if score is None:
            return False
        if isinstance(score, float) and math.isnan(score):
            return False
        return True

    def compare_score(self, old_score, new_score) -> bool:
        """Determine whether new_score is an improvement over old_score.

        Args:
            old_score: The previous best score (may be None or NaN).
            new_score: The candidate score to compare (may be None or NaN).

        Returns:
            True if new_score is a valid improvement over old_score.
        """
        # Invalid new_score (None/NaN) is never an improvement
        if not self._is_valid_score(new_score):
            return False
        # Invalid old_score (None/NaN) means any valid new_score is an improvement
        if not self._is_valid_score(old_score):
            return True
        # Both valid: compare based on optimization direction
        if old_score < new_score and self.is_lower_better == False:
            return True
        elif old_score > new_score and self.is_lower_better == True:
            return True
        else:
            return False

    def _create_improve_exp(self, exp_index: int) -> ImproveExp:
        """Create an independent ImproveExp instance for parallel execution.

        Each parallel task gets its own agent copies via copy_agent to avoid
        context conflicts during concurrent LLM calls.

        Args:
            exp_index: Experiment index used to generate unique exp_name and agent names.

        Returns:
            A new ImproveExp instance with independent agent copies.
        """
        improve_agent_copy = self.copy_agent(
            self.agents.improve_agent, new_agent_name=f"improve_exp_{exp_index}"
        ) if self.agents.improve_agent else None
        debug_agent_copy = self.copy_agent(
            self.agents.debug_agent, new_agent_name=f"debug_exp_{exp_index}"
        ) if self.agents.debug_agent else None
        metric_agent_copy = self.copy_agent(
            self.agents.metric_agent, new_agent_name=f"metric_exp_{exp_index}"
        ) if self.agents.metric_agent else None
        exp_name = f"exp_{exp_index}_improve"
        return ImproveExp(
            improve_agent_copy, debug_agent_copy, metric_agent_copy,
            self.config, exp_name
        )

    def run(self, task_description: str, output_file: str | None = None) -> dict:
        """Execute the full ML Master 2 pipeline.

        Runs prefetch -> draft -> iterative (research -> parallel improve) cycles
        with a global timeout watchdog. On timeout, triggers wisdom promotion.

        Args:
            task_description: Natural language description of the ML task.
            output_file: Optional path to save the trajectory output.

        Returns:
            A dict with 'status' ('completed' or 'failed') and optional 'error'.
        """
        # Start the watchdog daemon thread
        watchdog = TimeoutWatchdog(RUN_TIMEOUT_SECONDS)
        watchdog.start()
        self.logger.info(f"Watchdog started ({RUN_TIMEOUT_SECONDS} seconds)")
        try:
            self.setup()

            self._setup_trajectory_file(output_file)
            
            prefetch_exp = PrefetchExp(self.agents.prefetch_agent, self.config,f"exp_{self.exp_index}_prefetch")
            self.exp_index += 1
            embedding_config = getattr(self.config, "embedding", {})
            embedding_model = embedding_config.get("openai", {}).get("model", "text-embedding-3-large")
            data_knowledge, model_knowledge, self.prefetch_descriptor = prefetch_exp.run(task_description,vec_dir=os.path.join(os.getcwd(), "playground/ml_master_2/example_wisdom"),nodes_data=os.path.join(os.getcwd(), "playground/ml_master_2/example_wisdom/db.json"),model=embedding_model)
            data_preview = generate(self.session.config.workspace_path)
            self.logger.info(f"Data preview: {data_preview}")
            self.logger.info("Running experiment...")
            draft_exp = DraftExp(self.agents.draft_agent, self.agents.debug_agent, self.agents.metric_agent, self.config,f"exp_{self.exp_index}_draft")
            draft_workspace_name = f"exp_{self.exp_index}_draft"
            self.exp_index += 1
            draft_result = self.execute_parallel_tasks([partial(draft_exp.run, task_description=task_description, data_preview=data_preview, data_knowledge=data_knowledge, model_knowledge=model_knowledge)], max_workers=1, workspace_names=[draft_workspace_name])
            is_sucess, validation_score, uid, self.best_solution = draft_result[0]
            self.initial_code = self.best_solution
            if is_sucess:
                self.best_score = validation_score
                shutil.copy(os.path.join(draft_exp.workspace_path, "submission", f"submission_{uid}.csv"), os.path.join(self.agents.draft_agent.session.config.workspace_path, "best_submission", f"submission.csv"))
                save_code_to_file(os.path.join(self.session.config.workspace_path, "best_solution"), "best_solution.py", self.best_solution)
                self.real_time_best_solution = self.best_solution
            for reseach_round in range(20):
                # Record results for each direction and idea in the current research_round
                # Structure: {direction: {idea: {"improved": bool, "is_best_in_direction": bool, "score": float|None}}}
                research_round_idea_results: dict[str, dict[tuple, dict]] = {}
                base_solution = self.best_solution  # Best code at the start of this round

                research_exp = ResearchExp(self.agents.reseach_agent, self.config, self.initial_code, f"exp_{self.exp_index}_research")
                research_workspace_name = f"exp_{self.exp_index}_research"
                self.exp_index += 1
                research_results = self.execute_parallel_tasks(
                    [partial(research_exp.run, task_description=task_description, data_preview=data_preview, best_solution=self.best_solution, research_plan_and_result=self.research_plan_and_result)],
                    max_workers=1,
                    workspace_names=[research_workspace_name]
                )
                research_result = research_results[0]
                if isinstance(research_result, Exception):
                    self.logger.error(f"Research failed: {research_result}")
                    raise research_result
                research_plan = research_result
                # Read max parallel ideas from config, default to 2
                session_config = self.config.session.get("local", {})
                parallel_config = session_config.get("parallel", {})
                idea_max_workers = parallel_config.get("max_parallel", 2)

                for direction in research_plan:
                    direction_best_solution = self.best_solution
                    direction_best_score = self.best_score
                    direction_baseline_score = self.best_score  # Baseline score for this direction to judge improvements (not updated during iteration)
                    direction_best_idea = None  # Track the idea that brought the best score in this direction
                    research_round_idea_results[direction] = {}

                    ideas = list(research_plan[direction].items())
                    if not ideas:
                        continue

                    # Build parallel tasks: one task per idea, all based on direction_best_solution
                    tasks = []
                    workspace_names = []
                    improve_exp_list = []
                    for i, idea in enumerate(ideas):
                        exp_index = self.exp_index + i
                        improve_exp = self._create_improve_exp(exp_index)
                        improve_exp_list.append(improve_exp)
                        task = partial(
                            improve_exp.run,
                            task_description=task_description,
                            data_preview=data_preview,
                            best_solution=direction_best_solution,
                            idea=idea,
                        )
                        tasks.append(task)
                        workspace_names.append(improve_exp.exp_name)
                    self.exp_index += len(ideas)

                    # Execute all ideas in this direction in parallel, max idea_max_workers concurrent
                    improve_results = self.execute_parallel_tasks(
                        tasks, max_workers=idea_max_workers, workspace_names=workspace_names
                    )

                    # Process results: collect in order, find the best
                    # 'improved' is compared against direction_baseline_score, indicating improvement relative to the baseline at the start of this direction
                    # This way, multiple ideas better than the baseline within the same direction are all correctly marked as "brought improvement"
                    for i, (idea, result) in enumerate(zip(ideas, improve_results)):
                        improve_exp = improve_exp_list[i]
                        if isinstance(result, Exception):
                            self.logger.error(f"Idea {idea} failed: {result}")
                            validation_score = None
                            is_sucess = False
                            uid = None
                            solution = None
                        else:
                            is_sucess, validation_score, uid, solution = result

                        improved = self.compare_score(direction_baseline_score, validation_score)
                        research_round_idea_results[direction][idea] = {
                            "improved": improved,
                            "is_best_in_direction": False,
                            "score": validation_score,
                        }
                        if improved and is_sucess and solution is not None:
                            direction_best_score = validation_score
                            direction_best_solution = solution
                            direction_best_idea = idea
                            shutil.copy(
                                os.path.join(improve_exp.workspace_path, "submission", f"submission_{uid}.csv"),
                                os.path.join(self.agents.improve_agent.session.config.workspace_path, "best_submission", f"submission.csv"),
                            )
                            save_code_to_file(
                                os.path.join(self.session.config.workspace_path, "best_solution"),
                                "best_solution.py",
                                direction_best_solution,
                            )
                            self.real_time_best_solution = direction_best_solution
                    # Mark the best idea in this direction
                    if direction_best_idea is not None:
                        research_round_idea_results[direction][direction_best_idea]["is_best_in_direction"] = True

                    self.best_solution = direction_best_solution
                    self.best_score = direction_best_score

                # At the end of research_round, research_round_idea_results is complete
                # Store research_plan and results as text in research_plan_and_result
                plan_text = json.dumps(research_plan, ensure_ascii=False, indent=2)
                self.research_plan_and_result.extend([plan_text])

                self.logger.info(f"Round {reseach_round} results: {research_round_idea_results}")
                knowledge_promotion_exp = KnowledgePromotionExp(self.agents.knowledge_promotion_agent, self.config, f"exp_{self.exp_index}_knowledge_promotion")
                knowledge_promotion_workspace_name = f"exp_{self.exp_index}_knowledge_promotion"
                self.exp_index += 1
                knowledge_promotion_results = self.execute_parallel_tasks(
                    [partial(knowledge_promotion_exp.run, task_description=task_description, data_preview=data_preview, base_solution=base_solution, best_solution=self.best_solution, research_plan=research_plan, research_round_idea_results=research_round_idea_results)],
                    max_workers=1,
                    workspace_names=[knowledge_promotion_workspace_name]
                )
                knowledge_promotion_result = knowledge_promotion_results[0]
                if isinstance(knowledge_promotion_result, Exception):
                    self.logger.error(f"Knowledge promotion failed: {knowledge_promotion_result}")
                    raise knowledge_promotion_result
                self.research_plan_and_result.extend([knowledge_promotion_result])
            result = {
                "status": "completed",
                "steps": 0,
            }
            return result
        except GlobalTimeoutInterrupt:
            # Precisely catch the interrupt exception thrown by the watchdog
            self.logger.warning(f"Watchdog triggered: experiment has run for {RUN_TIMEOUT_SECONDS} seconds, forced interruption, starting wisdom promotion")
            wisdom_promotion_exp = WisdomPromotionExp(self.agents.wisdom_promotion_agent, self.config, f"exp_{self.exp_index}_wisdom_promotion")
            wisdom_promotion_workspace_name = f"exp_{self.exp_index}_wisdom_promotion"
            self.exp_index += 1
            wisdom_promotion_results = self.execute_parallel_tasks(
                [partial(wisdom_promotion_exp.run, task_description=task_description, best_solution=self.real_time_best_solution)],
                max_workers=1,
                workspace_names=[wisdom_promotion_workspace_name]
            )
            self.logger.info(f"Wisdom promotion finished")
            self.logger.info(f"Task descriptor: {self.prefetch_descriptor}")
            self.logger.info(f"Wisdom promotion result: {wisdom_promotion_results}")
            result = {
                "status": "completed",
                "steps": 0,
            }
            return result
        except Exception as e:
            self.logger.error(f"ML Master 2 task execution failed: {e}", exc_info=True)
            result = {
                "status": "failed",
                "steps": 0,
                "error": str(e),
            }
            return result

        finally:
            if 'watchdog' in locals():
                watchdog.stop()
            self.cleanup()

    def execute_parallel_tasks(self, tasks: List[Callable], max_workers: int = 3, workspace_names: List[str] | None = None) -> List[Any]:
        """Generic parallel task executor with resource allocation and workspace isolation.

        Args:
            tasks: List of callable tasks to execute in parallel.
            max_workers: Maximum number of concurrent workers.
            workspace_names: Optional list of workspace names for each task (for split_workspace_for_exp).

        Returns:
            List of results in the same order as tasks (exceptions are returned as results for failed tasks).
        """
        self.logger.info(f"Starting parallel execution of {len(tasks)} tasks with {max_workers} workers.")

        results = [None] * len(tasks)

        # Check if parallel resource allocation is enabled
        session_config = self.config.session.get("local", {})
        parallel_config = session_config.get("parallel", {})
        parallel_enabled = parallel_config.get("enabled", False)
        split_workspace = parallel_config.get("split_workspace_for_exp", False)

        # Track currently running child thread IDs for forced interruption on global timeout
        active_worker_tids = set()
        tids_lock = threading.Lock()

        # Wrap task function to set parallel index and independent workspace
        def wrap_task(task_func, parallel_index):
            def wrapped():
                current_tid = threading.get_ident()
                with tids_lock:
                    active_worker_tids.add(current_tid)

                try:
                    # If parallel resource allocation is enabled, set session's parallel index
                    if parallel_enabled and self.session is not None:
                        from evomaster.agent.session.local import LocalSession
                        if isinstance(self.session, LocalSession):
                            self.session.set_parallel_index(parallel_index)
                            self.logger.debug(f"Set parallel index: {parallel_index}")

                            # If split_workspace_for_exp is enabled, create independent workspace for current exp
                            if split_workspace:
                                import os
                                main_workspace = self.session.config.workspace_path
                                exp_name = workspace_names[parallel_index] if workspace_names and parallel_index < len(workspace_names) else f"exp_{parallel_index}"
                                exp_workspace = os.path.join(main_workspace, exp_name)
                                # Create exp workspace (with symlinks) via env
                                self.session._env.setup_exp_workspace(exp_workspace)
                                os.makedirs(os.path.join(exp_workspace, "submission"), exist_ok=True)
                                os.makedirs(os.path.join(exp_workspace, "working"), exist_ok=True)
                                # Set thread-local workspace path
                                self.session.set_workspace_path(exp_workspace)
                                self.logger.info(f"Exp {parallel_index} using independent workspace: {exp_workspace}")
                                
                    return task_func()

                except GlobalTimeoutInterrupt:
                    self.logger.warning(f"Parallel task {parallel_index} (TID: {current_tid}) received interrupt signal, exiting and releasing resources...")
                    raise  # Continue raising so Executor catches it and marks Future as failed
                finally:
                    # Clean up thread-local state
                    if parallel_enabled and self.session is not None:
                        from evomaster.agent.session.local import LocalSession
                        if isinstance(self.session, LocalSession):
                            self.session.set_parallel_index(None)
                            if split_workspace:
                                self.session.set_workspace_path(None)

                    # Task ended, remove thread ID record
                    with tids_lock:
                        active_worker_tids.discard(current_tid)
            return wrapped
        
        executor = ThreadPoolExecutor(max_workers=max_workers)
        wrapped_tasks = [wrap_task(task, i) for i, task in enumerate(tasks)]
        future_to_index = {executor.submit(wrapped_task): i for i, wrapped_task in enumerate(wrapped_tasks)}

        try:
            from concurrent.futures import wait, FIRST_COMPLETED
            not_done = set(future_to_index.keys())
            
            while not_done:
                done, not_done = wait(
                    not_done, 
                    timeout=0.5, 
                    return_when=FIRST_COMPLETED
                )
                
                for future in done:
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results[index] = result
                    except Exception as exc:
                        self.logger.error(f"Task {index} generated an exception: {exc}")
                        results[index] = exc

            self.logger.info("Parallel execution completed.")
            return results

        finally:
            # 1. Cancel all queued, not-yet-started Futures
            for future in future_to_index:
                future.cancel()

            # 2. Actively inject global timeout exception into all still-running child threads
            # This forces child threads executing task_func to jump into wrapped()'s finally block
            with tids_lock:
                for tid in active_worker_tids:
                    try:
                        _async_raise(tid, GlobalTimeoutInterrupt)
                    except Exception as e:
                        self.logger.error(f"Unable to send interrupt signal to child thread {tid}: {e}")

            # 3. Forcibly shut down thread pool
            if sys.version_info >= (3, 9):
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=False)
