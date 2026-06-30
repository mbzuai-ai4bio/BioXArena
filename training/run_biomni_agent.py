#!/usr/bin/env python3
"""Run Biomni agent on BioXArena-XL tasks.

Unlike general LLM agents where the outer runner handles retries,
Biomni's A1 agent handles its own ReAct loop internally.
We just need to:
  1. Build the task prompt
  2. Call agent.go(prompt)
  3. Collect outputs (submission.csv, metrics.json, solution.py)

Examples:
    python run_biomni_agent.py --task sequence-genomics/cgbench-xl-variant
    python run_biomni_agent.py --domain chemical-biology --max-workers 2
    python run_biomni_agent.py --all-tasks --max-workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths (mirror the general LLM runner layout)
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
TRAINING_DIR = SCRIPT_PATH.parent
EVAL_ROOT = TRAINING_DIR.parent          # BioXArena-XL/
WORKSPACE_ROOT = EVAL_ROOT.parent
DEFAULT_TASKS_ROOT = EVAL_ROOT / "tasks"
DEFAULT_PROMPT_PATH = EVAL_ROOT / "prompts" / "unified_eval_prompt.py"
DEFAULT_ENV_PATH = EVAL_ROOT / ".env"
DEFAULT_ROUND_NAME = "round1"
BIOMNI_AGENT_DIR = EVAL_ROOT / "agents" / "Biomni"

REQUIRED_OUTPUT_LABELS = ["submission.csv", "metrics.json", "solution.py"]

SAFE_NATIVE_CRASH_RETRY_APPENDIX = """

# NATIVE-CRASH RETRY MODE
The previous attempt for this task exited from native code before returning a result.
Retry with a more robust implementation, but still train a real model and optimize the task metric.
- Avoid repeating the likely crash pattern: large in-process SVD/PCA/UMAP/Scanpy transforms, large pandas column-by-column concatenation, GPU/native library chains, or native GBDT training on huge dense/categorical matrices.
- Prefer stable, resource-aware feature engineering: direct float32 features, variance/top-k feature selection, sparse one-hot/text hashing, chunked processing, and explicit deletion of large intermediates.
- Prefer robust scikit-learn models first, such as LogisticRegression/SGDClassifier/Ridge for high-dimensional sparse data, or RandomForest/ExtraTrees/HistGradientBoosting for moderate tabular data. Use heavier libraries only if the data size and feature matrix make them low risk.
- Set OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS, and NUMEXPR_NUM_THREADS to 1 at the top of solution.py.
- Execute heavy training through solution.py as a subprocess and check return codes instead of running fragile native-heavy steps directly in the interactive process.
"""

MISSING_OUTPUTS_RETRY_APPENDIX = """

# MISSING OUTPUTS RETRY MODE
The previous attempt finished without all required outputs.
Inspect the existing output directory first, especially biomni_log.txt and solution.py if present.
Find the traceback or execution mistake that prevented submission.csv, metrics.json, or solution.py from being produced, then fix it.
- Do not weaken the modeling plan just because outputs were missing; this retry is for debugging and validation.
- When running solution.py or any generated script, use subprocess.run(..., capture_output=True, text=True), print stdout and stderr, and check returncode before claiming success.
- Before the final response, verify that submission.csv, metrics.json, and solution.py exist in the output directory.
- Validate submission.csv against sample_submission.csv for column names, row count, and first-column values/order.
"""

TIMEOUT_RETRY_APPENDIX = """

# TIMEOUT RETRY MODE
The previous attempt exceeded the 2-hour per-task wall-clock limit.
Retry with a runtime-bounded implementation that must complete within the retry wall-clock budget.
- First estimate train and test size, per-item cost, and total inference cost before choosing the method.
- Produce complete valid outputs early, then improve only if time remains.
- For file-manifest, voxel, image, segmentation, or large-array tasks, avoid expensive full-resolution model inference over every pixel/voxel unless it is simple NumPy/PyTorch vectorization and clearly fits the budget.
- If using learned models, train them on a bounded sample of the training data, and keep test-time inference simple, chunked, and vectorized enough to finish within the wall-clock budget.
- Prefer methods with predictable runtime over higher-ceiling methods whose inference cost is uncertain.
- Save intermediate prediction files incrementally, but do not stop until submission.csv, metrics.json, solution.py, and all referenced files are complete.
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskSpec:
    domain: str
    task: str
    task_dir: Path
    description_path: Path
    output_dir: Path

    @property
    def key(self) -> str:
        return f"{self.domain}/{self.task}"


@dataclass
class TaskRunResult:
    task_key: str
    status: str
    output_dir: str
    error: str | None
    duration_sec: float


# ---------------------------------------------------------------------------
# Task discovery (same logic as general LLM runner)
# ---------------------------------------------------------------------------
def discover_tasks(
    tasks_root: Path,
    output_root: Path,
) -> list[TaskSpec]:
    specs: list[TaskSpec] = []
    if not tasks_root.exists():
        return specs
    for domain_dir in sorted(tasks_root.iterdir()):
        if not domain_dir.is_dir() or domain_dir.name.startswith("."):
            continue
        for task_dir in sorted(domain_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            public_dir = task_dir / "public"
            desc_path = public_dir / "description.md"
            if not desc_path.exists():
                continue
            specs.append(
                TaskSpec(
                    domain=domain_dir.name,
                    task=task_dir.name,
                    task_dir=public_dir,
                    description_path=desc_path,
                    output_dir=output_root / domain_dir.name / task_dir.name,
                )
            )
    return specs


def filter_tasks(
    all_tasks: list[TaskSpec],
    selected_domains: list[str],
    selected_tasks: list[str],
    run_all: bool,
) -> list[TaskSpec]:
    if run_all:
        return sorted(all_tasks, key=lambda t: t.key)
    selected: dict[str, TaskSpec] = {}
    for t in all_tasks:
        if t.domain in selected_domains:
            selected[t.key] = t
        if t.key in selected_tasks:
            selected[t.key] = t
    return [selected[k] for k in sorted(selected)]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def load_prompt_template(prompt_path: Path) -> str:
    ns: dict[str, Any] = {}
    exec(prompt_path.read_text(encoding="utf-8"), ns)
    return ns["PROMPT_TEMPLATE"]


def build_biomni_prompt(prompt_template: str, task_spec: TaskSpec) -> str:
    description = task_spec.description_path.read_text(encoding="utf-8")
    prompt = prompt_template.format(
        task_dir=str(task_spec.task_dir),
        output_dir=str(task_spec.output_dir),
        description=description,
    )
    return prompt


def _coerce_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _backfill_metrics(output_dir: Path, duration_sec: float, token_usage: dict[str, int] | None) -> None:
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        return

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return

    train_time = max(0.0, _coerce_float(metrics.get("train_time_sec")))
    test_time = max(0.0, _coerce_float(metrics.get("test_time_sec")))
    code_total_time = round(max(0.0, duration_sec), 2)
    solution_generation_time = round(max(0.0, code_total_time - train_time - test_time), 2)

    metrics["solution_generation_time_sec"] = solution_generation_time
    metrics["train_time_sec"] = round(train_time, 2)
    metrics["test_time_sec"] = round(test_time, 2)
    metrics["code_total_time_sec"] = code_total_time

    if token_usage is not None:
        metrics["input_tokens"] = int(token_usage.get("input_tokens", 0))
        metrics["output_tokens"] = int(token_usage.get("output_tokens", 0))

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Biomni agent runner
# ---------------------------------------------------------------------------
def run_single_task(
    task_spec: TaskSpec,
    prompt_template: str,
    llm_model: str,
    source: str,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
    timeout_seconds: int,
    print_lock: threading.Lock,
) -> TaskRunResult:
    """Run Biomni A1 agent on a single task."""
    task_start = time.time()
    task_spec.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from _core_pkg_guard import restore_core_packages
        restore_core_packages()
    except Exception:
        pass

    # Pre-create biomni_data as a symlink to the shared copy so A1 skips
    # the 15GB S3 download and avoids per-task redundant copies.
    _shared_biomni_data = Path("/lustre/scratch/shared-folders/bioagent-project/biomni_data")
    _local_biomni_data = task_spec.output_dir / "biomni_data"
    if _shared_biomni_data.is_dir() and not _local_biomni_data.exists():
        try:
            _local_biomni_data.symlink_to(_shared_biomni_data)
        except Exception:
            pass

    with print_lock:
        print(f"[START] {task_spec.key}")

    try:
        # Set cwd to task output directory so any files the agent writes
        # (models, data, etc.) land there instead of polluting training/.
        os.chdir(str(task_spec.output_dir))

        # Limit per-worker memory to ~60GB (256GB total / 4 workers, with headroom)
        import resource
        mem_limit = 60 * 1024 * 1024 * 1024  # 60GB in bytes
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))

        # Let the system handle GPU assignment naturally.

        # Import Biomni A1 here (after sys.path is set)
        from biomni.agent.a1 import A1

        # Set temperature via env var (Biomni reads it from config)
        os.environ["BIOMNI_TEMPERATURE"] = str(temperature)

        # Initialize agent
        agent = A1(
            path=str(task_spec.output_dir),
            llm=llm_model,
            source=source,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        agent.configure()

        # Build prompt
        prompt = build_biomni_prompt(prompt_template, task_spec)

        # Save prompt for debugging
        (task_spec.output_dir / "biomni_prompt.txt").write_text(prompt, encoding="utf-8")

        # Run agent
        log, final_response = agent.go(prompt)
        token_usage = agent.get_token_usage()

        # Save agent log
        (task_spec.output_dir / "biomni_log.txt").write_text(
            "\n".join(str(entry) for entry in log), encoding="utf-8"
        )

        duration = time.time() - task_start
        _backfill_metrics(task_spec.output_dir, duration, token_usage)

        # Check required outputs
        missing = [f for f in REQUIRED_OUTPUT_LABELS if not (task_spec.output_dir / f).exists()]

        if missing:
            status = "failed"
            error = f"Missing outputs: {missing}"
        else:
            status = "success"
            error = None

        with print_lock:
            print(f"[{status.upper()}] {task_spec.key} ({duration:.0f}s)")

        return TaskRunResult(
            task_key=task_spec.key,
            status=status,
            output_dir=str(task_spec.output_dir),
            error=error,
            duration_sec=duration,
        )

    except Exception as exc:
        duration = time.time() - task_start
        error_text = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        with print_lock:
            print(f"[ERROR] {task_spec.key} ({duration:.0f}s): {exc}")

        # Save error
        error_path = task_spec.output_dir / "FAILED.json"
        error_path.write_text(
            json.dumps({"task": task_spec.key, "error": error_text, "duration_sec": duration}, indent=2),
            encoding="utf-8",
        )

        return TaskRunResult(
            task_key=task_spec.key,
            status="failed",
            output_dir=str(task_spec.output_dir),
            error=str(exc),
            duration_sec=duration,
        )


# ---------------------------------------------------------------------------
# Per-task wall-clock wrapper (multiprocessing-based hard timeout)
# ---------------------------------------------------------------------------
def _child_run_task(
    task_spec,
    prompt_template,
    llm_model,
    source,
    base_url,
    api_key,
    temperature,
    timeout_seconds,
    result_queue,
):
    """Child-process entry: runs run_single_task and pushes the result."""
    try:
        local_lock = threading.Lock()
        result = run_single_task(
            task_spec=task_spec,
            prompt_template=prompt_template,
            llm_model=llm_model,
            source=source,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            print_lock=local_lock,
        )
        result_queue.put(result)
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(
            TaskRunResult(
                task_key=task_spec.key,
                status="failed",
                output_dir=str(task_spec.output_dir),
                error=f"child process crashed: {type(exc).__name__}: {exc}",
                duration_sec=0.0,
            )
        )


def run_task_with_wall_clock(
    task_spec,
    prompt_template,
    llm_model,
    source,
    base_url,
    api_key,
    temperature,
    timeout_seconds,
    wall_clock_sec,
    print_lock,
    safe_retry_on_native_crash=False,
    native_crash_retry_used=False,
    retry_on_missing_outputs=False,
    missing_outputs_retry_used=False,
    retry_on_timeout=False,
    timeout_retry_used=False,
):
    """Run a single task in a child process with a hard wall-clock limit."""
    # Let the system handle GPU assignment naturally.

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(
        target=_child_run_task,
        args=(
            task_spec,
            prompt_template,
            llm_model,
            source,
            base_url,
            api_key,
            temperature,
            timeout_seconds,
            q,
        ),
    )
    start = time.time()
    proc.start()
    proc.join(wall_clock_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        duration = time.time() - start
        with print_lock:
            print(f"[TIMEOUT] {task_spec.key} ({duration:.0f}s) exceeded wall-clock {wall_clock_sec}s")
        if retry_on_timeout and not timeout_retry_used:
            with print_lock:
                print(
                    f"[TIMEOUT-RETRY] {task_spec.key} exceeded wall-clock; "
                    f"retrying once with runtime-bounded prompt ({wall_clock_sec}s retry budget)"
                )
            retry_result = run_task_with_wall_clock(
                task_spec=task_spec,
                prompt_template=prompt_template + TIMEOUT_RETRY_APPENDIX,
                llm_model=llm_model,
                source=source,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                wall_clock_sec=wall_clock_sec,
                print_lock=print_lock,
                safe_retry_on_native_crash=safe_retry_on_native_crash,
                native_crash_retry_used=native_crash_retry_used,
                retry_on_missing_outputs=retry_on_missing_outputs,
                missing_outputs_retry_used=missing_outputs_retry_used,
                retry_on_timeout=retry_on_timeout,
                timeout_retry_used=True,
            )
            retry_result.duration_sec += duration
            if retry_result.status != "success" and retry_result.error:
                retry_result.error = (
                    f"Timeout retry failed after initial timeout of {wall_clock_sec}s: "
                    f"{retry_result.error}"
                )
            return retry_result
        try:
            task_spec.output_dir.mkdir(parents=True, exist_ok=True)
            (task_spec.output_dir / "FAILED.json").write_text(
                json.dumps(
                    {
                        "task": task_spec.key,
                        "error": f"Task wall-clock exceeded {wall_clock_sec}s",
                        "duration_sec": duration,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass
        return TaskRunResult(
            task_key=task_spec.key,
            status="failed",
            output_dir=str(task_spec.output_dir),
            error=f"Task wall-clock exceeded {wall_clock_sec}s",
            duration_sec=duration,
        )

    try:
        result = q.get(timeout=10)
        if (
            retry_on_missing_outputs
            and not missing_outputs_retry_used
            and result.status == "failed"
            and result.error
            and result.error.startswith("Missing outputs")
        ):
            duration = time.time() - start
            retry_wall_clock = max(60, wall_clock_sec - int(duration))
            with print_lock:
                print(
                    f"[OUTPUT-RETRY] {task_spec.key} returned missing outputs; "
                    f"retrying once with debugging prompt ({retry_wall_clock}s remaining)"
                )
            retry_result = run_task_with_wall_clock(
                task_spec=task_spec,
                prompt_template=prompt_template + MISSING_OUTPUTS_RETRY_APPENDIX,
                llm_model=llm_model,
                source=source,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                wall_clock_sec=retry_wall_clock,
                print_lock=print_lock,
                safe_retry_on_native_crash=safe_retry_on_native_crash,
                native_crash_retry_used=native_crash_retry_used,
                retry_on_missing_outputs=retry_on_missing_outputs,
                missing_outputs_retry_used=True,
                retry_on_timeout=retry_on_timeout,
                timeout_retry_used=timeout_retry_used,
            )
            retry_result.duration_sec += duration
            if retry_result.status != "success" and retry_result.error:
                retry_result.error = (
                    f"Missing outputs retry failed after initial error={result.error!r}: "
                    f"{retry_result.error}"
                )
            return retry_result
        return result
    except Exception as exc:
        duration = time.time() - start
        if (
            safe_retry_on_native_crash
            and not native_crash_retry_used
            and proc.exitcode is not None
            and proc.exitcode < 0
        ):
            retry_wall_clock = max(60, wall_clock_sec - int(duration))
            with print_lock:
                print(
                    f"[SAFE-RETRY] {task_spec.key} child exited with code={proc.exitcode}; "
                    f"retrying once with conservative prompt ({retry_wall_clock}s remaining)"
                )
            retry_result = run_task_with_wall_clock(
                task_spec=task_spec,
                prompt_template=prompt_template + SAFE_NATIVE_CRASH_RETRY_APPENDIX,
                llm_model=llm_model,
                source=source,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                wall_clock_sec=retry_wall_clock,
                print_lock=print_lock,
                safe_retry_on_native_crash=safe_retry_on_native_crash,
                native_crash_retry_used=True,
                retry_on_missing_outputs=retry_on_missing_outputs,
                missing_outputs_retry_used=missing_outputs_retry_used,
                retry_on_timeout=retry_on_timeout,
                timeout_retry_used=timeout_retry_used,
            )
            retry_result.duration_sec += duration
            if retry_result.status != "success" and retry_result.error:
                retry_result.error = (
                    f"Native crash retry failed after initial code={proc.exitcode}: "
                    f"{retry_result.error}"
                )
            return retry_result
        return TaskRunResult(
            task_key=task_spec.key,
            status="failed",
            output_dir=str(task_spec.output_dir),
            error=f"Child process exited (code={proc.exitcode}) without result: {exc}",
            duration_sec=duration,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run Biomni agent on BioXArena-XL tasks")
    parser.add_argument("--prefix-dir", type=str, default=str(WORKSPACE_ROOT))
    parser.add_argument("--model", type=str, default="qwen/qwen3.6-plus:free")
    parser.add_argument("--source", type=str, default="Custom",
                        help="LLM source: OpenAI, Anthropic, Custom, etc.")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Base URL for the LLM API (e.g., https://openrouter.ai/api/v1)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (or reads from .env)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=600,
                        help="Timeout for each code execution INSIDE Biomni agent (per shell call)")
    parser.add_argument("--task-wall-clock-sec", type=int, default=7200,
                        help="Hard per-task wall-clock in seconds; child process is killed when exceeded. Default 7200 (2h).")
    parser.add_argument("--safe-retry-on-native-crash", action="store_true",
                        help="On native child crashes such as SIGSEGV, retry the task once with a conservative safe-baseline prompt. Default off.")
    parser.add_argument("--retry-on-missing-outputs", action="store_true",
                        help="When a task returns Missing outputs, retry once with a debugging/validation prompt. Default off.")
    parser.add_argument("--retry-on-timeout", action="store_true",
                        help="When a task exceeds wall-clock, retry once with a runtime-bounded prompt. Default off.")
    parser.add_argument("--round-name", type=str, default=DEFAULT_ROUND_NAME)
    parser.add_argument("--task", action="append", default=[], dest="tasks",
                        help="Specific task(s), e.g., sequence-genomics/cgbench-xl-variant")
    parser.add_argument("--domain", action="append", default=[], dest="domains",
                        help="Run all tasks in domain(s)")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prefix_dir = Path(args.prefix_dir)
    tasks_root = prefix_dir / "BioXArena-Data-Public-XL"
    model_dir_name = args.model.replace("/", "__").replace(":", "_")
    output_root = prefix_dir / "BioXArena-Output-XL" / f"biomni__{model_dir_name}" / args.round_name

    # Load API credentials from .env if not provided
    if args.api_key is None or args.base_url is None:
        env_path = DEFAULT_ENV_PATH
        if env_path.exists():
            env_vars = {}
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip()
            if args.api_key is None:
                args.api_key = env_vars.get("api_key", "")
            if args.base_url is None:
                args.base_url = env_vars.get("api_url", "")

    # Add Biomni to sys.path
    biomni_path = str(BIOMNI_AGENT_DIR)
    if biomni_path not in sys.path:
        sys.path.insert(0, biomni_path)

    # Load prompt template
    prompt_template = load_prompt_template(DEFAULT_PROMPT_PATH)

    # Discover and filter tasks
    all_tasks = discover_tasks(tasks_root, output_root)
    selected = filter_tasks(all_tasks, args.domains, args.tasks, args.all_tasks)

    if not selected:
        print("No tasks selected. Use --task, --domain, or --all-tasks.")
        sys.exit(1)

    print(f"Agent: Biomni A1")
    print(f"Backend LLM: {args.model}")
    print(f"Source: {args.source}")
    print(f"Base URL: {args.base_url}")
    print(f"Temperature: {args.temperature}")
    print(f"Tasks: {len(selected)}")
    print(f"Output root: {output_root}")
    print(f"Max workers: {args.max_workers}")
    if args.safe_retry_on_native_crash:
        print("Safe retry on native crash: enabled")
    if args.retry_on_missing_outputs:
        print("Retry on missing outputs: enabled")
    if args.retry_on_timeout:
        print("Retry on timeout: enabled")
    print()

    if args.dry_run:
        for t in selected:
            print(f"  [DRY] {t.key} -> {t.output_dir}")
        return

    # Run tasks
    print_lock = threading.Lock()
    results: list[TaskRunResult] = []

    print(f"Per-task wall clock: {args.task_wall_clock_sec}s")
    if args.max_workers <= 1:
        for task_spec in selected:
            result = run_task_with_wall_clock(
                task_spec=task_spec,
                prompt_template=prompt_template,
                llm_model=args.model,
                source=args.source,
                base_url=args.base_url,
                api_key=args.api_key,
                temperature=args.temperature,
                timeout_seconds=args.timeout_seconds,
                wall_clock_sec=args.task_wall_clock_sec,
                print_lock=print_lock,
                safe_retry_on_native_crash=args.safe_retry_on_native_crash,
                retry_on_missing_outputs=args.retry_on_missing_outputs,
                retry_on_timeout=args.retry_on_timeout,
            )
            results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(
                    run_task_with_wall_clock,
                    task_spec=task_spec,
                    prompt_template=prompt_template,
                    llm_model=args.model,
                    source=args.source,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    temperature=args.temperature,
                    timeout_seconds=args.timeout_seconds,
                    wall_clock_sec=args.task_wall_clock_sec,
                    print_lock=print_lock,
                    safe_retry_on_native_crash=args.safe_retry_on_native_crash,
                    retry_on_missing_outputs=args.retry_on_missing_outputs,
                    retry_on_timeout=args.retry_on_timeout,
                ): task_spec
                for task_spec in selected
            }
            for future in as_completed(futures):
                results.append(future.result())

    # Summary
    results.sort(key=lambda r: r.task_key)
    success = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {success} success, {failed} failed, {len(results)} total")
    print(f"{'='*60}")
    for r in results:
        marker = "✓" if r.status == "success" else "✗"
        print(f"  {marker} {r.task_key} ({r.duration_sec:.0f}s)" + (f" - {r.error}" if r.error else ""))

    # Save summary JSON
    summary_path = output_root / f"biomni_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "agent": "biomni",
        "model": args.model,
        "source": args.source,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "round_name": args.round_name,
        "output_root": str(output_root),
        "selected_tasks": [t.key for t in selected],
        "results": [asdict(r) for r in results],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()