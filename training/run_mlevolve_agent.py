#!/usr/bin/env python3
"""Run MLEvolve (MCGS-based) agent on BioXArena tasks.

MLEvolve uses Monte Carlo Graph Search with multi-agent collaboration:
  - Draft Agent: initial solution generation
  - Improve Agent: solution improvement
  - Debug Agent: bug fixing
  - Evolution Agent: trajectory-aware evolution
  - Fusion Agent: cross-branch fusion

Examples:
    python run_mlevolve_agent.py --task chemical-biology/tox21-sr-are
    python run_mlevolve_agent.py --all-tasks --max-workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
TRAINING_DIR = SCRIPT_PATH.parent
EVAL_ROOT = TRAINING_DIR.parent
WORKSPACE_ROOT = EVAL_ROOT.parent
DEFAULT_TASKS_ROOT = EVAL_ROOT / "tasks"
DEFAULT_ENV_PATH = EVAL_ROOT / ".env"
DEFAULT_ROUND_NAME = "round1"
MLEVOLVE_DIR = EVAL_ROOT / "agents" / "MLEvolve"
MLEVOLVE_EXEC_TIMEOUT = 5400

REQUIRED_OUTPUT_LABELS = ["submission.csv", "metrics.json", "solution.py"]
_PATH_COLUMN_SUFFIXES = ("_file", "_path", "_dir")


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


def discover_tasks(tasks_root: Path, output_root: Path) -> list[TaskSpec]:
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


def find_latest_artifact(search_root: Path, relative_pattern: str) -> Path | None:
    matches = [p for p in search_root.rglob(relative_pattern) if p.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def find_latest_bundle_dir(search_root: Path, relative_pattern: str) -> Path | None:
    bundle_file = find_latest_artifact(search_root, relative_pattern)
    if bundle_file is None:
        return None
    return bundle_file.parent


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_numeric_metric(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(f"`{key}` must be a number or numeric string in metrics.json.")


def parse_nonnegative_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return 0
        return max(0, int(float(stripped)))
    return 0


def parse_optional_numeric_metric(payload: dict[str, Any], key: str) -> float | None:
    try:
        return parse_numeric_metric(payload, key)
    except Exception:
        return None


def parse_execution_time_metric(metric_txt_path: Path | None) -> float | None:
    if metric_txt_path is None or not metric_txt_path.exists():
        return None

    for line in metric_txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("Execution Time(s):"):
            continue
        raw_value = line.split(":", 1)[1].strip()
        if not raw_value or raw_value.upper() == "N/A":
            return None
        return float(raw_value)

    return None


def resolve_solution_generated_time(task_start_time: float, solution_path: Path | None) -> float:
    if solution_path is None or not solution_path.exists():
        return time.time()
    try:
        return min(time.time(), max(task_start_time, solution_path.stat().st_mtime))
    except OSError:
        return time.time()


def collect_llm_usage(search_root: Path) -> tuple[int, int, bool]:
    if not search_root.exists():
        return 0, 0, False

    total_input_tokens = 0
    total_output_tokens = 0
    found_usage = False

    for usage_path in sorted(search_root.rglob("llm_usage.jsonl")):
        for raw_line in usage_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            total_input_tokens += parse_nonnegative_int(payload.get("input_tokens"))
            total_output_tokens += parse_nonnegative_int(payload.get("output_tokens"))
            found_usage = True

    return total_input_tokens, total_output_tokens, found_usage


def infer_model_used(solution_path: Path | None) -> str:
    if solution_path is None or not solution_path.exists():
        return "Unknown"

    try:
        source = solution_path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return "Unknown"

    keyword_map = (
        ("xgboost", "XGBoost"),
        ("xgbclassifier", "XGBoost"),
        ("xgbregressor", "XGBoost"),
        ("lightgbm", "LightGBM"),
        ("lgbmclassifier", "LightGBM"),
        ("lgbmregressor", "LightGBM"),
        ("catboost", "CatBoost"),
        ("randomforest", "RandomForest"),
        ("extratrees", "ExtraTrees"),
        ("logisticregression", "LogisticRegression"),
        ("linearregression", "LinearRegression"),
        ("svc(", "SVM"),
        ("svr(", "SVM"),
        ("transformers", "Transformers"),
        ("automodel", "Transformers"),
        ("bert", "Transformers"),
        ("resnet", "CNN"),
        ("conv2d", "CNN"),
        ("torch.nn", "PyTorch"),
        ("tensorflow", "TensorFlow"),
        ("sklearn", "Scikit-learn"),
    )
    for keyword, model_label in keyword_map:
        if keyword in source:
            return model_label

    return "Unknown"


def combine_notes(*note_groups: Any) -> str:
    combined: list[str] = []
    seen: set[str] = set()
    for note_group in note_groups:
        if note_group is None:
            continue
        if isinstance(note_group, str):
            candidates = [note_group]
        elif isinstance(note_group, (list, tuple, set)):
            candidates = [str(item) for item in note_group if str(item).strip()]
        else:
            candidates = [str(note_group)]
        for note in candidates:
            normalized = note.strip()
            if not normalized or normalized in seen:
                continue
            combined.append(normalized)
            seen.add(normalized)
    return "\n".join(combined)


def write_final_metrics_json(
    output_path: Path,
    source_metrics_path: Path | None,
    metric_txt_path: Path | None,
    solution_path: Path | None,
    task_start_time: float,
    solution_generated_time: float,
    input_tokens: int,
    output_tokens: int,
    usage_found: bool,
    model: str,
    temperature: float,
    steps: int,
) -> None:
    source_payload: dict[str, Any] = {}
    if source_metrics_path is not None and source_metrics_path.exists():
        payload = read_json(source_metrics_path)
        if not isinstance(payload, dict):
            raise ValueError("source metrics.json must contain a JSON object.")
        source_payload = dict(payload)

    aggregate_exec_time = parse_execution_time_metric(metric_txt_path)
    train_time_sec = parse_optional_numeric_metric(source_payload, "train_time_sec")
    test_time_sec = parse_optional_numeric_metric(source_payload, "test_time_sec")

    fallback_notes: list[str] = []
    if train_time_sec is None or test_time_sec is None:
        if aggregate_exec_time is not None:
            if train_time_sec is None:
                train_time_sec = aggregate_exec_time
            if test_time_sec is None:
                test_time_sec = 0.0
            fallback_notes.append(
                "MLEvolve did not emit phase-separated train/test timings; runner used aggregate candidate execution time from metric.txt and set any missing phase timing to 0."
            )
        else:
            if train_time_sec is None:
                train_time_sec = 0.0
            if test_time_sec is None:
                test_time_sec = 0.0
            fallback_notes.append(
                "MLEvolve did not emit train/test timings; runner defaulted missing timing fields to 0."
            )

    model_used = source_payload.get("model_used") or infer_model_used(solution_path)
    if model_used == "Unknown":
        model_used = "Unknown (inspect solution.py)"

    phase_execution_time_sec = float(train_time_sec) + float(test_time_sec)
    solution_generation_time_sec = max(
        0.0,
        solution_generated_time - task_start_time - phase_execution_time_sec,
    )

    notes = combine_notes(
        source_payload.get("notes"),
        None if source_metrics_path is not None else "metrics.json synthesized by run_mlevolve_agent_dxb.py from MLEvolve artifacts.",
        None if usage_found else "MLEvolve did not persist token usage metadata; input_tokens/output_tokens defaulted to 0.",
        fallback_notes,
        f"MLEvolve search metadata: steps={steps}, code_model={model}, temperature={temperature}.",
    )

    final_payload = dict(source_payload)
    final_payload["solution_generation_time_sec"] = round(solution_generation_time_sec, 6)
    final_payload["train_time_sec"] = round(float(train_time_sec), 6)
    final_payload["test_time_sec"] = round(float(test_time_sec), 6)
    final_payload["code_total_time_sec"] = round(
        solution_generation_time_sec + float(train_time_sec) + float(test_time_sec), 6
    )
    final_payload["input_tokens"] = input_tokens if usage_found else 0
    final_payload["output_tokens"] = output_tokens if usage_found else 0
    final_payload["model_used"] = model_used
    final_payload["model_param_count"] = source_payload.get("model_param_count")
    final_payload["notes"] = notes

    write_json(output_path, final_payload)


def validate_metrics_json(metrics_path: Path) -> None:
    payload = read_json(metrics_path)
    if not isinstance(payload, dict):
        raise ValueError("metrics.json must contain a JSON object.")

    required_fields = (
        "solution_generation_time_sec",
        "train_time_sec",
        "test_time_sec",
        "code_total_time_sec",
        "input_tokens",
        "output_tokens",
        "model_used",
        "model_param_count",
        "notes",
    )
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        raise ValueError(
            "metrics.json is missing required field(s): " + ", ".join(missing_fields)
        )

    parse_numeric_metric(payload, "solution_generation_time_sec")
    parse_numeric_metric(payload, "train_time_sec")
    parse_numeric_metric(payload, "test_time_sec")
    parse_numeric_metric(payload, "code_total_time_sec")
    parse_nonnegative_int(payload.get("input_tokens"))
    parse_nonnegative_int(payload.get("output_tokens"))


def verify_required_outputs(output_dir: Path) -> list[str]:
    missing: list[str] = []
    for filename in REQUIRED_OUTPUT_LABELS:
        if not (output_dir / filename).exists():
            missing.append(filename)
    return missing


def looks_like_artifact_value(value: str, column_name: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    normalized = stripped.replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute():
        return True
    if "/" in stripped or "\\" in stripped:
        return True
    return column_name.lower().endswith(_PATH_COLUMN_SUFFIXES)


def collect_submission_artifact_targets(submission_csv: Path, output_dir: Path) -> set[Path]:
    if not submission_csv.exists():
        return set()

    targets: set[Path] = set()
    with submission_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return targets

        for row in reader:
            for column_name in reader.fieldnames[1:]:
                raw_value = (row.get(column_name) or "").strip()
                if not looks_like_artifact_value(raw_value, column_name):
                    continue
                artifact_path = Path(raw_value.replace("\\", "/"))
                if artifact_path.is_absolute():
                    try:
                        artifact_path = artifact_path.relative_to(output_dir)
                    except ValueError:
                        continue
                if not artifact_path.parts:
                    continue
                top_level_target = output_dir / artifact_path.parts[0]
                targets.add(top_level_target)

    return targets


def clear_exported_submission_bundle(output_dir: Path) -> None:
    submission_csv = output_dir / "submission.csv"
    for target in sorted(
        collect_submission_artifact_targets(submission_csv, output_dir),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()

    if submission_csv.exists():
        submission_csv.unlink()


def copy_bundle_dir_contents(source_dir: Path, target_dir: Path) -> None:
    for source_path in source_dir.iterdir():
        target_path = target_dir / source_path.name
        if source_path.is_dir():
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)


def clear_benchmark_outputs(output_dir: Path) -> None:
    clear_exported_submission_bundle(output_dir)
    for filename in REQUIRED_OUTPUT_LABELS + ["FAILED.json"]:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def clear_mlevolve_run_dirs(output_dir: Path) -> None:
    for dirname in ("logs", "workspaces"):
        path = output_dir / dirname
        if path.exists():
            shutil.rmtree(path)


def write_failed_marker(
    task_spec: TaskSpec,
    error: str | None,
    duration_sec: float,
    missing_outputs: list[str],
    exit_code: int | None = None,
) -> None:
    payload = {
        "task_key": task_spec.key,
        "status": "failed",
        "output_dir": str(task_spec.output_dir),
        "final_error": error,
        "duration_sec": duration_sec,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "missing_required_outputs": missing_outputs,
        "exit_code": exit_code,
    }
    write_json(task_spec.output_dir / "FAILED.json", payload)


def run_single_task(
    task_spec: TaskSpec,
    mlevolve_dir: Path,
    dataset_root: Path,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    steps: int,
    time_limit: int,
    print_lock: threading.Lock,
) -> TaskRunResult:
    """Run MLEvolve MCGS agent on a single task via subprocess."""
    task_start = time.time()
    task_spec.output_dir.mkdir(parents=True, exist_ok=True)
    clear_benchmark_outputs(task_spec.output_dir)
    clear_mlevolve_run_dirs(task_spec.output_dir)
    try:
        from _core_pkg_guard import restore_core_packages
        restore_core_packages()
    except Exception:
        pass

    with print_lock:
        print(f"[START] {task_spec.key}")

    try:
        log_root = task_spec.output_dir / "logs"
        workspace_root = task_spec.output_dir / "workspaces"
        exp_id = task_spec.key
        exp_name = task_spec.task

        cmd = [
            "timeout",
            "--foreground",
            "--signal=TERM",
            "--kill-after=10s",
            f"{time_limit}s",
            sys.executable,
            str(mlevolve_dir / "run.py"),
            f"exp_id={exp_id}",
            f"data_dir={task_spec.task_dir}",
            f"dataset_dir={dataset_root}",
            f"desc_file={task_spec.description_path}",
            f"exp_name={exp_name}",
            f"log_dir={log_root}",
            f"workspace_dir={workspace_root}",
            f"agent.code.model={model}",
            f"agent.code.temp={temperature}",
            f"agent.code.base_url={base_url}",
            f"agent.code.api_key={api_key}",
            f"agent.feedback.model={model}",
            f"agent.feedback.temp={temperature}",
            f"agent.feedback.base_url={base_url}",
            f"agent.feedback.api_key={api_key}",
            f"agent.steps={steps}",
            f"agent.time_limit={time_limit}",
            f"exec.timeout={MLEVOLVE_EXEC_TIMEOUT}",
            "preprocess_data=False",
            "copy_data=False",
            "start_cpu_id=0",
            "cpu_number=16",
        ]

        (task_spec.output_dir / "mlevolve_cmd.txt").write_text(
            " \\\n  ".join(cmd), encoding="utf-8"
        )

        result = subprocess.run(
            cmd,
            cwd=str(mlevolve_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        (task_spec.output_dir / "mlevolve_stdout.log").write_text(
            result.stdout or "", encoding="utf-8"
        )
        (task_spec.output_dir / "mlevolve_stderr.log").write_text(
            result.stderr or "", encoding="utf-8"
        )

        best_submission_dir = (
            find_latest_bundle_dir(workspace_root, "best_submission/submission.csv")
            or find_latest_bundle_dir(workspace_root, "top_solution/top1/submission.csv")
        )
        best_submission = (
            best_submission_dir / "submission.csv"
            if best_submission_dir is not None
            else find_latest_artifact(workspace_root, "submission.csv")
        )
        best_solution = (
            find_latest_artifact(workspace_root, "best_solution/solution.py")
            or find_latest_artifact(log_root, "best_solution.py")
            or find_latest_artifact(log_root, "solution.py")
        )
        best_metric_txt = (
            find_latest_artifact(workspace_root, "best_solution/metric.txt")
            or find_latest_artifact(workspace_root, "top_solution/top1/metric.txt")
        )
        best_metrics = (
            find_latest_artifact(workspace_root, "metrics.json")
            or find_latest_artifact(log_root, "metrics.json")
        )

        # Copy outputs to expected locations
        if best_submission_dir is not None:
            copy_bundle_dir_contents(best_submission_dir, task_spec.output_dir)
        elif best_submission is not None:
            shutil.copy2(best_submission, task_spec.output_dir / "submission.csv")
        if best_solution is not None:
            shutil.copy2(best_solution, task_spec.output_dir / "solution.py")
        duration = time.time() - task_start
        input_tokens, output_tokens, usage_found = collect_llm_usage(log_root)
        solution_generated_time = resolve_solution_generated_time(task_start, best_solution)

        missing = verify_required_outputs(task_spec.output_dir)
        metrics_error: str | None = None
        if best_solution is not None:
            try:
                write_final_metrics_json(
                    output_path=task_spec.output_dir / "metrics.json",
                    source_metrics_path=best_metrics,
                    metric_txt_path=best_metric_txt,
                    solution_path=best_solution,
                    task_start_time=task_start,
                    solution_generated_time=solution_generated_time,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    usage_found=usage_found,
                    model=model,
                    temperature=temperature,
                    steps=steps,
                )
            except Exception as exc:
                metrics_error = f"metrics.json synthesis failed: {exc}"

        metrics_path = task_spec.output_dir / "metrics.json"
        missing = verify_required_outputs(task_spec.output_dir)
        if metrics_error is None and metrics_path.exists():
            try:
                validate_metrics_json(metrics_path)
            except Exception as exc:
                metrics_error = f"metrics.json validation failed: {exc}"

        metrics_ready = metrics_error is None and metrics_path.exists()
        if metrics_ready:
            status = "success"
            error = None
            failed_marker = task_spec.output_dir / "FAILED.json"
            if failed_marker.exists():
                failed_marker.unlink()
        else:
            status = "failed"
            if metrics_error is not None:
                error = metrics_error
                if "metrics.json" not in missing:
                    missing.append("valid metrics.json")
            elif result.returncode == 124:
                error = f"Timeout after {time_limit}s"
            elif result.returncode != 0:
                error = f"MLEvolve exited with code {result.returncode}"
            elif missing:
                error = "Missing required outputs: " + ", ".join(missing)
            else:
                error = "metrics.json was not generated"

        if status == "failed":
            write_failed_marker(
                task_spec=task_spec,
                error=error,
                duration_sec=duration,
                missing_outputs=missing,
                exit_code=result.returncode,
            )

        with print_lock:
            print(f"[{status.upper()}] {task_spec.key} ({duration:.0f}s)")

        return TaskRunResult(
            task_key=task_spec.key, status=status,
            output_dir=str(task_spec.output_dir), error=error, duration_sec=duration,
        )

    except Exception as exc:
        duration = time.time() - task_start
        with print_lock:
            print(f"[ERROR] {task_spec.key} ({duration:.0f}s): {exc}")
        write_failed_marker(
            task_spec=task_spec,
            error=str(exc),
            duration_sec=duration,
            missing_outputs=verify_required_outputs(task_spec.output_dir),
        )
        return TaskRunResult(
            task_key=task_spec.key, status="failed",
            output_dir=str(task_spec.output_dir), error=str(exc), duration_sec=duration,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLEvolve agent on BioXArena tasks")
    parser.add_argument("--prefix-dir", type=str, default=str(WORKSPACE_ROOT))
    parser.add_argument("--model", type=str, default="google/gemma-4-31b-it")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--time-limit", type=int, default=3600)
    parser.add_argument("--round-name", type=str, default=DEFAULT_ROUND_NAME)
    parser.add_argument("--task", action="append", default=[], dest="tasks")
    parser.add_argument("--domain", action="append", default=[], dest="domains")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prefix_dir = Path(args.prefix_dir).resolve()
    tasks_root = (prefix_dir / "BioXArena-Data-Public").resolve()
    model_dir_name = args.model.replace("/", "__").replace(":", "_")
    output_root = prefix_dir / "BioXArena-Output" / f"mlevolve__{model_dir_name}" / args.round_name
    mlevolve_dir = MLEVOLVE_DIR.resolve()
    env_path = prefix_dir / "BioXArena" / ".env"

    if args.api_key is None or args.base_url is None:
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
                args.base_url = env_vars.get("api_url", "https://openrouter.ai/api/v1")

    if args.api_key is None:
        args.api_key = ""
    if args.base_url is None:
        args.base_url = "https://openrouter.ai/api/v1"

    all_tasks = discover_tasks(tasks_root, output_root)
    selected = filter_tasks(all_tasks, args.domains, args.tasks, args.all_tasks)

    if not selected:
        print("No tasks selected. Use --task, --domain, or --all-tasks.")
        sys.exit(1)

    print(f"Agent: MLEvolve (MCGS)")
    print(f"Backend LLM: {args.model}")
    print(f"Base URL: {args.base_url}")
    print(f"Temperature: {args.temperature}")
    print(f"MCGS steps: {args.steps}")
    print(f"Time limit: {args.time_limit}s")
    print(f"Tasks: {len(selected)}")
    print(f"Output root: {output_root}")
    print(f"Max workers: {args.max_workers}")
    print()

    if args.dry_run:
        for t in selected:
            print(f"  [DRY] {t.key} -> {t.output_dir}")
        return

    print_lock = threading.Lock()
    results: list[TaskRunResult] = []
    if args.max_workers <= 1:
        for task_spec in selected:
            results.append(run_single_task(
                task_spec=task_spec,
                mlevolve_dir=mlevolve_dir,
                dataset_root=tasks_root,
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                temperature=args.temperature,
                steps=args.steps,
                time_limit=args.time_limit,
                print_lock=print_lock,
            ))
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(
                    run_single_task,
                    task_spec=ts,
                    mlevolve_dir=mlevolve_dir,
                    dataset_root=tasks_root,
                    model=args.model,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    temperature=args.temperature,
                    steps=args.steps,
                    time_limit=args.time_limit,
                    print_lock=print_lock,
                ): ts for ts in selected
            }
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda r: r.task_key)
    success = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {success} success, {failed} failed, {len(results)} total")
    print(f"{'='*60}")
    for r in results:
        marker = "✓" if r.status == "success" else "✗"
        print(f"  {marker} {r.task_key} ({r.duration_sec:.0f}s)" + (f" - {r.error}" if r.error else ""))

    summary_path = output_root / f"mlevolve_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "agent": "mlevolve", "model": args.model, "base_url": args.base_url,
        "temperature": args.temperature, "steps": args.steps,
        "time_limit": args.time_limit, "round_name": args.round_name,
        "output_root": str(output_root),
        "selected_tasks": [t.key for t in selected],
        "results": [asdict(r) for r in results],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
