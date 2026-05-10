#!/usr/bin/env python3
"""Run MLMaster2.0 agent on BioXArena tasks.

This wrapper keeps the BioXArena invocation contract aligned with the existing
MLEvolve runner:
  - same task selection CLI (`--task` / `--domain` / `--all-tasks`)
  - same output contract (`submission.csv`, `metrics.json`, `solution.py`)
  - same per-task output layout under `BioXArena-Output`

MLMaster2.0 itself is executed through EvoMaster:
    python run.py --agent ml_master_2 --config <generated.yaml> --task <description.md>
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass


SCRIPT_PATH = Path(__file__).resolve()
TRAINING_DIR = SCRIPT_PATH.parent
EVAL_ROOT = TRAINING_DIR.parent
WORKSPACE_ROOT = EVAL_ROOT.parent
DEFAULT_ROUND_NAME = "round1"
EVOMASTER_DIR = EVAL_ROOT / "agents" / "EvoMaster"
MLMASTER2_PLAYGROUND_DIR = EVOMASTER_DIR / "playground" / "ml_master_2"
MLMASTER2_TEMPLATE_CONFIG = (
    EVOMASTER_DIR / "configs" / "ml_master_2" / "deepseek-v3.2-example.yaml"
)

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
            description_path = public_dir / "description.md"
            if not description_path.exists():
                continue
            specs.append(
                TaskSpec(
                    domain=domain_dir.name,
                    task=task_dir.name,
                    task_dir=public_dir,
                    description_path=description_path,
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
        return sorted(all_tasks, key=lambda task: task.key)
    selected: dict[str, TaskSpec] = {}
    for task in all_tasks:
        if task.domain in selected_domains:
            selected[task.key] = task
        if task.key in selected_tasks:
            selected[task.key] = task
    return [selected[key] for key in sorted(selected)]


def load_env_file(env_path: Path) -> dict[str, str]:
    env_vars: dict[str, str] = {}
    if not env_path.exists():
        return env_vars
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        env_vars[key] = value
    return env_vars


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
    raise ValueError(f"`{key}` must be numeric in metrics.json.")


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


def collect_llm_usage(
    search_root: Path,
    extra_log_paths: list[Path] | None = None,
) -> tuple[int, int, bool]:
    total_input_tokens = 0
    total_output_tokens = 0
    found_usage = False

    if search_root.exists():
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
                total_input_tokens += parse_nonnegative_int(
                    payload.get("input_tokens", payload.get("prompt_tokens"))
                )
                total_output_tokens += parse_nonnegative_int(
                    payload.get("output_tokens", payload.get("completion_tokens"))
                )
                found_usage = True

    for log_path in extra_log_paths or []:
        if not log_path.exists():
            continue
        for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "Usage:" not in raw_line:
                continue
            payload_text = raw_line.split("Usage:", 1)[1].strip()
            try:
                payload = ast.literal_eval(payload_text)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            total_input_tokens += parse_nonnegative_int(
                payload.get("input_tokens", payload.get("prompt_tokens"))
            )
            total_output_tokens += parse_nonnegative_int(
                payload.get("output_tokens", payload.get("completion_tokens"))
            )
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
    for keyword, label in keyword_map:
        if keyword in source:
            return label
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


def resolve_solution_generated_time(task_start_time: float, solution_path: Path | None) -> float:
    if solution_path is None or not solution_path.exists():
        return time.time()
    try:
        return min(time.time(), max(task_start_time, solution_path.stat().st_mtime))
    except OSError:
        return time.time()


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
                "MLMaster2.0 did not emit phase-separated train/test timings; runner used aggregate candidate execution time from metric.txt and set any missing phase timing to 0."
            )
        else:
            if train_time_sec is None:
                train_time_sec = 0.0
            if test_time_sec is None:
                test_time_sec = 0.0
            fallback_notes.append(
                "MLMaster2.0 did not emit train/test timings; runner defaulted missing timing fields to 0."
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
        None
        if source_metrics_path is not None
        else "metrics.json synthesized by run_mlmaster2.0_agent_zdz.py from MLMaster2.0 artifacts.",
        None
        if usage_found
        else "MLMaster2.0 did not persist token usage metadata; input_tokens/output_tokens defaulted to 0.",
        fallback_notes,
        f"MLMaster2.0 search metadata: steps={steps}, code_model={model}, temperature={temperature}.",
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
                targets.add(output_dir / artifact_path.parts[0])

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


def clear_previous_outputs(output_dir: Path) -> None:
    clear_exported_submission_bundle(output_dir)
    for filename in (
        ["metrics.json", "solution.py"]
        + [
            "FAILED.json",
            "mlmaster2_cmd.txt",
            "mlmaster2_stdout.log",
            "mlmaster2_stderr.log",
            "mlmaster2_config.yaml",
            "metric_direction.json",
        ]
    ):
        path = output_dir / filename
        if path.exists():
            path.unlink()

    run_root = output_dir / "evomaster_run"
    if run_root.exists():
        shutil.rmtree(run_root)


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


def find_latest_artifact(search_root: Path, relative_pattern: str) -> Path | None:
    if not search_root.exists():
        return None
    matches = [path for path in search_root.rglob(relative_pattern) if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def export_best_artifacts(run_root: Path, output_dir: Path) -> tuple[Path | None, Path | None]:
    best_submission_dir = (
        (run_root / "workspaces" / "task_0" / "best_submission")
        if (run_root / "workspaces" / "task_0" / "best_submission" / "submission.csv").exists()
        else None
    )
    if best_submission_dir is None:
        best_submission = find_latest_artifact(run_root, "best_submission/submission.csv")
        if best_submission is not None:
            best_submission_dir = best_submission.parent
    best_submission = (
        best_submission_dir / "submission.csv"
        if best_submission_dir is not None
        else find_latest_artifact(run_root, "submission.csv")
    )

    best_solution = run_root / "workspaces" / "task_0" / "best_solution" / "best_solution.py"
    if not best_solution.exists():
        best_solution = (
            find_latest_artifact(run_root, "best_solution/best_solution.py")
            or find_latest_artifact(run_root, "best_solution.py")
        )

    exported_submission: Path | None = None
    exported_solution: Path | None = None

    if best_submission_dir is not None and (best_submission_dir / "submission.csv").exists():
        copy_bundle_dir_contents(best_submission_dir, output_dir)
        exported_submission = output_dir / "submission.csv"
    elif best_submission is not None and best_submission.exists():
        exported_submission = output_dir / "submission.csv"
        shutil.copy2(best_submission, exported_submission)

    if best_solution is not None and best_solution.exists():
        exported_solution = output_dir / "solution.py"
        shutil.copy2(best_solution, exported_solution)

    return exported_submission, exported_solution


def finalize_task_run(
    *,
    task_spec: TaskSpec,
    run_root: Path,
    task_start: float,
    model: str,
    temperature: float,
    steps: int,
    time_limit: int,
    exit_code: int | None,
    runtime_error: str | None,
    print_lock: threading.Lock,
) -> TaskRunResult:
    exported_submission, exported_solution = export_best_artifacts(
        run_root=run_root,
        output_dir=task_spec.output_dir,
    )
    source_metrics = find_latest_artifact(run_root, "metrics.json")
    metric_txt_path = find_latest_artifact(run_root, "metric.txt")
    input_tokens, output_tokens, usage_found = collect_llm_usage(
        run_root,
        extra_log_paths=[
            task_spec.output_dir / "mlmaster2_stderr.log",
            task_spec.output_dir / "mlmaster2_stdout.log",
        ],
    )

    duration = time.time() - task_start
    solution_path = exported_solution if exported_solution and exported_solution.exists() else None
    solution_generated_time = resolve_solution_generated_time(task_start, solution_path)

    metrics_error: str | None = None
    if solution_path is not None:
        try:
            write_final_metrics_json(
                output_path=task_spec.output_dir / "metrics.json",
                source_metrics_path=source_metrics,
                metric_txt_path=metric_txt_path,
                solution_path=solution_path,
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
        elif exit_code == 124:
            error = f"Timeout after {time_limit}s"
        elif runtime_error:
            error = runtime_error
        elif exit_code is not None and exit_code != 0:
            error = f"MLMaster2.0 exited with code {exit_code}"
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
            exit_code=exit_code,
        )

    with print_lock:
        print(f"[{status.upper()}] {task_spec.key} ({duration:.0f}s)")

    return TaskRunResult(
        task_key=task_spec.key,
        status=status,
        output_dir=str(task_spec.output_dir),
        error=error,
        duration_sec=duration,
    )


def infer_metric_direction(_: Path) -> tuple[bool, str]:
    return False, "user-specified global rule: all BioXArena tasks are higher-is-better"


def build_task_config(
    *,
    template_config_path: Path,
    output_dir: Path,
    task_spec: TaskSpec,
    tasks_root: Path,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    time_limit: int,
    grading_server_url: str | None,
    env_vars: dict[str, str],
) -> tuple[Path, bool, str]:
    config = yaml.safe_load(template_config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("MLMaster2.0 template config must be a YAML mapping.")

    is_lower_better, direction_reason = infer_metric_direction(task_spec.description_path)

    config["competition_id"] = task_spec.key
    config["exp_id"] = task_spec.key
    config["is_lower_better"] = is_lower_better
    config["data_root"] = str(tasks_root)
    config["grading_servers"] = [grading_server_url] if grading_server_url else []

    llm_cfg = config.setdefault("llm", {})
    if not isinstance(llm_cfg, dict):
        raise ValueError("llm section in template config must be a mapping.")
    llm_cfg["openrouter"] = {
        "provider": "openrouter",
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": 65536,
        "timeout": 3600,
        "max_retries": 3,
        "retry_delay": 1.0,
    }
    llm_cfg["default"] = "openrouter"

    agents_cfg = config.setdefault("agents", {})
    if not isinstance(agents_cfg, dict):
        raise ValueError("agents section in template config must be a mapping.")
    for agent_name, agent_cfg in agents_cfg.items():
        if isinstance(agent_cfg, dict):
            agent_cfg["llm"] = "openrouter"
            system_prompt = agent_cfg.get("system_prompt_file")
            if isinstance(system_prompt, str) and system_prompt:
                agent_cfg["system_prompt_file"] = str(
                    MLMASTER2_PLAYGROUND_DIR / system_prompt
                )
            user_prompt = agent_cfg.get("user_prompt_file")
            if isinstance(user_prompt, str) and user_prompt:
                agent_cfg["user_prompt_file"] = str(
                    MLMASTER2_PLAYGROUND_DIR / user_prompt
                )
            if agent_name == "prefetch":
                agent_cfg["wisdom_file"] = str(
                    MLMASTER2_PLAYGROUND_DIR / "example_wisdom" / "db.json"
                )

    session_cfg = config.setdefault("session", {})
    if not isinstance(session_cfg, dict):
        raise ValueError("session section in template config must be a mapping.")
    session_cfg["type"] = "local"
    local_cfg = session_cfg.setdefault("local", {})
    if not isinstance(local_cfg, dict):
        raise ValueError("session.local section in template config must be a mapping.")
    local_cfg["timeout"] = max(time_limit, 60)
    local_cfg["gpu_devices"] = ["0"]
    local_cfg["cpu_devices"] = "0-15"
    local_cfg["symlinks"] = {str(task_spec.task_dir): "input"}
    parallel_cfg = local_cfg.setdefault("parallel", {})
    if isinstance(parallel_cfg, dict):
        parallel_cfg["enabled"] = True
        parallel_cfg["max_parallel"] = 1
        parallel_cfg["split_workspace_for_exp"] = True

    embedding_cfg = config.setdefault("embedding", {})
    if isinstance(embedding_cfg, dict):
        embedding_cfg["type"] = "openai"
        openai_embedding_cfg = embedding_cfg.setdefault("openai", {})
        if isinstance(openai_embedding_cfg, dict):
            openai_embedding_cfg["api_key"] = (
                env_vars.get("OPENAI_EMBEDDING_API_KEY")
                or env_vars.get("OPENAI_API_KEY")
                or openai_embedding_cfg.get("api_key", "")
            )
            openai_embedding_cfg["base_url"] = (
                env_vars.get("OPENAI_EMBEDDING_BASE_URL")
                or env_vars.get("OPENAI_BASE_URL")
                or env_vars.get("GPT_BASE_URL")
                or openai_embedding_cfg.get("base_url", "")
            )

    config_path = output_dir / "mlmaster2_config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    write_json(
        output_dir / "metric_direction.json",
        {
            "task_key": task_spec.key,
            "is_lower_better": is_lower_better,
            "reason": direction_reason,
        },
    )
    return config_path, is_lower_better, direction_reason


def build_child_env(
    *,
    env_vars: dict[str, str],
    model: str,
    api_key: str,
    base_url: str,
    time_limit: int,
) -> dict[str, str]:
    child_env = os.environ.copy()
    child_env.update(env_vars)
    openai_api_key = env_vars.get("OPENAI_EMBEDDING_API_KEY") or env_vars.get("OPENAI_API_KEY")
    openai_base_url = (
        env_vars.get("OPENAI_EMBEDDING_BASE_URL")
        or env_vars.get("OPENAI_BASE_URL")
        or env_vars.get("GPT_BASE_URL")
    )

    child_env.setdefault("OPENROUTER_API_KEY", api_key or "")
    child_env.setdefault("OPENROUTER_MODEL", model)
    child_env.setdefault("GPT_CHAT_MODEL", env_vars.get("GPT_CHAT_MODEL") or model)
    child_env.setdefault("DEEPSEEK_API_KEY", env_vars.get("DEEPSEEK_API_KEY") or api_key or "")
    child_env.setdefault("DEEPSEEK_API_BASE", env_vars.get("DEEPSEEK_API_BASE") or base_url or "")
    if openai_api_key:
        child_env["OPENAI_API_KEY"] = openai_api_key
        child_env.setdefault("OPENAI_EMBEDDING_API_KEY", openai_api_key)
    if openai_base_url:
        child_env["OPENAI_BASE_URL"] = openai_base_url
        child_env.setdefault("OPENAI_EMBEDDING_BASE_URL", openai_base_url)
    child_env["MLMASTER2_RUN_TIMEOUT_SECONDS"] = str(time_limit)
    return child_env


def run_single_task(
    *,
    task_spec: TaskSpec,
    tasks_root: Path,
    template_config_path: Path,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    steps: int,
    time_limit: int,
    env_vars: dict[str, str],
    grading_server_url: str | None,
    print_lock: threading.Lock,
) -> TaskRunResult:
    task_start = time.time()
    task_spec.output_dir.mkdir(parents=True, exist_ok=True)
    clear_previous_outputs(task_spec.output_dir)

    try:
        from _core_pkg_guard import restore_core_packages

        restore_core_packages()
    except Exception:
        pass

    with print_lock:
        print(f"[START] {task_spec.key}")

    run_root: Path | None = None
    try:
        config_path, _, _ = build_task_config(
            template_config_path=template_config_path,
            output_dir=task_spec.output_dir,
            task_spec=task_spec,
            tasks_root=tasks_root,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            time_limit=time_limit,
            grading_server_url=grading_server_url,
            env_vars=env_vars,
        )

        run_root = task_spec.output_dir / "evomaster_run"
        run_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            "timeout",
            "--foreground",
            "--signal=TERM",
            "--kill-after=10s",
            f"{time_limit}s",
            sys.executable,
            str(EVOMASTER_DIR / "run.py"),
            "--agent",
            "ml_master_2",
            "--config",
            str(config_path),
            "--task",
            str(task_spec.description_path),
            "--run-dir",
            str(run_root),
        ]
        (task_spec.output_dir / "mlmaster2_cmd.txt").write_text(
            " \\\n  ".join(cmd), encoding="utf-8"
        )

        child_env = build_child_env(
            env_vars=env_vars,
            model=model,
            api_key=api_key,
            base_url=base_url,
            time_limit=time_limit,
        )

        result = subprocess.run(
            cmd,
            cwd=str(EVOMASTER_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )

        (task_spec.output_dir / "mlmaster2_stdout.log").write_text(
            result.stdout or "", encoding="utf-8"
        )
        (task_spec.output_dir / "mlmaster2_stderr.log").write_text(
            result.stderr or "", encoding="utf-8"
        )

        return finalize_task_run(
            task_spec=task_spec,
            run_root=run_root,
            task_start=task_start,
            model=model,
            temperature=temperature,
            steps=steps,
            time_limit=time_limit,
            exit_code=result.returncode,
            runtime_error=None,
            print_lock=print_lock,
        )
    except Exception as exc:
        if run_root is not None and run_root.exists():
            return finalize_task_run(
                task_spec=task_spec,
                run_root=run_root,
                task_start=task_start,
                model=model,
                temperature=temperature,
                steps=steps,
                time_limit=time_limit,
                exit_code=None,
                runtime_error=str(exc),
                print_lock=print_lock,
            )

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
            task_key=task_spec.key,
            status="failed",
            output_dir=str(task_spec.output_dir),
            error=str(exc),
            duration_sec=duration,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLMaster2.0 agent on BioXArena tasks")
    parser.add_argument("--prefix-dir", type=str, default=str(WORKSPACE_ROOT))
    parser.add_argument("--model", type=str, default="google/gemini-3.1-pro-preview")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=999)
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
    output_root = (
        prefix_dir / "BioXArena-Output" / f"mlmaster2.0__{model_dir_name}" / args.round_name
    )
    env_path = prefix_dir / "BioXArena" / ".env"
    env_vars = load_env_file(env_path)

    if args.api_key is None:
        args.api_key = env_vars.get("api_key", "")
    if args.base_url is None:
        args.base_url = env_vars.get("api_url", "https://openrouter.ai/api/v1")

    grading_server_port = os.environ.get("GRADING_SERVER_PORT", "").strip()
    grading_server_url = None
    if grading_server_port:
        grading_server_url = f"http://127.0.0.1:{grading_server_port}"

    all_tasks = discover_tasks(tasks_root, output_root)
    selected = filter_tasks(all_tasks, args.domains, args.tasks, args.all_tasks)

    if not selected:
        print("No tasks selected. Use --task, --domain, or --all-tasks.")
        sys.exit(1)

    print("Agent: MLMaster2.0")
    print(f"Backend LLM: {args.model}")
    print(f"Base URL: {args.base_url}")
    print(f"Temperature: {args.temperature}")
    print(f"Search steps: {args.steps}")
    print(f"Time limit: {args.time_limit}s")
    print(f"Tasks: {len(selected)}")
    print(f"Output root: {output_root}")
    print(f"Max workers: {args.max_workers}")
    print()

    if args.dry_run:
        for task in selected:
            print(f"  [DRY] {task.key} -> {task.output_dir}")
        return

    if not MLMASTER2_TEMPLATE_CONFIG.exists():
        print(f"Template config not found: {MLMASTER2_TEMPLATE_CONFIG}", file=sys.stderr)
        sys.exit(1)

    print_lock = threading.Lock()
    results: list[TaskRunResult] = []

    if args.max_workers <= 1:
        for task_spec in selected:
            results.append(
                run_single_task(
                    task_spec=task_spec,
                    tasks_root=tasks_root,
                    template_config_path=MLMASTER2_TEMPLATE_CONFIG,
                    model=args.model,
                    api_key=args.api_key or "",
                    base_url=args.base_url or "",
                    temperature=args.temperature,
                    steps=args.steps,
                    time_limit=args.time_limit,
                    env_vars=env_vars,
                    grading_server_url=grading_server_url,
                    print_lock=print_lock,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(
                    run_single_task,
                    task_spec=task_spec,
                    tasks_root=tasks_root,
                    template_config_path=MLMASTER2_TEMPLATE_CONFIG,
                    model=args.model,
                    api_key=args.api_key or "",
                    base_url=args.base_url or "",
                    temperature=args.temperature,
                    steps=args.steps,
                    time_limit=args.time_limit,
                    env_vars=env_vars,
                    grading_server_url=grading_server_url,
                    print_lock=print_lock,
                ): task_spec
                for task_spec in selected
            }
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda result: result.task_key)
    success = sum(1 for result in results if result.status == "success")
    failed = sum(1 for result in results if result.status == "failed")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {success} success, {failed} failed, {len(results)} total")
    print(f"{'=' * 60}")
    for result in results:
        marker = "✓" if result.status == "success" else "✗"
        print(
            f"  {marker} {result.task_key} ({result.duration_sec:.0f}s)"
            + (f" - {result.error}" if result.error else "")
        )

    summary_path = output_root / f"mlmaster2.0_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "agent": "mlmaster2.0",
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "steps": args.steps,
        "time_limit": args.time_limit,
        "round_name": args.round_name,
        "output_root": str(output_root),
        "selected_tasks": [task.key for task in selected],
        "results": [asdict(result) for result in results],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
