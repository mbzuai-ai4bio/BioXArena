#!/usr/bin/env python3
"""Evaluate BioXArena submissions produced by LLM training runs.

Examples:
    python evaluation/evaluate_llm_agents.py --model deepseek/deepseek-v3.2 --task sequence/active-regulatory-element
    python evaluation/evaluate_llm_agents.py --model deepseek/deepseek-v3.2 --domain sequence --max-workers 4
    python evaluation/evaluate_llm_agents.py --model deepseek/deepseek-v3.2 --all-tasks --max-workers 8
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve() # /<work_root>/BioXArena/evaluation/evaluate_llm_agents.py
EVAL_DIR = SCRIPT_PATH.parent # /<work_root>/BioXArena/evaluation
EVAL_ROOT = EVAL_DIR.parent # /<work_root>/BioXArena
WORKSPACE_ROOT = EVAL_ROOT.parent # /<work_root>
DEFAULT_PREFIX_ROOT = WORKSPACE_ROOT
DEFAULT_TASKS_ROOT = EVAL_ROOT / "tasks" # /<work_root>/BioXArena/tasks
DEFAULT_PRIVATE_ROOT_NAME = "BioXArena-Data-Private"
DEFAULT_OUTPUT_ROOT_NAME = "BioXArena-Output"
DEFAULT_ROUND_NAME = "round1"
RELATIVE_SUBMISSION_PATH_COLUMNS: dict[str, tuple[str, ...]] = {
    "imaging/amos-organ-segmentation": ("prediction_file",),
    "structure/protein-structure-prediction": ("coords_file",),
}

if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))


@dataclass(frozen=True)
class TaskSpec:
    domain: str
    task: str
    grade_path: Path
    answers_path: Path
    output_dir: Path

    @property
    def key(self) -> str:
        return f"{self.domain}/{self.task}"


@dataclass(frozen=True)
class MetricInfo:
    name: str
    direction: str
    normalization: str


@dataclass
class TaskEvaluationResult:
    task_key: str
    status: str
    score: float | None
    normalized_score: float
    metric: dict[str, str]
    remark: str
    results_path: str
    submission_path: str
    answers_path: str
    failed_marker_present: bool


def sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "__", value.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "unknown"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_grade_module(grade_path: Path, task_key: str) -> Any:
    module_name = "grade_" + sanitize_path_component(task_key)
    spec = importlib.util.spec_from_file_location(module_name, grade_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load grade module from {grade_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grade_fn = getattr(module, "grade", None)
    if not callable(grade_fn):
        raise RuntimeError(f"`grade` function not found in {grade_path}")
    return module


def discover_tasks(tasks_root: Path, private_root: Path, output_root: Path) -> list[TaskSpec]:
    task_specs: list[TaskSpec] = []
    for grade_path in sorted(tasks_root.glob("*/*/grade.py")):
        rel = grade_path.relative_to(tasks_root)
        domain, task = rel.parts[0], rel.parts[1]
        answers_path = private_root / domain / task / "private" / "answers.csv"
        output_dir = output_root / domain / task
        task_specs.append(
            TaskSpec(
                domain=domain,
                task=task,
                grade_path=grade_path,
                answers_path=answers_path,
                output_dir=output_dir,
            )
        )
    return task_specs


def resolve_task_selector(selector: str, tasks_by_key: dict[str, TaskSpec]) -> TaskSpec:
    selector = selector.strip()
    if selector in tasks_by_key:
        return tasks_by_key[selector]

    matches = [spec for spec in tasks_by_key.values() if spec.task == selector]
    if not matches:
        raise ValueError(f"Unknown task selector: {selector}")
    if len(matches) > 1:
        keys = ", ".join(spec.key for spec in matches)
        raise ValueError(f"Ambiguous task selector '{selector}'. Use one of: {keys}")
    return matches[0]


def select_tasks(
    all_tasks: list[TaskSpec],
    selected_domains: list[str],
    selected_tasks: list[str],
    run_all_tasks: bool,
) -> list[TaskSpec]:
    tasks_by_key = {task.key: task for task in all_tasks}
    selected: dict[str, TaskSpec] = {}

    for domain in selected_domains:
        domain_matches = [task for task in all_tasks if task.domain == domain]
        if not domain_matches:
            raise ValueError(f"Unknown domain: {domain}")
        for task in domain_matches:
            selected[task.key] = task

    for selector in selected_tasks:
        task = resolve_task_selector(selector, tasks_by_key)
        selected[task.key] = task

    if run_all_tasks or (not selected and not selected_domains and not selected_tasks):
        for task in all_tasks:
            selected[task.key] = task

    ordered_keys = sorted(selected)
    return [selected[key] for key in ordered_keys]


def infer_metric_info(grade_module: Any) -> MetricInfo:
    try:
        grade_source = inspect.getsource(grade_module.grade)
    except (OSError, TypeError):
        grade_source = ""
    source = grade_source.lower()

    if "grade_macro_roc_auc" in source:
        return MetricInfo(
            name="macro_roc_auc",
            direction="higher_is_better",
            normalization="already_in_0_1",
        )
    if "grade_roc_auc" in source:
        return MetricInfo(
            name="roc_auc",
            direction="higher_is_better",
            normalization="already_in_0_1",
        )
    if "grade_macro_f1" in source:
        return MetricInfo(
            name="macro_f1",
            direction="higher_is_better",
            normalization="already_in_0_1",
        )
    if "grade_accuracy" in source or "preds == labels" in source:
        return MetricInfo(
            name="accuracy",
            direction="higher_is_better",
            normalization="already_in_0_1",
        )
    if "grade_auprc" in source or "average_precision_score" in source:
        return MetricInfo(
            name="auprc",
            direction="higher_is_better",
            normalization="already_in_0_1",
        )
    if "grade_c_index" in source:
        return MetricInfo(
            name="c_index",
            direction="higher_is_better",
            normalization="already_in_0_1",
        )
    if (
        "grade_pearson" in source
        or "pearsonr(" in source
        or "np.corrcoef(" in source
        or "corrcoef(" in source
        or "pearson correlation" in source
    ):
        return MetricInfo(
            name="pearson",
            direction="higher_is_better",
            normalization="map_minus1_1_to_0_1",
        )
    if (
        "grade_spearman" in source
        or "spearmanr(" in source
        or "spearman rank correlation" in source
    ):
        return MetricInfo(
            name="spearman",
            direction="higher_is_better",
            normalization="map_minus1_1_to_0_1",
        )
    return MetricInfo(
        name="score",
        direction="higher_is_better",
        normalization="auto_by_score_range",
    )


def default_metric_info() -> MetricInfo:
    return MetricInfo(
        name="score",
        direction="higher_is_better",
        normalization="auto_by_score_range",
    )


def infer_task_metric_info(task_spec: TaskSpec) -> MetricInfo:
    try:
        grade_module = load_grade_module(task_spec.grade_path, task_spec.key)
    except Exception:
        return default_metric_info()
    return infer_metric_info(grade_module)


def normalize_score(score: float, metric_info: MetricInfo) -> float:
    if metric_info.normalization == "map_minus1_1_to_0_1":
        normalized = (score + 1.0) / 2.0
        return max(0.0, min(1.0, normalized))
    if metric_info.normalization == "already_in_0_1":
        return max(0.0, min(1.0, score))

    if -1.0 <= score <= 1.0:
        if score < 0.0:
            return (score + 1.0) / 2.0
        return score
    return max(0.0, min(1.0, score))


def build_results_payload(
    *,
    status: str,
    metric_info: MetricInfo,
    score: float | None,
    normalized_score: float,
    remark: str,
) -> dict[str, Any]:
    return {
        "Status": status,
        "Metric": {
            "name": metric_info.name,
            "direction": metric_info.direction,
            "normalization": metric_info.normalization,
        },
        "Score": score,
        "Normalized_Score": normalized_score,
        "Remark": remark,
    }


def resolve_submission_path(value: Any, output_dir: Path) -> Any:
    if pd.isna(value):
        return value

    candidate = Path(str(value))
    if candidate.is_absolute():
        return str(candidate)
    return str(output_dir / candidate)


def resolve_submission_file_columns(task_spec: TaskSpec, submission_df: pd.DataFrame) -> pd.DataFrame:
    path_columns = RELATIVE_SUBMISSION_PATH_COLUMNS.get(task_spec.key)
    if not path_columns:
        return submission_df

    resolved_df = submission_df.copy()
    for column in path_columns:
        if column not in resolved_df.columns:
            raise ValueError(
                f"Submission for {task_spec.key} is missing required path column: {column}"
            )
        resolved_df[column] = resolved_df[column].map(
            lambda value: resolve_submission_path(value, task_spec.output_dir)
        )
    return resolved_df


def evaluate_single_task(task_spec: TaskSpec, print_lock: threading.Lock) -> TaskEvaluationResult:
    task_spec.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = task_spec.output_dir / "submission.csv"
    failed_marker_path = task_spec.output_dir / "FAILED.json"
    results_path = task_spec.output_dir / "results.json"

    metric_info = infer_task_metric_info(task_spec)

    if failed_marker_path.exists():
        remark = "Training marked this task as failed because FAILED.json is present."
        payload = build_results_payload(
            status="Fail",
            metric_info=metric_info,
            score=None,
            normalized_score=0.0,
            remark=remark,
        )
        write_json(results_path, payload)
        with print_lock:
            print(f"[FAIL] {task_spec.key}: {remark}")
        return TaskEvaluationResult(
            task_key=task_spec.key,
            status="Fail",
            score=None,
            normalized_score=0.0,
            metric=payload["Metric"],
            remark=remark,
            results_path=str(results_path),
            submission_path=str(submission_path),
            answers_path=str(task_spec.answers_path),
            failed_marker_present=True,
        )

    if not submission_path.exists():
        remark = "Missing submission.csv in the task output directory."
        payload = build_results_payload(
            status="Fail",
            metric_info=metric_info,
            score=None,
            normalized_score=0.0,
            remark=remark,
        )
        write_json(results_path, payload)
        with print_lock:
            print(f"[FAIL] {task_spec.key}: {remark}")
        return TaskEvaluationResult(
            task_key=task_spec.key,
            status="Fail",
            score=None,
            normalized_score=0.0,
            metric=payload["Metric"],
            remark=remark,
            results_path=str(results_path),
            submission_path=str(submission_path),
            answers_path=str(task_spec.answers_path),
            failed_marker_present=False,
        )

    if not task_spec.answers_path.exists():
        remark = f"Missing answers.csv: {task_spec.answers_path}"
        payload = build_results_payload(
            status="Fail",
            metric_info=metric_info,
            score=None,
            normalized_score=0.0,
            remark=remark,
        )
        write_json(results_path, payload)
        with print_lock:
            print(f"[FAIL] {task_spec.key}: {remark}")
        return TaskEvaluationResult(
            task_key=task_spec.key,
            status="Fail",
            score=None,
            normalized_score=0.0,
            metric=payload["Metric"],
            remark=remark,
            results_path=str(results_path),
            submission_path=str(submission_path),
            answers_path=str(task_spec.answers_path),
            failed_marker_present=False,
        )

    try:
        grade_module = load_grade_module(task_spec.grade_path, task_spec.key)
        metric_info = infer_metric_info(grade_module)
        submission_df = pd.read_csv(submission_path)
        submission_df = resolve_submission_file_columns(task_spec, submission_df)
        answers_df = pd.read_csv(task_spec.answers_path)
        raw_score = grade_module.grade(submission_df, answers_df)
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(f"Grade returned a non-finite value: {raw_score!r}")
    except Exception as exc:
        remark = f"Evaluation failed: {exc}"
        payload = build_results_payload(
            status="Fail",
            metric_info=metric_info,
            score=None,
            normalized_score=0.0,
            remark=remark,
        )
        write_json(results_path, payload)
        with print_lock:
            print(f"[FAIL] {task_spec.key}: {remark}")
        return TaskEvaluationResult(
            task_key=task_spec.key,
            status="Fail",
            score=None,
            normalized_score=0.0,
            metric=payload["Metric"],
            remark=remark,
            results_path=str(results_path),
            submission_path=str(submission_path),
            answers_path=str(task_spec.answers_path),
            failed_marker_present=False,
        )

    normalized_score = normalize_score(score, metric_info)
    remark = "Evaluation succeeded."
    payload = build_results_payload(
        status="Succeed",
        metric_info=metric_info,
        score=score,
        normalized_score=normalized_score,
        remark=remark,
    )
    write_json(results_path, payload)
    with print_lock:
        print(
            f"[SUCCEED] {task_spec.key}: "
            f"{metric_info.name}={score:.6f}, normalized={normalized_score:.6f}"
        )
    return TaskEvaluationResult(
        task_key=task_spec.key,
        status="Succeed",
        score=score,
        normalized_score=normalized_score,
        metric=payload["Metric"],
        remark=remark,
        results_path=str(results_path),
        submission_path=str(submission_path),
        answers_path=str(task_spec.answers_path),
        failed_marker_present=False,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate BioXArena submission.csv outputs.")
    parser.add_argument("--task", action="append", default=[], help="Task selector, e.g. chemical-biology/cyp-inhibition-multi-label or cyp-inhibition-multi-label.")
    parser.add_argument("--domain", action="append", default=[], help="Run all tasks in a domain, e.g. chemical-biology.")
    parser.add_argument("--all-tasks", action="store_true", help="Evaluate all discovered tasks.")
    parser.add_argument("--list-tasks", action="store_true", help="List discoverable tasks and exit.")
    parser.add_argument("--model", required=True, help="Model name used by the training runner.")
    parser.add_argument("--round-name", default=DEFAULT_ROUND_NAME, help="Round/output subdirectory name appended after the model-specific output directory.")
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_ROOT, help="Shared prefix containing BioXArena-Output and BioXArena-Data-Private.")
    parser.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT, help="Path to BioXArena/tasks.")
    parser.add_argument("--private-root", type=Path, default=None, help="Base root for private answers. Overrides --prefix-dir/BioXArena-Data-Private.")
    parser.add_argument("--output-root", type=Path, default=None, help="Base output root. Overrides --prefix-dir/BioXArena-Output.")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of task evaluations to run in parallel.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    resolved_prefix_dir = args.prefix_dir.resolve()
    resolved_private_root = (
        args.private_root.resolve()
        if args.private_root
        else resolved_prefix_dir / DEFAULT_PRIVATE_ROOT_NAME
    )
    base_output_root = (
        args.output_root.resolve()
        if args.output_root
        else resolved_prefix_dir / DEFAULT_OUTPUT_ROOT_NAME
    )
    resolved_model_dir = sanitize_path_component(args.model)
    resolved_round_name = sanitize_path_component(args.round_name)
    resolved_output_root = base_output_root / resolved_model_dir / resolved_round_name

    all_tasks = discover_tasks(args.tasks_root, resolved_private_root, resolved_output_root)
    try:
        selected_tasks = select_tasks(all_tasks, args.domain, args.task, args.all_tasks)
    except ValueError as exc:
        parser.error(str(exc))

    if args.list_tasks:
        for task in selected_tasks or all_tasks:
            print(task.key)
        return 0

    if not selected_tasks:
        parser.error("No tasks selected.")

    print(f"Selected {len(selected_tasks)} task(s) for evaluation.")
    print(f"Model: {args.model}")
    print(f"Model dir: {resolved_model_dir}")
    print(f"Round name: {resolved_round_name}")
    print(f"Private root: {resolved_private_root}")
    print(f"Output root: {resolved_output_root}")

    max_workers = max(1, min(args.max_workers, len(selected_tasks)))
    print(f"Parallel workers: {max_workers}")

    resolved_output_root.mkdir(parents=True, exist_ok=True)
    print_lock = threading.Lock()
    results: list[TaskEvaluationResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(evaluate_single_task, task_spec=task, print_lock=print_lock): task
            for task in selected_tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:
                task.output_dir.mkdir(parents=True, exist_ok=True)
                results_path = task.output_dir / "results.json"
                fallback_metric = infer_task_metric_info(task)
                remark = f"Evaluation runner crashed: {exc}"
                payload = build_results_payload(
                    status="Fail",
                    metric_info=fallback_metric,
                    score=None,
                    normalized_score=0.0,
                    remark=remark,
                )
                write_json(results_path, payload)
                result = TaskEvaluationResult(
                    task_key=task.key,
                    status="Fail",
                    score=None,
                    normalized_score=0.0,
                    metric=payload["Metric"],
                    remark=remark,
                    results_path=str(results_path),
                    submission_path=str(task.output_dir / "submission.csv"),
                    answers_path=str(task.answers_path),
                    failed_marker_present=(task.output_dir / "FAILED.json").exists(),
                )
                with print_lock:
                    print(f"[FAIL] {task.key}: {remark}")
            results.append(result)

    ordered_results = sorted(results, key=lambda item: item.task_key)
    summary_path = resolved_output_root / (
        f"evaluation_summary_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.json"
    )
    summary_payload = {
        "model": args.model,
        "model_dir": resolved_model_dir,
        "round_name": resolved_round_name,
        "private_root": str(resolved_private_root),
        "resolved_output_root": str(resolved_output_root),
        "selected_tasks": [task.key for task in selected_tasks],
        "max_workers": max_workers,
        "results": [asdict(result) for result in ordered_results],
    }
    write_json(summary_path, summary_payload)

    success_count = sum(result.status == "Succeed" for result in ordered_results)
    fail_count = len(ordered_results) - success_count
    print(f"Completed {len(ordered_results)} task(s): {success_count} succeeded, {fail_count} failed.")
    print(f"Summary written to: {summary_path}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
