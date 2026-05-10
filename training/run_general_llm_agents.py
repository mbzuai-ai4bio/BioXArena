#!/usr/bin/env python3
"""Run general LLM agents on BioXArena tasks via an OpenAI-compatible API.

Examples:
    python run_general_llm_agents.py --task sequence/active-regulatory-element
    python run_general_llm_agents.py --domain sequence --max-workers 4
    python run_general_llm_agents.py --all-tasks --model anthropic/claude-3.7-sonnet
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


SCRIPT_PATH = Path(__file__).resolve()  # /<work_root>/BioXArena/training/run_general_llm_agents.py
TRAINING_DIR = SCRIPT_PATH.parent  # /<work_root>/BioXArena/training/
EVAL_ROOT = TRAINING_DIR.parent  # /<work_root>/BioXArena
WORKSPACE_ROOT = EVAL_ROOT.parent  # default prefix if none is provided explicitly  /<work_root>
DEFAULT_PREFIX_ROOT = WORKSPACE_ROOT
DEFAULT_TASKS_ROOT = EVAL_ROOT / "tasks"  # /<work_root>/BioXArena/tasks
DEFAULT_PROMPT_PATH = EVAL_ROOT / "prompts" / "unified_eval_prompt.py"
DEFAULT_ENV_PATH = EVAL_ROOT / ".env"
DEFAULT_MODEL = "openai/gpt-5.4-mini"
DEFAULT_ROUND_NAME = "round1"
API_KEY_ENV = "api_key"
API_URL_ENV = "api_url"
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_REQUEST_TIMEOUT_SEC = 180
DEFAULT_RUN_TIMEOUT_SEC = 6 * 60 * 60
REQUIRED_OUTPUT_LABELS = [
    "submission.csv",
    "metrics.json",
    "solution.py",
]


@dataclass(frozen=True)
class TaskSpec:
    domain: str
    task: str
    grade_path: Path
    task_dir: Path
    description_path: Path
    output_dir: Path

    @property
    def key(self) -> str:
        return f"{self.domain}/{self.task}"


@dataclass
class AttemptRecord:
    attempt: int
    request_ok: bool
    execution_ok: bool
    missing_required_outputs: list[str]
    error: str | None = None
    exit_code: int | None = None
    duration_sec: float | None = None
    solution_path: str | None = None


@dataclass
class TaskRunResult:
    task_key: str
    status: str
    attempts_used: int
    output_dir: str
    submission_path: str | None
    metrics_path: str | None
    error: str | None
    attempts: list[AttemptRecord]


class LLMClient:
    def __init__(
        self,
        api_key: str,
        api_url: str,
        model: str,
        timeout_sec: int,
        enable_reasoning: bool,
        temperature: float | None,
        max_completion_tokens: int | None,
        seed: int | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.timeout_sec = timeout_sec
        self.enable_reasoning = enable_reasoning
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.seed = seed
        self.client = OpenAI(
            base_url=self.api_url,
            api_key=self.api_key,
            timeout=self.timeout_sec,
        )

    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "extra_body": {"reasoning": {"enabled": self.enable_reasoning}},
        }
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature
        if self.max_completion_tokens is not None:
            request_kwargs["max_tokens"] = self.max_completion_tokens
        if self.seed is not None:
            request_kwargs["seed"] = self.seed

        response = self.client.chat.completions.create(
            **request_kwargs,
        )
        return response.model_dump(mode="json", exclude_none=True)


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_prompt_template(prompt_path: Path) -> str:
    spec = importlib.util.spec_from_file_location("unified_eval_prompt", prompt_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load prompt module from {prompt_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompt_template = getattr(module, "PROMPT_TEMPLATE", None)
    if not isinstance(prompt_template, str):
        raise RuntimeError(f"PROMPT_TEMPLATE not found in {prompt_path}")
    return prompt_template


def sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "__", value.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "unknown"


def discover_tasks(tasks_root: Path, data_root: Path, output_root: Path) -> list[TaskSpec]:
    task_specs: list[TaskSpec] = []
    for grade_path in sorted(tasks_root.glob("*/*/grade.py")):
        rel = grade_path.relative_to(tasks_root)
        domain, task = rel.parts[0], rel.parts[1]
        task_dir = data_root / domain / task / "public"
        description_path = task_dir / "description.md"
        output_dir = output_root / domain / task
        task_specs.append(
            TaskSpec(
                domain=domain,
                task=task,
                grade_path=grade_path,
                task_dir=task_dir,
                description_path=description_path,
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


def build_user_prompt(prompt_template: str, task_spec: TaskSpec) -> str:
    description = task_spec.description_path.read_text(encoding="utf-8")
    prompt = prompt_template.format(
        task_dir=str(task_spec.task_dir),
        output_dir=str(task_spec.output_dir),
        description=description,
    )
    output_instruction = """

# Response Format
Return only one complete Python program for `{output_dir}/solution.py`.
- Put the full code inside exactly one fenced ```python code block.
- Do not include any explanation before or after the code block.
- The code must be directly executable with `python solution.py`.
- The code must create `{output_dir}` if needed and save all required outputs there.
""".strip()
    return prompt + "\n\n" + output_instruction.format(output_dir=task_spec.output_dir)


def normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def extract_code_block(text: str) -> str:
    pattern = re.compile(r"```(?:python)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip() + "\n"
    stripped = text.strip()
    if not stripped:
        raise ValueError("Model response was empty.")
    if "def " in stripped or "import " in stripped or "from " in stripped:
        return stripped + ("\n" if not stripped.endswith("\n") else "")
    raise ValueError("Model response did not contain a Python code block.")


def truncate_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


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


def extract_token_usage(raw_response: dict[str, Any]) -> tuple[int, int, bool]:
    usage = raw_response.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, False

    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    return parse_nonnegative_int(input_tokens), parse_nonnegative_int(output_tokens), True


def validate_required_metric_fields(payload: dict[str, Any]) -> None:
    required_fields = ("model_used", "model_param_count", "notes")
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        raise ValueError(
            "metrics.json is missing required field(s): " + ", ".join(missing_fields)
        )


def inject_wrapper_total_time(
    metrics_path: Path,
    task_start_time: float,
    solution_generated_time: float,
    input_tokens: int,
    output_tokens: int,
    usage_missing: bool,
) -> None:
    payload = read_json(metrics_path)
    if not isinstance(payload, dict):
        raise ValueError("metrics.json must contain a JSON object.")

    validate_required_metric_fields(payload)
    train_time_sec = parse_numeric_metric(payload, "train_time_sec")
    test_time_sec = parse_numeric_metric(payload, "test_time_sec")
    solution_generation_time_sec = max(0.0, solution_generated_time - task_start_time)
    payload["solution_generation_time_sec"] = round(solution_generation_time_sec, 6)
    payload["code_total_time_sec"] = round(solution_generation_time_sec + train_time_sec + test_time_sec, 6)
    payload["input_tokens"] = input_tokens
    payload["output_tokens"] = output_tokens
    if usage_missing:
        warning_note = (
            "WARNING: At least one API response was missing usage metadata, so "
            "input_tokens/output_tokens may be incomplete."
        )
        existing_notes = payload.get("notes")
        if existing_notes is None:
            payload["notes"] = warning_note
        elif isinstance(existing_notes, str):
            if warning_note not in existing_notes:
                separator = "\n" if existing_notes.strip() else ""
                payload["notes"] = existing_notes + separator + warning_note
        else:
            payload["notes"] = f"{existing_notes}\n{warning_note}"
    write_json(metrics_path, payload)


def failed_marker_path(output_dir: Path) -> Path:
    return output_dir / "FAILED.json"


def clear_failed_marker(output_dir: Path) -> None:
    marker_path = failed_marker_path(output_dir)
    if marker_path.exists():
        marker_path.unlink()


def write_failed_marker(
    task_spec: TaskSpec,
    attempts_used: int,
    error: str | None,
    attempts: list[AttemptRecord],
) -> None:
    payload = {
        "task_key": task_spec.key,
        "status": "failed",
        "attempts_used": attempts_used,
        "output_dir": str(task_spec.output_dir),
        "final_error": error,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "attempts": [asdict(attempt) for attempt in attempts],
    }
    if attempts:
        payload["last_missing_outputs"] = attempts[-1].missing_required_outputs
        payload["last_exit_code"] = attempts[-1].exit_code
        payload["last_solution_path"] = attempts[-1].solution_path
    write_json(failed_marker_path(task_spec.output_dir), payload)


def build_repair_message(
    task_spec: TaskSpec,
    error: str,
    missing_outputs: list[str],
    stderr_text: str,
    extra_requirements: list[str] | None = None,
) -> str:
    lines = [
        f"The previous attempt for task `{task_spec.key}` did not satisfy the requirements.",
        "",
        "Please return a corrected full replacement for `solution.py`.",
        "Return only one fenced ```python code block and no additional text.",
    ]
    if extra_requirements:
        lines.append("")
        lines.append("Additional requirements for the next attempt:")
        for item in extra_requirements:
            lines.append(f"- {item}")
    if missing_outputs:
        lines.append("")
        lines.append("Missing required outputs:")
        for item in missing_outputs:
            lines.append(f"- {item}")
    if error:
        lines.append("")
        lines.append("Execution or validation error:")
        lines.append("```text")
        lines.append(truncate_text(error, limit=3000))
        lines.append("```")
    if stderr_text:
        lines.append("")
        lines.append("stderr tail:")
        lines.append("```text")
        lines.append(truncate_text(stderr_text, limit=3000))
        lines.append("```")
    return "\n".join(lines)


def verify_required_outputs(output_dir: Path) -> list[str]:
    missing: list[str] = []
    for filename in ("submission.csv", "metrics.json", "solution.py"):
        if not (output_dir / filename).exists():
            missing.append(filename)
    return missing


def run_solution_file(
    solution_path: Path,
    cwd: Path,
    timeout_sec: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(solution_path)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
        check=False,
    )


def ensure_task_inputs(task_spec: TaskSpec) -> None:
    if not task_spec.task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_spec.task_dir}")
    if not task_spec.description_path.exists():
        raise FileNotFoundError(f"description.md not found: {task_spec.description_path}")
    sample_submission = task_spec.task_dir / "sample_submission.csv"
    if not sample_submission.exists():
        raise FileNotFoundError(f"sample_submission.csv not found: {sample_submission}")


def run_single_task(
    task_spec: TaskSpec,
    prompt_template: str,
    client: LLMClient,
    max_attempts: int,
    run_timeout_sec: int,
    dry_run: bool,
    print_lock: threading.Lock,
    task_timeout_sec: int = 7200,
) -> TaskRunResult:
    attempts: list[AttemptRecord] = []
    task_start_time = time.time()
    task_deadline = task_start_time + task_timeout_sec
    # Restore core packages to pinned versions before each task to undo any
    # drift caused by previous LLM-issued `pip install` commands.
    try:
        from _core_pkg_guard import restore_core_packages
        restore_core_packages()
    except Exception:
        pass
    ensure_task_inputs(task_spec)
    task_spec.output_dir.mkdir(parents=True, exist_ok=True)

    initial_prompt = build_user_prompt(prompt_template, task_spec)
    messages: list[dict[str, Any]] = [{"role": "user", "content": initial_prompt}]
    total_input_tokens = 0
    total_output_tokens = 0
    usage_missing = False

    if dry_run:
        prompt_path = task_spec.output_dir / "dry_run_prompt.txt"
        prompt_path.write_text(initial_prompt, encoding="utf-8")
        return TaskRunResult(
            task_key=task_spec.key,
            status="dry_run",
            attempts_used=0,
            output_dir=str(task_spec.output_dir),
            submission_path=None,
            metrics_path=None,
            error=None,
            attempts=[],
        )

    for attempt_index in range(1, max_attempts + 1):
        # Enforce per-task hard wall-clock across all attempts.
        remaining = task_deadline - time.time()
        if remaining <= 0:
            with print_lock:
                print(f"[TASK-TIMEOUT] {task_spec.key}: wall-clock {task_timeout_sec}s exceeded before attempt {attempt_index}")
            attempts.append(
                AttemptRecord(
                    attempt=attempt_index,
                    request_ok=False,
                    execution_ok=False,
                    missing_required_outputs=REQUIRED_OUTPUT_LABELS.copy(),
                    error=f"Task wall-clock exceeded {task_timeout_sec}s",
                    duration_sec=0.0,
                )
            )
            break
        # Cap this attempt's subprocess timeout to whatever is left in the task budget.
        effective_run_timeout = max(1, min(run_timeout_sec, int(remaining)))
        request_start = time.time()
        request_payload_path = task_spec.output_dir / f"api_request_attempt_{attempt_index:02d}.json"
        response_payload_path = task_spec.output_dir / f"api_response_attempt_{attempt_index:02d}.json"
        solution_attempt_path = task_spec.output_dir / f"solution_attempt_{attempt_index:02d}.py"
        stdout_path = task_spec.output_dir / f"execution_attempt_{attempt_index:02d}.stdout.log"
        stderr_path = task_spec.output_dir / f"execution_attempt_{attempt_index:02d}.stderr.log"

        write_json(request_payload_path, {"messages": messages, "model": client.model})

        try:
            raw_response = client.chat(messages)
            attempt_input_tokens, attempt_output_tokens, has_usage = extract_token_usage(raw_response)
            total_input_tokens += attempt_input_tokens
            total_output_tokens += attempt_output_tokens
            write_json(response_payload_path, raw_response)

            if not has_usage:
                usage_missing = True
                with print_lock:
                    print(
                        f"[WARN] {task_spec.key} attempt {attempt_index}: API response missing usage; "
                        "input_tokens/output_tokens may be incomplete."
                    )

            message = raw_response["choices"][0]["message"]
            content = normalize_content(message.get("content"))
            code = extract_code_block(content)
            solution_attempt_path.write_text(code, encoding="utf-8")
            (task_spec.output_dir / "solution.py").write_text(code, encoding="utf-8")
            solution_generated_time = time.time()

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": message.get("content"),
            }
            if "reasoning_details" in message:
                assistant_message["reasoning_details"] = message.get("reasoning_details")
            messages.append(assistant_message)

        except Exception as exc:
            error_text = f"API request/code extraction failed: {exc}"
            attempts.append(
                AttemptRecord(
                    attempt=attempt_index,
                    request_ok=False,
                    execution_ok=False,
                    missing_required_outputs=REQUIRED_OUTPUT_LABELS.copy(),
                    error=error_text,
                    duration_sec=time.time() - request_start,
                )
            )
            messages.append(
                {
                    "role": "user",
                        "content": build_repair_message(
                            task_spec=task_spec,
                            error=error_text,
                            missing_outputs=REQUIRED_OUTPUT_LABELS.copy(),
                            stderr_text="",
                        ),
                    }
            )
            continue

        try:
            completed = run_solution_file(
                solution_path=task_spec.output_dir / "solution.py",
                cwd=task_spec.output_dir,
                timeout_sec=effective_run_timeout,
            )
            stdout_path.write_text(completed.stdout or "", encoding="utf-8")
            stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        except subprocess.TimeoutExpired as exc:
            timeout_error = f"Execution timed out after {run_timeout_sec} seconds."
            timeout_stdout = ensure_text(exc.stdout)
            timeout_stderr = ensure_text(exc.stderr)
            stdout_path.write_text(timeout_stdout, encoding="utf-8")
            stderr_path.write_text(timeout_stderr, encoding="utf-8")
            attempts.append(
                AttemptRecord(
                    attempt=attempt_index,
                    request_ok=True,
                    execution_ok=False,
                    missing_required_outputs=REQUIRED_OUTPUT_LABELS.copy(),
                    error=timeout_error,
                    duration_sec=time.time() - request_start,
                    solution_path=str(solution_attempt_path),
                )
            )
            messages.append(
                {
                    "role": "user",
                        "content": build_repair_message(
                            task_spec=task_spec,
                            error=timeout_error,
                            missing_outputs=REQUIRED_OUTPUT_LABELS.copy(),
                            stderr_text=timeout_stderr,
                            extra_requirements=[
                                "The previous attempt exceeded the configured execution timeout.",
                                f"Your solution must finish within the configured `--run-timeout-sec` limit for this run ({run_timeout_sec} seconds).",
                                "Optimize runtime substantially and avoid expensive full-dataset or full-image passes when a simpler approximation can satisfy the task.",
                            ],
                        ),
                    }
            )
            continue

        missing_outputs = verify_required_outputs(task_spec.output_dir)
        metrics_injection_error: str | None = None
        if completed.returncode == 0 and not missing_outputs:
            try:
                inject_wrapper_total_time(
                    metrics_path=task_spec.output_dir / "metrics.json",
                    task_start_time=task_start_time,
                    solution_generated_time=solution_generated_time,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    usage_missing=usage_missing,
                )
            except Exception as exc:
                metrics_injection_error = f"metrics.json validation/injection failed: {exc}"

        if metrics_injection_error:
            failure_reason = metrics_injection_error
            missing_outputs.append("valid metrics.json with required fields and numeric train_time_sec/test_time_sec")
            execution_ok = False
        else:
            execution_ok = completed.returncode == 0 and not missing_outputs
            failure_reason: str | None
            if execution_ok:
                failure_reason = None
            elif completed.returncode != 0:
                failure_reason = f"Process exited with code {completed.returncode}"
            else:
                failure_reason = "Missing required outputs: " + ", ".join(missing_outputs)
        attempt_record = AttemptRecord(
            attempt=attempt_index,
            request_ok=True,
            execution_ok=execution_ok,
            missing_required_outputs=missing_outputs,
            error=failure_reason,
            exit_code=completed.returncode,
            duration_sec=time.time() - request_start,
            solution_path=str(solution_attempt_path),
        )
        attempts.append(attempt_record)

        if execution_ok:
            clear_failed_marker(task_spec.output_dir)
            with print_lock:
                print(f"[SUCCESS] {task_spec.key} (attempt {attempt_index})")
            return TaskRunResult(
                task_key=task_spec.key,
                status="success",
                attempts_used=attempt_index,
                output_dir=str(task_spec.output_dir),
                submission_path=str(task_spec.output_dir / "submission.csv"),
                metrics_path=str(task_spec.output_dir / "metrics.json"),
                error=None,
                attempts=attempts,
            )

        repair_message = build_repair_message(
            task_spec=task_spec,
            error=failure_reason or "",
            missing_outputs=missing_outputs,
            stderr_text=completed.stderr or "",
        )
        messages.append({"role": "user", "content": repair_message})

    final_error = attempts[-1].error if attempts else "Unknown failure."
    write_failed_marker(
        task_spec=task_spec,
        attempts_used=max_attempts,
        error=final_error,
        attempts=attempts,
    )
    with print_lock:
        print(f"[FAILED] {task_spec.key} after {max_attempts} attempt(s)")
    return TaskRunResult(
        task_key=task_spec.key,
        status="failed",
        attempts_used=max_attempts,
        output_dir=str(task_spec.output_dir),
        submission_path=str(task_spec.output_dir / "submission.csv") if (task_spec.output_dir / "submission.csv").exists() else None,
        metrics_path=str(task_spec.output_dir / "metrics.json") if (task_spec.output_dir / "metrics.json").exists() else None,
        error=final_error,
        attempts=attempts,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run general LLM agents on BioXArena tasks.")
    parser.add_argument("--task", action="append", default=[], help="Task selector, e.g. sequence/active-regulatory-element or active-regulatory-element.")
    parser.add_argument("--domain", action="append", default=[], help="Run all tasks in a domain, e.g. sequence.")
    parser.add_argument("--all-tasks", action="store_true", help="Run all 76 BioXArena tasks.")
    parser.add_argument("--list-tasks", action="store_true", help="List discoverable tasks and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Write prompts without calling the API or executing generated code.")
    parser.add_argument("--model", default=None, help="Model name understood by the configured API endpoint.")
    parser.add_argument("--round-name", default=DEFAULT_ROUND_NAME, help="Round/output subdirectory name appended after the model-specific output directory.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH, help="Path to the .env file.")
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_ROOT, help="Shared prefix for BioXArena-Data-Public and BioXArena-Output. Defaults to the parent of BioXArena.")
    parser.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT, help="Path to BioXArena/tasks.")
    parser.add_argument("--data-root", type=Path, default=None, help="Path to the BioXArena-Data-Public data root. Overrides --prefix-dir/BioXArena-Data-Public.")
    parser.add_argument("--output-root", type=Path, default=None, help="Base output root. The script appends <model_name>/<round_name>/ under this directory.")
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH, help="Path to unified_eval_prompt.py.")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of tasks to run in parallel.")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS, help="Max repair attempts per task.")
    parser.add_argument("--request-timeout-sec", type=int, default=DEFAULT_REQUEST_TIMEOUT_SEC, help="API HTTP timeout.")
    parser.add_argument("--run-timeout-sec", type=int, default=DEFAULT_RUN_TIMEOUT_SEC, help="Timeout for each generated solution.py execution.")
    parser.add_argument("--task-timeout-sec", type=int, default=7200, help="Hard wall-clock per task across all attempts (default 7200=2h).")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature for each request. Provided by the external shell script if needed.")
    parser.add_argument("--max-completion-tokens", type=int, default=None, help="Optional token budget for each completion.")
    parser.add_argument("--disable-reasoning", action="store_true", help="Disable reasoning mode.")
    parser.add_argument("--seed", type=int, default=None, help="Optional integer seed forwarded to the API for reproducibility.")
    parser.add_argument("--model-dir", default=None, help="Override the model-name-based output subdirectory. If set, output goes to <output_root>/<model_dir>/<round_name>.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    load_env_file(args.env_file)
    resolved_model = args.model or DEFAULT_MODEL
    api_key = os.environ.get(API_KEY_ENV)
    resolved_api_url = os.environ.get(API_URL_ENV)
    resolved_round_name = sanitize_path_component(args.round_name)
    resolved_model_dir = sanitize_path_component(args.model_dir) if args.model_dir else sanitize_path_component(resolved_model)
    resolved_prefix_dir = args.prefix_dir.resolve()
    resolved_data_root = args.data_root.resolve() if args.data_root else resolved_prefix_dir / "BioXArena-Data-Public"
    base_output_root = args.output_root.resolve() if args.output_root else resolved_prefix_dir / "BioXArena-Output"
    resolved_output_root = base_output_root / resolved_model_dir / resolved_round_name
    if not api_key and not args.dry_run:
        parser.error(
            f"{API_KEY_ENV} is not set. Put it in {args.env_file} "
            "or export it in the environment."
        )
    if not resolved_api_url and not args.dry_run:
        parser.error(
            f"{API_URL_ENV} is not set. Put it in {args.env_file} "
            "or export it in the environment."
        )

    prompt_template = load_prompt_template(args.prompt_path)
    all_tasks = discover_tasks(args.tasks_root, resolved_data_root, resolved_output_root)
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

    missing_task_inputs = [
        task.key
        for task in selected_tasks
        if not task.task_dir.exists() or not task.description_path.exists()
    ]
    if missing_task_inputs:
        parser.error(
            "Missing task inputs for: "
            + ", ".join(missing_task_inputs)
            + ". Check --prefix-dir, --data-root, and task availability."
        )

    max_workers = max(1, min(args.max_workers, len(selected_tasks)))
    client = LLMClient(
        api_key=api_key or "",
        api_url=resolved_api_url or "",
        model=resolved_model,
        timeout_sec=args.request_timeout_sec,
        enable_reasoning=not args.disable_reasoning,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        seed=args.seed,
    )

    print(f"Selected {len(selected_tasks)} task(s).")
    print(f"Model: {resolved_model}")
    print(f"Model dir: {resolved_model_dir}")
    print(f"Round name: {resolved_round_name}")
    print(f"Temperature: {args.temperature}")
    print(f"Seed: {args.seed}")
    print(f"Prefix dir: {resolved_prefix_dir}")
    print(f"Data root: {resolved_data_root}")
    print(f"Base output root: {base_output_root}")
    print(f"Output root: {resolved_output_root}")
    print(f"Parallel workers: {max_workers}")

    started_at = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    summary_path = resolved_output_root / f"general_llm_runner_summary_{started_at}.json"
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    print_lock = threading.Lock()
    results: list[TaskRunResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                run_single_task,
                task_spec=task,
                prompt_template=prompt_template,
                client=client,
                max_attempts=args.max_attempts,
                run_timeout_sec=args.run_timeout_sec,
                task_timeout_sec=args.task_timeout_sec,
                dry_run=args.dry_run,
                print_lock=print_lock,
            ): task
            for task in selected_tasks
        }

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:
                task.output_dir.mkdir(parents=True, exist_ok=True)
                write_failed_marker(
                    task_spec=task,
                    attempts_used=0,
                    error=str(exc),
                    attempts=[],
                )
                result = TaskRunResult(
                    task_key=task.key,
                    status="failed",
                    attempts_used=0,
                    output_dir=str(task.output_dir),
                    submission_path=None,
                    metrics_path=None,
                    error=str(exc),
                    attempts=[],
                )
                with print_lock:
                    print(f"[FAILED] {task.key}: {exc}")
            results.append(result)

    ordered_results = sorted(results, key=lambda item: item.task_key)
    summary_payload = {
        "model": resolved_model,
        "model_dir": resolved_model_dir,
        "round_name": resolved_round_name,
        "api_url": resolved_api_url,
        "base_output_root": str(base_output_root),
        "resolved_output_root": str(resolved_output_root),
        "selected_tasks": [task.key for task in selected_tasks],
        "max_workers": max_workers,
        "max_attempts": args.max_attempts,
        "dry_run": args.dry_run,
        "results": [
            {
                **{
                    key: value
                    for key, value in asdict(result).items()
                    if key != "attempts"
                },
                "attempts": [asdict(attempt) for attempt in result.attempts],
            }
            for result in ordered_results
        ],
    }
    write_json(summary_path, summary_payload)

    success_count = sum(result.status in {"success", "dry_run"} for result in ordered_results)
    print(f"Completed {len(ordered_results)} task(s): {success_count} succeeded/dry-run, {len(ordered_results) - success_count} failed.")
    print(f"Summary written to: {summary_path}")

    return 0 if success_count == len(ordered_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
