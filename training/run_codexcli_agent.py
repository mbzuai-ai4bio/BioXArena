#!/usr/bin/env python3
"""Run OpenAI Codex CLI (codex exec) as a full coding-agent baseline on BioXArena tasks.

Codex CLI is a general-purpose coding-agent CLI: it owns its own tool-use loop (shell
exec, file read/write, patch application) and its own internal retry/repair behavior.
This runner's job is only to:
  1. Stage the task (point Codex at the public task capsule + a private output dir).
  2. Invoke `codex exec` non-interactively against OpenRouter (OpenAI Responses API),
     enforcing a hard wall-clock kill (default 2h).
  3. Collect submission.csv / metrics.json / solution.py, the full JSONL transcript,
     and token usage parsed from Codex's own `turn.completed` event.

Codex is pointed at OpenRouter via an *isolated* CODEX_HOME containing a small
config.toml with a custom model_provider. This never touches the operator's personal
~/.codex/config.toml. Reasoning effort is left at the Codex default.

Examples:
    python run_codexcli_agent.py --model openai/gpt-5.6-sol --task chemical-biology/bace1-binding-affinity
    python run_codexcli_agent.py --model openai/gpt-6-astra --all-tasks --max-workers 4 --num-gpus 4
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openrouter_config import OPENROUTER_BASE_URL, resolve_api_key

# ---------------------------------------------------------------------------
# Paths (mirror the Biomni/STELLA runner layout)
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
TRAINING_DIR = SCRIPT_PATH.parent
EVAL_ROOT = TRAINING_DIR.parent  # repository root
WORKSPACE_ROOT = EVAL_ROOT.parent
DEFAULT_TASKS_ROOT = EVAL_ROOT / "tasks"
DEFAULT_PROMPT_PATH = EVAL_ROOT / "prompts" / "unified_eval_prompt.py"
DEFAULT_ROUND_NAME = "round1"
# CLI state (config, sessions, shell snapshots that may contain environment variables) is kept
# outside the results tree so that packaged outputs never include it.
RUNTIME_ROOT = EVAL_ROOT / ".runtime"

DEFAULT_PROXY_BASE_URL = OPENROUTER_BASE_URL
DEFAULT_MODEL = "openai/gpt-5.6-sol"

REQUIRED_OUTPUT_LABELS = ["submission.csv", "metrics.json", "solution.py"]

CODEX_APPENDIX = """

# Coding-agent CLI notes
You are running as an autonomous coding-agent CLI (Codex) with direct shell and file
access. Work entirely within the directories given to you above ({task_dir} for
reading public data, {output_dir} for writing your solution and outputs). Do not
attempt to list, guess, or read any path outside of these two directories -- in
particular, never search for or open files named like "answers", "private", or
"labels" anywhere on the filesystem. Finish only once submission.csv, metrics.json,
and solution.py all exist and are valid in {output_dir}.
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
    input_tokens: int = 0
    output_tokens: int = 0
    exit_code: int | None = None


# ---------------------------------------------------------------------------
# Task discovery (identical contract to run_biomni_agent.py / run_stella_agent.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Prompt builder (reuses the same unified_eval_prompt.py template as every other
# agent in this project, plus a short CLI-specific isolation reminder appendix)
# ---------------------------------------------------------------------------
def load_prompt_template(prompt_path: Path) -> str:
    ns: dict[str, Any] = {}
    exec(prompt_path.read_text(encoding="utf-8"), ns)
    return ns["PROMPT_TEMPLATE"]


def build_prompt(prompt_template: str, task_spec: TaskSpec) -> str:
    description = task_spec.description_path.read_text(encoding="utf-8")
    base = prompt_template.format(
        task_dir=str(task_spec.task_dir),
        output_dir=str(task_spec.output_dir),
        description=description,
    )
    appendix = CODEX_APPENDIX.format(task_dir=str(task_spec.task_dir), output_dir=str(task_spec.output_dir))
    return base + appendix


# ---------------------------------------------------------------------------
# Credentials / isolated Codex config
# ---------------------------------------------------------------------------
def ensure_codex_home(codex_home: Path, base_url: str, model: str) -> None:
    """Write an isolated CODEX_HOME/config.toml pointing at OpenRouter.

    Idempotent: safe to call before every task/worker. Never touches the operator's
    real ~/.codex directory because CODEX_HOME is overridden per-subprocess.
    """
    codex_home.mkdir(parents=True, exist_ok=True)
    config_path = codex_home / "config.toml"
    config_text = f"""model_provider = "openrouter"
model = "{model}"
approval_policy = "never"
sandbox_mode = "danger-full-access"
disable_response_storage = true

[model_providers.openrouter]
name = "openrouter"
base_url = "{base_url}"
env_key = "OPENROUTER_API_KEY"
wire_api = "responses"
"""
    # Only rewrite if different. Multiple SLURM jobs/processes call this concurrently
    # at startup, so the temp file must be unique per-process (PID) -- a shared tmp
    # filename caused a real FileNotFoundError race (two processes both renaming the
    # same "config.toml.tmp" -> config.toml, the second finding it already consumed).
    try:
        needs_write = not config_path.exists() or config_path.read_text(encoding="utf-8") != config_text
    except OSError:
        needs_write = True
    if needs_write:
        tmp_path = codex_home / f".config.toml.tmp.{os.getpid()}"
        try:
            tmp_path.write_text(config_text, encoding="utf-8")
            os.replace(tmp_path, config_path)
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Output validation / idempotency
# ---------------------------------------------------------------------------
def _missing_required_outputs(output_dir: Path) -> list[str]:
    missing = [f for f in REQUIRED_OUTPUT_LABELS if not (output_dir / f).exists()]
    if "submission.csv" not in missing:
        sub_path = output_dir / "submission.csv"
        try:
            if sub_path.stat().st_size < 5:
                missing.append("submission.csv")
        except OSError:
            missing.append("submission.csv")
    return missing


def task_already_done(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    if (output_dir / "FAILED.json").exists():
        return False
    return not _missing_required_outputs(output_dir)


# ---------------------------------------------------------------------------
# Transcript parsing (Codex --json JSONL events)
# ---------------------------------------------------------------------------
def parse_codex_usage(transcript_path: Path) -> dict[str, int]:
    input_tokens = 0
    output_tokens = 0
    if not transcript_path.exists():
        return {"input_tokens": 0, "output_tokens": 0}
    try:
        for line in transcript_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if evt.get("type") == "turn.completed":
                usage = evt.get("usage") or {}
                input_tokens += int(usage.get("input_tokens", 0) or 0)
                output_tokens += int(usage.get("output_tokens", 0) or 0)
    except Exception:
        pass
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def backfill_metrics(output_dir: Path, duration_sec: float, token_usage: dict[str, int]) -> None:
    metrics_path = output_dir / "metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}

    def _f(v: Any) -> float:
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    train_time = max(0.0, _f(metrics.get("train_time_sec")))
    test_time = max(0.0, _f(metrics.get("test_time_sec")))
    code_total_time = round(max(0.0, duration_sec), 2)
    solution_generation_time = round(max(0.0, code_total_time - train_time - test_time), 2)

    metrics["solution_generation_time_sec"] = metrics.get("solution_generation_time_sec", solution_generation_time)
    metrics["train_time_sec"] = round(train_time, 2)
    metrics["test_time_sec"] = round(test_time, 2)
    metrics["code_total_time_sec"] = code_total_time
    metrics["input_tokens"] = token_usage.get("input_tokens", 0)
    metrics["output_tokens"] = token_usage.get("output_tokens", 0)
    metrics.setdefault("model_used", metrics.get("model_used", "N/A"))
    metrics.setdefault("model_param_count", metrics.get("model_param_count", "N/A"))
    metrics.setdefault("notes", metrics.get("notes", ""))

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Single-task runner (subprocess with hard wall-clock kill)
# ---------------------------------------------------------------------------
def run_single_task(
    task_spec: TaskSpec,
    prompt_template: str,
    model: str,
    base_url: str,
    api_key: str,
    codex_home: Path,
    wall_clock_sec: int,
    gpu_id: int | None,
    print_lock: threading.Lock,
) -> TaskRunResult:
    task_spec.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    with print_lock:
        print(f"[START] {task_spec.key} (gpu={gpu_id})")

    prompt = build_prompt(prompt_template, task_spec)
    (task_spec.output_dir / "agent_prompt.txt").write_text(prompt, encoding="utf-8")

    transcript_path = task_spec.output_dir / "codexcli_transcript.jsonl"
    stderr_path = task_spec.output_dir / "codexcli_stderr.log"

    env = os.environ.copy()
    env["CODEX_HOME"] = str(codex_home)
    env["OPENROUTER_API_KEY"] = api_key
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--json",
        "-C",
        str(task_spec.output_dir),
        "-m",
        model,
        prompt,
    ]

    proc = None
    exit_code: int | None = None
    error: str | None = None
    timed_out = False
    try:
        with open(transcript_path, "w", encoding="utf-8") as out_f, open(stderr_path, "w", encoding="utf-8") as err_f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(task_spec.output_dir),
                env=env,
                stdout=out_f,
                stderr=err_f,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                exit_code = proc.wait(timeout=wall_clock_sec)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    time.sleep(5)
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=15)
                exit_code = proc.returncode
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    duration = time.time() - start
    token_usage = parse_codex_usage(transcript_path)
    backfill_metrics(task_spec.output_dir, duration, token_usage)

    if timed_out:
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
        status = "failed"
        error = f"Task wall-clock exceeded {wall_clock_sec}s"
    else:
        missing = _missing_required_outputs(task_spec.output_dir)
        if missing:
            status = "failed"
            error = error or f"Missing/invalid outputs: {missing} (exit_code={exit_code})"
            (task_spec.output_dir / "FAILED.json").write_text(
                json.dumps(
                    {"task": task_spec.key, "error": error, "duration_sec": duration, "exit_code": exit_code},
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            status = "success"

    with print_lock:
        print(
            f"[{status.upper()}] {task_spec.key} ({duration:.0f}s) exit={exit_code} "
            f"in_tok={token_usage['input_tokens']} out_tok={token_usage['output_tokens']}"
        )

    return TaskRunResult(
        task_key=task_spec.key,
        status=status,
        output_dir=str(task_spec.output_dir),
        error=error,
        duration_sec=duration,
        input_tokens=token_usage["input_tokens"],
        output_tokens=token_usage["output_tokens"],
        exit_code=exit_code,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenAI Codex CLI (codex exec) on BioXArena tasks")
    parser.add_argument("--prefix-dir", type=str, default=str(WORKSPACE_ROOT))
    parser.add_argument("--tasks-root", type=str, default=None, help="Public data root containing <domain>/<task>/public/description.md (default: <prefix-dir>/BioXArena-Data-Public).")
    parser.add_argument("--output-root", type=str, default=None, help="Override full output root (default: <prefix-dir>/BioXArena-Output/<model-dir>/<round-name>)")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--round-name", type=str, default=DEFAULT_ROUND_NAME)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base-url", type=str, default=DEFAULT_PROXY_BASE_URL)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--codex-home", type=str, default=None, help="Isolated CODEX_HOME (default: <repo>/.runtime/<model-dir>/codex_home)")
    parser.add_argument("--task-wall-clock-sec", type=int, default=7200)
    parser.add_argument("--task", action="append", default=[], dest="tasks")
    parser.add_argument("--domain", action="append", default=[], dest="domains")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs visible to this job; workers are round-robin assigned CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--force", action="store_true", help="Re-run tasks even if valid outputs already exist.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prefix_dir = Path(args.prefix_dir)
    tasks_root = Path(args.tasks_root) if args.tasks_root else prefix_dir / "BioXArena-Data-Public"
    model_dir = args.model_dir or f"codexcli__{args.model.replace('/', '__')}"
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = prefix_dir / "BioXArena-Output" / model_dir / args.round_name

    codex_home = Path(args.codex_home) if args.codex_home else RUNTIME_ROOT / model_dir / "codex_home"
    api_key = resolve_api_key(args.api_key)
    ensure_codex_home(codex_home, args.base_url, args.model)

    prompt_template = load_prompt_template(DEFAULT_PROMPT_PATH)

    all_tasks = discover_tasks(tasks_root, output_root)
    selected = filter_tasks(all_tasks, args.domains, args.tasks, args.all_tasks)
    if not selected:
        print("No tasks selected. Use --task, --domain, or --all-tasks.")
        raise SystemExit(1)

    if not args.force:
        pending = [t for t in selected if not task_already_done(t.output_dir)]
        skipped = len(selected) - len(pending)
        if skipped:
            print(f"Skipping {skipped} already-completed task(s) (use --force to re-run).")
        selected = pending

    print("Agent: Codex CLI (codex exec)")
    print(f"Backend LLM: {args.model} via {args.base_url}")
    print(f"CODEX_HOME: {codex_home}")
    print(f"Tasks: {len(selected)}")
    print(f"Output root: {output_root}")
    print(f"Max workers: {args.max_workers}  Num GPUs: {args.num_gpus}")
    print(f"Per-task wall clock: {args.task_wall_clock_sec}s")
    print()

    if args.dry_run:
        for t in selected:
            print(f"  [DRY] {t.key} -> {t.output_dir}")
        return

    if not selected:
        print("Nothing to do (all selected tasks already complete).")
        return

    print_lock = threading.Lock()
    results: list[TaskRunResult] = []

    def _run(idx: int, task_spec: TaskSpec) -> TaskRunResult:
        gpu_id = idx % max(1, args.num_gpus) if args.num_gpus > 0 else None
        return run_single_task(
            task_spec=task_spec,
            prompt_template=prompt_template,
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
            codex_home=codex_home,
            wall_clock_sec=args.task_wall_clock_sec,
            gpu_id=gpu_id,
            print_lock=print_lock,
        )

    if args.max_workers <= 1:
        for i, task_spec in enumerate(selected):
            results.append(_run(i, task_spec))
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(_run, i, task_spec): task_spec for i, task_spec in enumerate(selected)}
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

    summary_path = output_root / f"codexcli_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "agent": "codexcli",
        "model": args.model,
        "base_url": args.base_url,
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
