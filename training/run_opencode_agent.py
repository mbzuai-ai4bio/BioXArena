#!/usr/bin/env python3
"""Run OpenCode CLI (opencode run) as a full coding-agent baseline on BioXArena tasks.

Same contract as run_codexcli_agent.py (see that file's docstring for the full
rationale): OpenCode is a provider-agnostic coding-agent CLI that owns its own
tool-use loop. This runner only stages the task, invokes `opencode run`
non-interactively against OpenRouter, enforces a hard wall-clock kill (default 2h),
and collects submission.csv / metrics.json / solution.py plus the full JSONL
transcript and token usage parsed from OpenCode's own step_finish events.

OpenCode is pointed at OpenRouter via an isolated OPENCODE_CONFIG file defining a
custom provider plus isolated XDG_DATA_HOME/XDG_CONFIG_HOME directories -- this never
touches any pre-existing personal OpenCode configuration on this machine. Claude
models (`anthropic/...`) use the "@ai-sdk/anthropic" provider (Anthropic Messages
format, OpenRouter's /api/v1/messages); all other models use
"@ai-sdk/openai-compatible" (chat completions), as in the paper runs.

Examples:
    python run_opencode_agent.py --model openai/gpt-5.6-sol --task chemical-biology/bace1-binding-affinity
    python run_opencode_agent.py --model anthropic/claude-opus-4.6 --all-tasks --max-workers 4 --num-gpus 4
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
# Paths (mirror the Biomni/STELLA/Codex-CLI runner layout)
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
DEFAULT_PROVIDER_ID = "openrouter"

REQUIRED_OUTPUT_LABELS = ["submission.csv", "metrics.json", "solution.py"]

OPENCODE_APPENDIX = """

# Coding-agent CLI notes
You are running as an autonomous coding-agent CLI (OpenCode) with direct shell and
file access. Work entirely within the directories given to you above ({task_dir}
for reading public data, {output_dir} for writing your solution and outputs). Do
not attempt to list, guess, or read any path outside of these two directories --
in particular, never search for or open files named like "answers", "private", or
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
    cost_usd: float = 0.0
    exit_code: int | None = None


# ---------------------------------------------------------------------------
# Task discovery (identical contract to the other BioXArena agent runners)
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
# Prompt builder
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
    appendix = OPENCODE_APPENDIX.format(task_dir=str(task_spec.task_dir), output_dir=str(task_spec.output_dir))
    return base + appendix


# ---------------------------------------------------------------------------
# Credentials / isolated OpenCode config
# ---------------------------------------------------------------------------
def provider_settings(model: str) -> tuple[str, dict[str, int]]:
    """AI-SDK provider package and context/output limits used in the paper runs."""
    if model.startswith("anthropic/"):
        return "@ai-sdk/anthropic", {"context": 200000, "output": 32000}
    return "@ai-sdk/openai-compatible", {"context": 400000, "output": 65536}


def ensure_opencode_config(config_path: Path, base_url: str, model: str, provider_id: str) -> None:
    """Write an isolated opencode.json pointing at OpenRouter. Idempotent."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    provider_npm, limit = provider_settings(model)
    payload = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            provider_id: {
                "npm": provider_npm,
                "name": "OpenRouter",
                "options": {
                    "baseURL": base_url,
                    "apiKey": "{env:OPENROUTER_API_KEY}",
                },
                "models": {
                    model: {
                        "name": model,
                        "limit": limit,
                    }
                },
            }
        },
        "model": f"{provider_id}/{model}",
        "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"},
    }
    text = json.dumps(payload, indent=2)
    # Multiple SLURM jobs/processes call this concurrently at startup; use a
    # per-process (PID) temp filename to avoid the same shared-tmp-file race
    # found and fixed in run_codexcli_agent.py's ensure_codex_home().
    try:
        needs_write = not config_path.exists() or config_path.read_text(encoding="utf-8") != text
    except OSError:
        needs_write = True
    if needs_write:
        tmp_path = config_path.parent / f".opencode.json.tmp.{os.getpid()}"
        try:
            tmp_path.write_text(text, encoding="utf-8")
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
# Transcript parsing (opencode --format json JSONL events)
# ---------------------------------------------------------------------------
def parse_opencode_usage(transcript_path: Path) -> dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    cost = 0.0
    if not transcript_path.exists():
        return {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    try:
        for line in transcript_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            part = evt.get("part") or {}
            if evt.get("type") == "step_finish" or part.get("type") == "step-finish":
                tokens = part.get("tokens") or {}
                input_tokens += int(tokens.get("input", 0) or 0)
                output_tokens += int(tokens.get("output", 0) or 0)
                cost += float(part.get("cost", 0.0) or 0.0)
    except Exception:
        pass
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": round(cost, 6)}


def backfill_metrics(output_dir: Path, duration_sec: float, token_usage: dict[str, Any]) -> None:
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
    metrics["cost_usd"] = token_usage.get("cost_usd", 0.0)
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
    provider_id: str,
    api_key: str,
    opencode_config: Path,
    xdg_data_home: Path,
    xdg_config_home: Path,
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

    transcript_path = task_spec.output_dir / "opencode_transcript.jsonl"
    stderr_path = task_spec.output_dir / "opencode_stderr.log"

    # OpenCode keeps a local SQLite session/state DB under XDG_DATA_HOME. Sharing one
    # XDG_DATA_HOME across concurrent workers causes "database is locked" failures
    # (observed during smoke testing), so every task gets its own isolated XDG dirs.
    task_xdg_data = xdg_data_home / task_spec.domain / task_spec.task
    task_xdg_config = xdg_config_home / task_spec.domain / task_spec.task
    task_xdg_data.mkdir(parents=True, exist_ok=True)
    task_xdg_config.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OPENCODE_CONFIG"] = str(opencode_config)
    env["OPENROUTER_API_KEY"] = api_key
    env["XDG_DATA_HOME"] = str(task_xdg_data)
    env["XDG_CONFIG_HOME"] = str(task_xdg_config)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "opencode",
        "run",
        "--dir",
        str(task_spec.output_dir),
        "-m",
        f"{provider_id}/{model}",
        "--auto",
        "--format",
        "json",
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
    token_usage = parse_opencode_usage(transcript_path)
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
        cost_usd=token_usage.get("cost_usd", 0.0),
        exit_code=exit_code,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenCode CLI (opencode run) on BioXArena tasks")
    parser.add_argument("--prefix-dir", type=str, default=str(WORKSPACE_ROOT))
    parser.add_argument("--tasks-root", type=str, default=None, help="Public data root containing <domain>/<task>/public/description.md (default: <prefix-dir>/BioXArena-Data-Public).")
    parser.add_argument("--output-root", type=str, default=None, help="Override full output root (default: <prefix-dir>/BioXArena-Output/<model-dir>/<round-name>)")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--round-name", type=str, default=DEFAULT_ROUND_NAME)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--provider-id", type=str, default=DEFAULT_PROVIDER_ID)
    parser.add_argument("--base-url", type=str, default=DEFAULT_PROXY_BASE_URL)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--opencode-config", type=str, default=None, help="Isolated opencode.json (default: <repo>/.runtime/<model-dir>/opencode/opencode.json)")
    parser.add_argument("--xdg-data-home", type=str, default=None, help="Default: <repo>/.runtime/<model-dir>/opencode/data")
    parser.add_argument("--xdg-config-home", type=str, default=None, help="Default: <repo>/.runtime/<model-dir>/opencode/xdg_config")
    parser.add_argument("--task-wall-clock-sec", type=int, default=7200)
    parser.add_argument("--task", action="append", default=[], dest="tasks")
    parser.add_argument("--domain", action="append", default=[], dest="domains")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prefix_dir = Path(args.prefix_dir)
    tasks_root = Path(args.tasks_root) if args.tasks_root else prefix_dir / "BioXArena-Data-Public"
    model_dir = args.model_dir or f"opencode__{args.model.replace('/', '__')}"
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = prefix_dir / "BioXArena-Output" / model_dir / args.round_name

    runtime_dir = RUNTIME_ROOT / model_dir / "opencode"
    opencode_config = Path(args.opencode_config) if args.opencode_config else runtime_dir / "opencode.json"
    xdg_data_home = Path(args.xdg_data_home) if args.xdg_data_home else runtime_dir / "data"
    xdg_config_home = Path(args.xdg_config_home) if args.xdg_config_home else runtime_dir / "xdg_config"
    api_key = resolve_api_key(args.api_key)
    ensure_opencode_config(opencode_config, args.base_url, args.model, args.provider_id)
    xdg_data_home.mkdir(parents=True, exist_ok=True)
    xdg_config_home.mkdir(parents=True, exist_ok=True)

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

    print("Agent: OpenCode (opencode run)")
    print(f"Backend LLM: {args.provider_id}/{args.model} via {args.base_url}")
    print(f"OPENCODE_CONFIG: {opencode_config}")
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
            provider_id=args.provider_id,
            api_key=api_key,
            opencode_config=opencode_config,
            xdg_data_home=xdg_data_home,
            xdg_config_home=xdg_config_home,
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

    summary_path = output_root / f"opencode_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "agent": "opencode",
        "model": f"{args.provider_id}/{args.model}",
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
