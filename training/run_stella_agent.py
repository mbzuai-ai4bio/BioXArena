#!/usr/bin/env python3
"""Run STELLA agent on BioXArena tasks.

STELLA is a self-evolving agent with multiple sub-agents:
  - Manager Agent: task decomposition and orchestration
  - Dev Agent: code execution
  - Tool Creation Agent: dynamic tool building
  - Critic Agent: quality evaluation

Unlike general LLM agents, STELLA handles its own ReAct loop
and self-evolving skill management internally.

All STELLA sub-agents (manager, dev, critic, tool creation) use the --model backbone,
served through OpenRouter (OpenAI-compatible API).

Paper settings (enabled by training/run_agent.py):
  STELLA_WAIT_BACKGROUND=1   wait (within the time limit) for background training
                             processes the agent left running before judging outputs
  STELLA_SAVE_TO_TASK_DIR=1  tell the dev agent to save outputs in the task output
                             directory instead of './agent_outputs'

Examples:
    python run_stella_agent.py --model anthropic/claude-sonnet-5 --task chemical-biology/tox21-sr-are --time-limit 7200
    python run_stella_agent.py --model deepseek/deepseek-v3.2 --domain chemical-biology --max-workers 2
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

from openrouter_config import OPENROUTER_BASE_URL, resolve_api_key

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
TRAINING_DIR = SCRIPT_PATH.parent
EVAL_ROOT = TRAINING_DIR.parent          # repository root
WORKSPACE_ROOT = EVAL_ROOT.parent
DEFAULT_TASKS_ROOT = EVAL_ROOT / "tasks"
DEFAULT_PROMPT_PATH = EVAL_ROOT / "prompts" / "unified_eval_prompt.py"
DEFAULT_ROUND_NAME = "round1"
STELLA_AGENT_DIR = EVAL_ROOT / "agents" / "STELLA"

REQUIRED_OUTPUT_LABELS = ["submission.csv", "metrics.json", "solution.py"]


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
# Task discovery
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


def build_stella_prompt(prompt_template: str, task_spec: TaskSpec) -> str:
    description = task_spec.description_path.read_text(encoding="utf-8")
    return prompt_template.format(
        task_dir=str(task_spec.task_dir),
        output_dir=str(task_spec.output_dir),
        description=description,
    )


# ---------------------------------------------------------------------------
# STELLA model patching
# ---------------------------------------------------------------------------
def patch_stella_models(model_id: str, api_key: str, base_url: str, temperature: float,
                        manager_model_id: str | None = None):
    """Monkey-patch STELLA's global model variables before agent initialization.

    This avoids modifying stella_core.py directly.

    Args:
        model_id: Model for Dev Agent + Tool Creation Agent (claude_model).
        manager_model_id: Model for Manager + Critic (gpt_model + gemini_model).
                         If None, uses model_id for all roles (single-model mode).
    """
    from smolagents import OpenAIServerModel

    mgr_id = manager_model_id or model_id

    model_kwargs = {"api_base": base_url, "api_key": api_key}
    # Sonnet 5 rejects the temperature field, including temperature=0.
    if "sonnet-5" not in model_id.lower():
        model_kwargs["temperature"] = temperature
    # Optional context-limit overrides, off unless set in the environment. Used only for
    # the 2026-10-07 rerun of six STELLA/DeepSeek-V3.2 tasks that failed on context length.
    if os.environ.get("STELLA_MAX_TOKENS"):
        model_kwargs["max_tokens"] = int(os.environ["STELLA_MAX_TOKENS"])
    if os.environ.get("STELLA_OPENROUTER_TRANSFORMS"):
        model_kwargs["extra_body"] = {"transforms": os.environ["STELLA_OPENROUTER_TRANSFORMS"].split(",")}

    # Dev Agent + Tool Creation Agent model
    dev_model = OpenAIServerModel(model_id=model_id, **model_kwargs)
    # Manager Agent model
    manager_model = OpenAIServerModel(model_id=mgr_id, **model_kwargs)
    # Critic Agent model
    critic_model = OpenAIServerModel(model_id=mgr_id, **model_kwargs)

    import stella_core
    stella_core.claude_model = dev_model      # -> dev_agent
    stella_core.gpt_model = manager_model     # -> manager_agent + tool_creation_agent (in stella_core)
    stella_core.gemini_model = critic_model   # -> critic_agent

    if manager_model_id:
        # tool_creation_agent uses gpt_model in stella_core, but per original paper
        # it should use the same model as dev_agent. Patch it after initialization
        # by storing the dev_model for later use.
        stella_core._dev_model_override = dev_model
        print(f"  Patched STELLA models -> Dev/Tool: {model_id}, Manager/Critic: {mgr_id} (temp={temperature})")
    else:
        stella_core._dev_model_override = None
        print(f"  Patched STELLA models -> {model_id} (temp={temperature})")


# ---------------------------------------------------------------------------
# STELLA agent runner
# ---------------------------------------------------------------------------
def run_single_task(
    task_spec: TaskSpec,
    prompt_template: str,
    model_id: str,
    api_key: str,
    base_url: str,
    temperature: float,
    enable_tool_creation: bool,
    time_limit: int,
    print_lock: threading.Lock,
    manager_model_id: str | None = None,
) -> TaskRunResult:
    """Run STELLA agent on a single task."""
    task_start = time.time()
    task_spec.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from _core_pkg_guard import restore_core_packages
        restore_core_packages()
    except Exception:
        pass

    with print_lock:
        print(f"[START] {task_spec.key}")

    try:
        # Set cwd to the task output directory so any files the agent writes
        # (models, data, etc.) land there instead of polluting training/.
        os.chdir(str(task_spec.output_dir))

        # Limit per-worker memory to ~60GB (256GB total / 4 workers, with headroom)
        import resource
        mem_limit = 60 * 1024 * 1024 * 1024  # 60GB in bytes
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))

        # Assign each worker a dedicated GPU based on task hash
        # Let the system handle GPU assignment naturally.

        # Set OPENROUTER_API_KEY for STELLA's internal use
        os.environ["OPENROUTER_API_KEY"] = api_key

        # Add STELLA to sys.path
        stella_path = str(STELLA_AGENT_DIR)
        if stella_path not in sys.path:
            sys.path.insert(0, stella_path)

        # Patch models before initialization
        patch_stella_models(model_id, api_key, base_url, temperature,
                            manager_model_id=manager_model_id)

        # Initialize STELLA
        from stella_core import initialize_stella
        manager_agent = initialize_stella(
            use_template=True,
            enable_tool_creation=enable_tool_creation,
            use_mem0=False,  # Disable external memory service
        )

        # The sub-agents (dev / critic / tool_creation) are built at stella_core import time and keep the
        # hard-coded default model object (google/gemini-3-flash-preview via OpenRouter); rebinding the
        # module-level model variables in patch_stella_models does not reach them. Re-point them here so every
        # agent uses the requested backbone (fixed 2026-10-08; earlier STELLA runs used Gemini-3-Flash sub-agents).
        # In mixed-model mode tool_creation_agent uses the dev model, matching the original paper design.
        import stella_core as _sc
        _sc.dev_agent.model = _sc.claude_model
        _sc.critic_agent.model = _sc.gemini_model
        if getattr(_sc, 'tool_creation_agent', None) is not None:
            _sc.tool_creation_agent.model = getattr(_sc, '_dev_model_override', None) or _sc.gpt_model
        _bound = {name: getattr(getattr(_sc, name, None), 'model', None) for name in ('dev_agent', 'critic_agent', 'tool_creation_agent')}
        _bound['manager'] = manager_agent.model
        _bound.update({f'managed:{n}': a.model for n, a in getattr(manager_agent, 'managed_agents', {}).items()})
        _expected = {model_id, manager_model_id or model_id}
        _wrong = {n: getattr(m, 'model_id', m) for n, m in _bound.items() if m is not None and getattr(m, 'model_id', None) not in _expected}
        if _wrong:
            raise RuntimeError(f"STELLA agents not bound to the requested backbone: {_wrong}")
        print("  STELLA agent models: " + ", ".join(f"{n}={getattr(m, 'model_id', m)}" for n, m in _bound.items()), flush=True)

        # Build prompt
        prompt = build_stella_prompt(prompt_template, task_spec)

        # Save prompt for debugging
        (task_spec.output_dir / "stella_prompt.txt").write_text(prompt, encoding="utf-8")

        # Run agent with time limit
        from concurrent.futures import ThreadPoolExecutor as _TPE, TimeoutError as _TE
        with _TPE(max_workers=1) as _pool:
            future = _pool.submit(manager_agent.run, prompt)
            try:
                result = future.result(timeout=time_limit)
            except _TE:
                result = "TIMEOUT: agent exceeded time limit"

        # Save agent output
        (task_spec.output_dir / "stella_output.txt").write_text(
            str(result), encoding="utf-8"
        )

        # Check required outputs
        missing = [f for f in REQUIRED_OUTPUT_LABELS if not (task_spec.output_dir / f).exists()]
        duration = time.time() - task_start

        if missing:
            status = "failed"
            error = f"Missing outputs: {missing}"
            # Record the failure at decision time (2026-10-08) so that files a leftover background
            # process writes later are not scored by the evaluator.
            # 2026-10-12: with STELLA_WAIT_BACKGROUND=1 the parent first waits for the task's background
            # processes (within the 7,200 s budget) and writes FAILED.json itself after re-checking outputs.
            if os.environ.get("STELLA_WAIT_BACKGROUND") != "1":
              (task_spec.output_dir / "FAILED.json").write_text(
                json.dumps({"task": task_spec.key, "error": error, "duration_sec": duration,
                            "note": "decided when the agent returned; later files do not count"}, indent=2),
                encoding="utf-8",
            )
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
    model_id,
    api_key,
    base_url,
    temperature,
    enable_tool_creation,
    time_limit,
    result_queue,
    manager_model_id=None,
):
    try:
        # 2026-10-13: tag every process the agent starts (inherited env) so background jobs launched from
        # /tmp or other directories are still recognised as belonging to this task.
        os.environ["BX_STELLA_TASK_DIR"] = str(Path(task_spec.output_dir).resolve())
        local_lock = threading.Lock()
        result = run_single_task(
            task_spec=task_spec,
            prompt_template=prompt_template,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            enable_tool_creation=enable_tool_creation,
            time_limit=time_limit,
            print_lock=local_lock,
            manager_model_id=manager_model_id,
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


def _task_processes(output_dir: Path, started_after: float) -> list:
    """This user's processes started after the task began whose cwd or command line points into the task dir."""
    import psutil

    me, out, found = os.getuid(), str(Path(output_dir).resolve()), []
    for p in psutil.process_iter(["pid", "uids", "create_time", "cwd", "cmdline"]):
        try:
            if p.pid == os.getpid() or p.info["uids"] is None or p.info["uids"].real != me:
                continue
            if (p.info["create_time"] or 0) < started_after - 1:
                continue
            if (p.info["cwd"] or "").startswith(out) or out in " ".join(p.info["cmdline"] or []):
                found.append(p)
            elif p.environ().get("BX_STELLA_TASK_DIR") == out:  # 2026-10-13: inherited task tag
                found.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def _wait_for_background(task_spec, start: float, time_limit: float, print_lock) -> dict:
    """2026-10-12 (STELLA_WAIT_BACKGROUND=1): the agent returned without all outputs but left background
    processes (e.g. ``python solution.py &``) running. Wait for them until the task's 7,200 s budget ends."""
    import psutil

    procs = _task_processes(task_spec.output_dir, start)
    info = {"waited": False, "processes": [f"{p.pid}:{' '.join(p.cmdline())[:120]}" for p in procs if p.is_running()]}
    if not procs:
        return info
    info["waited"] = True
    with print_lock:
        print(f"[WAIT] {task_spec.key}: agent returned with {len(procs)} background process(es); waiting until wall-clock {time_limit}s")
    while time.time() - start < time_limit:
        alive = [p for p in _task_processes(task_spec.output_dir, start) if p.status() != psutil.STATUS_ZOMBIE]
        if not alive:
            break
        time.sleep(15)
    info["wait_end_sec"] = time.time() - start
    info["budget_exhausted"] = time.time() - start >= time_limit
    return info


def _kill_task_leftovers(output_dir: Path, started_after: float) -> list[str]:
    """Kill processes the agent left running for this task (2026-10-08).

    Background jobs launched by the agent (e.g. ``nohup python train.py &``) survive the task's child process,
    keep using the GPU during the next task and may write outputs after the task was judged. Targets: this user's
    processes started after the task began whose cwd or command line points into the task's output directory.
    """
    import psutil

    me, out, killed = os.getuid(), str(Path(output_dir).resolve()), []
    victims = []
    for p in psutil.process_iter(["pid", "uids", "create_time", "cwd", "cmdline"]):
        try:
            if p.pid == os.getpid() or p.info["uids"] is None or p.info["uids"].real != me:
                continue
            if (p.info["create_time"] or 0) < started_after - 1:
                continue
            cwd = p.info["cwd"] or ""
            cmd = " ".join(p.info["cmdline"] or [])
            if cwd.startswith(out) or out in cmd or p.environ().get("BX_STELLA_TASK_DIR") == out:  # env tag: 2026-10-13
                victims.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    for p in victims:
        try:
            killed.append(f"{p.pid}:{' '.join(p.cmdline())[:120]}")
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    psutil.wait_procs(victims, timeout=10)
    return killed


def run_task_with_wall_clock(
    task_spec,
    prompt_template,
    model_id,
    api_key,
    base_url,
    temperature,
    enable_tool_creation,
    time_limit,
    print_lock,
    manager_model_id=None,
):
    """Run a single STELLA task in a child process with a hard wall-clock limit."""
    # Let the system handle GPU assignment naturally.

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(
        target=_child_run_task,
        args=(
            task_spec,
            prompt_template,
            model_id,
            api_key,
            base_url,
            temperature,
            enable_tool_creation,
            time_limit,
            q,
            manager_model_id,
        ),
    )
    start = time.time()
    proc.start()
    # Give the child a small grace period beyond time_limit so its own internal
    # timeout (future.result(timeout=time_limit)) has a chance to fire cleanly.
    proc.join(time_limit + 60)
    timed_out = proc.is_alive()
    if timed_out:
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
    wait_info = None
    if not timed_out and os.environ.get("STELLA_WAIT_BACKGROUND") == "1":
        wait_info = _wait_for_background(task_spec, start, time_limit, print_lock)
        if wait_info.get("budget_exhausted"):
            timed_out = True
    leftovers = _kill_task_leftovers(task_spec.output_dir, start)
    if leftovers:
        with print_lock:
            print(f"[CLEANUP] {task_spec.key}: killed {len(leftovers)} leftover process(es): {leftovers[:5]}")
        try:
            (task_spec.output_dir / "killed_leftover_processes.json").write_text(json.dumps(leftovers, indent=2), encoding="utf-8")
        except Exception:
            pass
    if timed_out:
        duration = time.time() - start
        with print_lock:
            print(f"[TIMEOUT] {task_spec.key} ({duration:.0f}s) exceeded wall-clock {time_limit}s")
        try:
            task_spec.output_dir.mkdir(parents=True, exist_ok=True)
            (task_spec.output_dir / "FAILED.json").write_text(
                json.dumps(
                    {
                        "task": task_spec.key,
                        "error": f"Task wall-clock exceeded {time_limit}s",
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
            error=f"Task wall-clock exceeded {time_limit}s",
            duration_sec=duration,
        )

    try:
        result = q.get(timeout=10)
    except Exception as exc:
        result = None
        child_error = exc
    if os.environ.get("STELLA_WAIT_BACKGROUND") == "1" and result is not None:
        # Final judgement after the background wait (outputs written within the 7,200 s budget count).
        missing = [f for f in REQUIRED_OUTPUT_LABELS if not (task_spec.output_dir / f).exists()]
        duration = time.time() - start
        if wait_info and wait_info.get("waited"):
            (task_spec.output_dir / "background_wait.json").write_text(json.dumps(wait_info, indent=2), encoding="utf-8")
        failed_marker = task_spec.output_dir / "FAILED.json"
        if missing and not failed_marker.exists():
            failed_marker.write_text(json.dumps({"task": task_spec.key, "error": f"Missing outputs: {missing}",
                                                 "duration_sec": duration,
                                                 "note": "decided after waiting for background processes within the time limit"},
                                                indent=2), encoding="utf-8")
        if not missing and not failed_marker.exists():
            result = TaskRunResult(task_key=task_spec.key, status="success", output_dir=str(task_spec.output_dir),
                                   error=None, duration_sec=duration)
        elif missing:
            result = TaskRunResult(task_key=task_spec.key, status="failed", output_dir=str(task_spec.output_dir),
                                   error=f"Missing outputs: {missing}", duration_sec=duration)
    if result is not None:
        return result
    try:
        raise child_error
    except Exception as exc:
        return TaskRunResult(
            task_key=task_spec.key,
            status="failed",
            output_dir=str(task_spec.output_dir),
            error=f"Child process exited (code={proc.exitcode}) without result: {exc}",
            duration_sec=time.time() - start,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run STELLA agent on BioXArena tasks")
    parser.add_argument("--prefix-dir", type=str, default=str(WORKSPACE_ROOT))
    parser.add_argument("--tasks-root", type=str, default=None,
                        help="Task data root (default: <prefix>/BioXArena-Data-Public)")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Exact output directory for this model/round")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Stable model directory label (used when deriving output-root)")
    parser.add_argument("--model", type=str, default="anthropic/claude-sonnet-5",
                        help="LLM model for Dev + Tool Creation agents (via OpenRouter)")
    parser.add_argument("--manager-model", type=str, default=None,
                        help="LLM model for Manager + Critic agents. If not set, uses --model for all.")
    parser.add_argument("--base-url", type=str, default=None,
                        help="API base URL (default: OpenRouter)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenRouter API key (default: OPENROUTER_API_KEY or <repo>/.env)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for all agents (0=deterministic)")
    parser.add_argument("--enable-tool-creation", action="store_true", default=True,
                        help="Enable STELLA's tool creation agent")
    parser.add_argument("--no-tool-creation", action="store_true",
                        help="Disable tool creation agent")
    parser.add_argument("--time-limit", type=int, default=3600,
                        help="Time limit per task in seconds (default: 3600)")
    parser.add_argument("--round-name", type=str, default=DEFAULT_ROUND_NAME)
    parser.add_argument("--task", action="append", default=[], dest="tasks")
    parser.add_argument("--domain", action="append", default=[], dest="domains")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.no_tool_creation:
        args.enable_tool_creation = False

    prefix_dir = Path(args.prefix_dir)
    tasks_root = Path(args.tasks_root) if args.tasks_root else prefix_dir / "BioXArena-Data-Public"
    model_dir_name = args.model_dir or f"stella__{args.model.replace('/', '__').replace(':', '_')}"
    output_root = (Path(args.output_root) if args.output_root else
                   prefix_dir / "BioXArena-Output" / model_dir_name / args.round_name)

    if args.base_url is None:
        args.base_url = OPENROUTER_BASE_URL
    if not args.dry_run:
        args.api_key = resolve_api_key(args.api_key)

    # Load prompt template
    prompt_template = load_prompt_template(DEFAULT_PROMPT_PATH)

    # Discover and filter tasks
    all_tasks = discover_tasks(tasks_root, output_root)
    selected = filter_tasks(all_tasks, args.domains, args.tasks, args.all_tasks)

    if not selected:
        print("No tasks selected. Use --task, --domain, or --all-tasks.")
        sys.exit(1)

    print(f"Agent: STELLA (self-evolving)")
    print(f"Backend LLM: {args.model}")
    print(f"Base URL: {args.base_url}")
    print(f"Temperature: {args.temperature}")
    print(f"Tool creation: {args.enable_tool_creation}")
    print(f"Time limit: {args.time_limit}s")
    print(f"Tasks: {len(selected)}")
    print(f"Output root: {output_root}")
    print(f"Max workers: {args.max_workers}")
    print()

    if args.dry_run:
        for t in selected:
            print(f"  [DRY] {t.key} -> {t.output_dir}")
        return

    # Run tasks
    print_lock = threading.Lock()
    results: list[TaskRunResult] = []

    mgr_model = getattr(args, 'manager_model', None)
    if args.max_workers <= 1:
        for task_spec in selected:
            result = run_task_with_wall_clock(
                task_spec=task_spec,
                prompt_template=prompt_template,
                model_id=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                temperature=args.temperature,
                enable_tool_creation=args.enable_tool_creation,
                time_limit=args.time_limit,
                print_lock=print_lock,
                manager_model_id=mgr_model,
            )
            results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(
                    run_task_with_wall_clock,
                    task_spec=task_spec,
                    prompt_template=prompt_template,
                    model_id=args.model,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    temperature=args.temperature,
                    enable_tool_creation=args.enable_tool_creation,
                    time_limit=args.time_limit,
                    print_lock=print_lock,
                    manager_model_id=mgr_model,
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
    summary_path = output_root / f"stella_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "agent": "stella",
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "enable_tool_creation": args.enable_tool_creation,
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
