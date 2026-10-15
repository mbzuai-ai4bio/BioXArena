#!/usr/bin/env python3
"""Unified entry point for running any BioXArena agent through OpenRouter.

Inputs: agent name, backbone model (an OpenRouter model id) and an OpenRouter API key.
The script picks the matching runner in this directory, fills in the paper settings
for that agent (2 h per task, 1 GPU, ...), and forwards the API key to the runner
through the OPENROUTER_API_KEY environment variable (it never appears on a command line).

Examples:
    # one task, any agent/backbone
    python run_agent.py --agent claudecode --model anthropic/claude-opus-4.6 \\
        --api-key "$OPENROUTER_API_KEY" --data-root /path/to/BioXArena-Data-Public \\
        --task chemical-biology/bace1-binding-affinity

    # reproduce one of the 18 paper configurations, part 1 of 4 (19 tasks)
    python run_agent.py --config codex_g6a --data-root /path/to/BioXArena-Data-Public --part 1

    # list agents and paper configurations
    python run_agent.py --list-configs

    # check that the backbone answers through OpenRouter (one short request, no task run)
    python run_agent.py --agent minimal --model z-ai/glm-5.1 --check
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from openrouter_config import OPENROUTER_BASE_URL, REPO_ROOT, resolve_api_key

TRAINING_DIR = Path(__file__).resolve().parent
NUM_PARTS = 4

# agent name -> (runner script, display name)
AGENTS = {
    "minimal": ("run_general_llm_agents.py", "Minimal"),
    "opencode": ("run_opencode_agent.py", "OpenCode"),
    "codex": ("run_codexcli_agent.py", "Codex CLI"),
    "claudecode": ("run_claudecode_agent.py", "Claude Code"),
    "biomni": ("run_biomni_agent.py", "Biomni"),
    "stella": ("run_stella_agent.py", "STELLA"),
    "mlevolve": ("run_mlevolve_agent.py", "MLEvolve"),
    "mlmaster2": ("run_mlmaster2.0_agent.py", "MLMaster2.0"),
}

# The 18 configurations of the paper's main leaderboard: id -> (agent, OpenRouter model, extra runner args).
PAPER_CONFIGS = {
    "minimal_g56s": ("minimal", "openai/gpt-5.6-sol", []),
    "minimal_o46": ("minimal", "anthropic/claude-opus-4.6", []),
    "minimal_ge": ("minimal", "google/gemini-3.1-pro-preview", []),
    "minimal_glm": ("minimal", "z-ai/glm-5.1", []),
    "minimal_qw": ("minimal", "qwen/qwen3.6-plus", []),
    "minimal_gm": ("minimal", "google/gemma-4-31b-it", []),
    "minimal_dv3": ("minimal", "deepseek/deepseek-v3.2", []),
    "opencode_g56s": ("opencode", "openai/gpt-5.6-sol", []),
    "opencode_o46": ("opencode", "anthropic/claude-opus-4.6", []),
    "codex_g56s": ("codex", "openai/gpt-5.6-sol", []),
    "codex_g6a": ("codex", "openai/gpt-6-astra", []),
    "claudecode_o46": ("claudecode", "anthropic/claude-opus-4.6", []),
    "claudecode_o5": ("claudecode", "anthropic/claude-opus-5", ["--max-budget-usd", "30"]),
    "claudecode_o55": ("claudecode", "anthropic/claude-opus-5.5", ["--max-budget-usd", "30"]),
    "biomni_s5": ("biomni", "anthropic/claude-sonnet-5", []),
    "stella_s5": ("stella", "anthropic/claude-sonnet-5", []),
    "mlevolve_ge": ("mlevolve", "google/gemini-3.1-pro-preview", []),
    "mlmaster2_dv4": ("mlmaster2", "deepseek/deepseek-v4-pro", []),
}


def discover_task_keys(data_root: Path) -> list[str]:
    return sorted(
        f"{d.parent.parent.name}/{d.parent.name}"
        for d in data_root.glob("*/*/public")
        if (d / "description.md").exists()
    )


def select_tasks(args: argparse.Namespace, data_root: Path) -> list[str]:
    if args.task:
        return list(args.task)
    keys = discover_task_keys(data_root)
    if not keys:
        raise SystemExit(f"No tasks found under {data_root} (expected <domain>/<task>/public/description.md).")
    if args.domain:
        keys = [k for k in keys if k.split("/")[0] in args.domain]
    if args.part:
        if not 1 <= args.part <= NUM_PARTS:
            raise SystemExit(f"--part must be between 1 and {NUM_PARTS}")
        size = -(-len(keys) // NUM_PARTS)
        keys = keys[(args.part - 1) * size: args.part * size]
    elif not (args.all_tasks or args.domain):
        raise SystemExit("Select tasks with --task, --domain, --part or --all-tasks.")
    return keys


def runner_args(agent: str, model: str, args: argparse.Namespace, data_root: Path, output_root: Path) -> list[str]:
    model_dir = f"{agent}__{model.replace('/', '__').replace(':', '_')}"
    run_dir = output_root / model_dir / args.round_name
    wall = str(args.time_limit)
    if agent == "minimal":
        return [
            "--prefix-dir", str(output_root.parent), "--tasks-root", str(REPO_ROOT / "tasks"),
            "--data-root", str(data_root), "--output-root", str(output_root),
            "--model", model, "--model-dir", model_dir, "--round-name", args.round_name,
            "--max-workers", str(args.max_workers), "--max-attempts", "3",
            "--request-timeout-sec", "180", "--run-timeout-sec", wall, "--task-timeout-sec", wall,
            "--temperature", "0", "--seed", "2026",
        ]
    common = ["--tasks-root", str(data_root), "--output-root", str(run_dir), "--model-dir", model_dir,
              "--round-name", args.round_name, "--model", model, "--max-workers", str(args.max_workers)]
    if agent in ("opencode", "codex", "claudecode"):
        return common + ["--task-wall-clock-sec", wall, "--num-gpus", str(args.num_gpus)]
    if agent == "biomni":
        extra = ["--biomni-data-dir", args.biomni_data_dir] if args.biomni_data_dir else []
        return common + ["--timeout-seconds", "600", "--task-wall-clock-sec", wall] + extra
    if agent == "stella":
        return common + ["--time-limit", wall]
    if agent in ("mlevolve", "mlmaster2"):
        return common + ["--temperature", "0", "--steps", "999", "--time-limit", wall]
    raise ValueError(agent)


def check_backbone(agent: str, model: str, api_key: str) -> int:
    """One short request to the backbone through OpenRouter (chat completions)."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 512,
    }).encode()
    req = urllib.request.Request(
        f"{OPENROUTER_BASE_URL}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        print(f"[FAIL] {agent} + {model}: HTTP {exc.code} {exc.read()[:300]!r}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {agent} + {model}: {type(exc).__name__}: {exc}")
        return 1
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    print(f"[OK] {agent} + {model}: reply={text.strip()[:40]!r} usage={data.get('usage', {}).get('total_tokens')}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", choices=sorted(AGENTS), help="Agent scaffold.")
    p.add_argument("--model", help="Backbone LLM as an OpenRouter model id, e.g. anthropic/claude-opus-4.6.")
    p.add_argument("--api-key", default=None, help="OpenRouter API key (default: OPENROUTER_API_KEY or <repo>/.env).")
    p.add_argument("--config", choices=sorted(PAPER_CONFIGS), help="One of the 18 paper configurations (sets --agent/--model).")
    p.add_argument("--data-root", type=Path, default=None,
                   help="Extracted BioXArena-Data-Public directory (default: <repo>/../BioXArena-Data-Public).")
    p.add_argument("--output-root", type=Path, default=None,
                   help="Output root (default: <repo>/../BioXArena-Output). Results go to <output-root>/<agent>__<model>/<round>/<domain>/<task>/.")
    p.add_argument("--round-name", default="round1")
    p.add_argument("--task", action="append", default=[], help="Task key, e.g. chemical-biology/bace1-binding-affinity (repeatable).")
    p.add_argument("--domain", action="append", default=[], help="Run all tasks of a domain (repeatable).")
    p.add_argument("--part", type=int, default=None, help=f"Run part k of {NUM_PARTS} (19 tasks each, sorted by task key).")
    p.add_argument("--all-tasks", action="store_true")
    p.add_argument("--time-limit", type=int, default=7200, help="Wall-clock limit per task in seconds (paper: 7200).")
    p.add_argument("--max-workers", type=int, default=1, help="Tasks run in parallel (paper: 1).")
    p.add_argument("--num-gpus", type=int, default=1, help="GPUs visible to this job (CLI agents assign them round-robin).")
    p.add_argument("--biomni-data-dir", default=None, help="Biomni data lake (default: <prefix>/biomni_data).")
    p.add_argument("--check", action="store_true", help="Only send one short request to the backbone and exit.")
    p.add_argument("--list-configs", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print the runner command without running it.")
    args, passthrough = p.parse_known_args()

    if args.list_configs:
        print("Agents:", ", ".join(sorted(AGENTS)))
        print("\nPaper configurations:")
        for cid, (agent, model, extra) in PAPER_CONFIGS.items():
            print(f"  {cid:16s} {AGENTS[agent][1]:12s} {model:32s} {' '.join(extra)}")
        return 0

    extra: list[str] = []
    if args.config:
        agent, model, extra = PAPER_CONFIGS[args.config]
        args.agent = args.agent or agent
        args.model = args.model or model
    if not args.agent or not args.model:
        p.error("--agent and --model (or --config) are required")

    api_key = resolve_api_key(args.api_key)
    if args.check:
        return check_backbone(args.agent, args.model, api_key)

    data_root = (args.data_root or REPO_ROOT.parent / "BioXArena-Data-Public").resolve()
    output_root = (args.output_root or REPO_ROOT.parent / "BioXArena-Output").resolve()
    tasks = select_tasks(args, data_root)

    runner = TRAINING_DIR / AGENTS[args.agent][0]
    cmd = [sys.executable, str(runner)] + runner_args(args.agent, args.model, args, data_root, output_root) + extra
    for t in tasks:
        cmd += ["--task", t]
    cmd += passthrough

    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = api_key
    env.setdefault("PYTHONUNBUFFERED", "1")
    if args.agent == "stella":
        env.setdefault("STELLA_WAIT_BACKGROUND", "1")
        env.setdefault("STELLA_SAVE_TO_TASK_DIR", "1")

    print(f"Agent: {AGENTS[args.agent][1]}  Backbone: {args.model}  Tasks: {len(tasks)}  Output: {output_root}")
    print("Command:", " ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(TRAINING_DIR), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
