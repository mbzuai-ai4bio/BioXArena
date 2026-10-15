"""Shared OpenRouter settings for all BioXArena agent runners.

All agents in BioXArena are evaluated through the OpenRouter API (https://openrouter.ai).
The API key is resolved in this order:
  1. an explicit value (``--api-key``),
  2. the ``OPENROUTER_API_KEY`` environment variable,
  3. ``OPENROUTER_API_KEY=...`` (or the legacy ``api_key=...``) in ``<repo>/.env``.
The key is never written to logs, configs or output files by the runners.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = REPO_ROOT / ".env"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Anthropic-compatible endpoint used by Claude Code (`ANTHROPIC_BASE_URL`); the CLI appends /v1/messages.
OPENROUTER_ANTHROPIC_BASE_URL = "https://openrouter.ai/api"


def _read_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_api_key(explicit_key: str | None = None, env_path: Path = DEFAULT_ENV_PATH) -> str:
    if explicit_key:
        return explicit_key
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    values = _read_env_file(env_path)
    key = values.get("OPENROUTER_API_KEY") or values.get("api_key")
    if key:
        return key
    raise RuntimeError(
        "No OpenRouter API key found. Pass --api-key, set OPENROUTER_API_KEY, "
        f"or add OPENROUTER_API_KEY=... to {env_path}."
    )
