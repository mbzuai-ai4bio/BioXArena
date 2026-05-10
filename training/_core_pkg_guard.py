"""Shared helper: restore core scientific packages to pinned versions.

Used by every runner (general_llm / biomni / stella / mlmaster / mlevolve) at the
start of each task to keep numpy / pandas / scipy / Pillow from drifting after a
LLM-issued `pip install`.

Idempotent and cheap: pip's check skips already-correct versions in <2s.
"""
from __future__ import annotations

import subprocess
import sys

CORE_PINS = [
    "numpy==1.26.4",
    "pandas==2.3.1",
    "scipy==1.13.1",
    "Pillow==11.1.0",
]


def restore_core_packages(verbose: bool = False) -> None:
    """Force-reinstall core packages to pinned versions if drifted.

    Cheap because pip uses cached wheels; no-op if versions already match
    in most pip versions, otherwise re-installs in seconds.
    """
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-deps", "--quiet", "--disable-pip-version-check",
        *CORE_PINS,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if verbose and result.returncode != 0:
            print(f"[core-pkg-guard] pip exit={result.returncode}: {result.stderr[:200]}")
    except Exception as exc:
        if verbose:
            print(f"[core-pkg-guard] failed to restore core packages: {exc}")
