"""Utilities for running `mlebench grade-sample` on best submissions."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_report_json(output: str) -> dict[str, Any] | None:
    """Parse the last JSON object containing `competition_id` from CLI output.

    Args:
        output: Combined stdout/stderr text from mlebench.

    Returns:
        dict[str, Any] | None: Parsed report JSON when found, otherwise `None`.
    """
    matches = re.findall(r"(\{[\s\S]*?\})", output)
    for chunk in reversed(matches):
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict) and "competition_id" in obj:
                return obj
        except Exception:
            pass
    return None


def grade_best_submission_and_save(
    *,
    workspace_dir: Path,
    competition_id: str,
    out_name: str = "mlebench_grade.json",
    overwrite: bool = False,
) -> Path:
    """Grade the newest CSV in `best_submission` and persist a JSON report.

    Args:
        workspace_dir: Workspace root that contains `best_submission`.
        competition_id: Competition identifier passed to mlebench.
        out_name: Output JSON file name under workspace directory.
        overwrite: Whether to overwrite an existing output file.

    Returns:
        Path: Path to the saved grading report JSON.
    """
    workspace_dir = Path(workspace_dir)
    out_path = workspace_dir / out_name

    if out_path.exists() and not overwrite:
        return out_path

    payload: dict[str, Any] = {
        "competition_id": competition_id,
        "workspace_dir": str(workspace_dir),
        "created_at": datetime.now().isoformat(),
        "status": "unknown",
    }

    best_dir = workspace_dir / "best_submission"
    csvs = list(best_dir.glob("*.csv"))
    if not csvs:
        payload.update(
            status="skipped",
            error="best_submission_csv_not_found",
            expected_dir=str(best_dir),
        )
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    # Use the most recently modified submission when multiple files exist.
    csvs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    csv_path = csvs[0]
    payload["submission_csv"] = str(csv_path)

    if shutil.which("mlebench") is None:
        payload.update(status="failed", error="mlebench_not_found_in_PATH")
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    cmd = ["mlebench", "grade-sample", str(csv_path), competition_id]
    process = subprocess.run(cmd, capture_output=True, text=True)
    combined = (process.stdout or "") + "\n" + (process.stderr or "")

    report = _parse_report_json(combined)
    payload.update(
        status="completed" if process.returncode == 0 and report is not None else "failed",
        returncode=process.returncode,
        report=report,
        raw_output_tail=combined[-6000:],
    )

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
