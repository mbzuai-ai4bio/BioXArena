"""Submission validation server — validates submissions against BioXArena task data."""

import logging
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request
import pandas as pd

from config import load_cfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

_cfg = load_cfg()
_data_dir = Path(_cfg.dataset_dir)


def _resolve_sample_submission_path(exp_id: str) -> Path:
    task_root = (_data_dir / exp_id / "public").resolve()
    sample_path = task_root / "sample_submission.csv"
    if not sample_path.exists():
        raise FileNotFoundError(
            f"BioXArena sample submission not found: {sample_path}"
        )
    return sample_path


def _validate_against_sample(submission_path: Path, sample_path: Path) -> tuple[bool, str]:
    sample_df = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
    submission_df = pd.read_csv(submission_path, dtype=str, keep_default_na=False)

    if submission_df.shape[0] != sample_df.shape[0]:
        return (
            False,
            f"Row count mismatch: expected {sample_df.shape[0]}, got {submission_df.shape[0]}.",
        )

    sample_columns = sample_df.columns.tolist()
    submission_columns = submission_df.columns.tolist()
    if submission_columns != sample_columns:
        return (
            False,
            f"Column mismatch: expected {sample_columns}, got {submission_columns}.",
        )

    if sample_columns:
        first_column = sample_columns[0]
        if not submission_df[first_column].equals(sample_df[first_column]):
            return (
                False,
                f"First column `{first_column}` values/order do not match sample_submission.csv.",
            )

    return True, "Submission format matches sample_submission.csv."


@app.post("/validate")
def validate():
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' in request"}), 400

    competition_id = request.headers.get("exp-id")
    if not competition_id:
        return jsonify({"error": "Missing 'exp-id' header"}), 400

    # Save uploaded file to a temp path (auto-cleaned)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        request.files["file"].save(tmp)
        tmp_path = Path(tmp.name)

    try:
        sample_path = _resolve_sample_submission_path(competition_id)
        is_valid, message = _validate_against_sample(tmp_path, sample_path)
        return jsonify({"is_valid": is_valid, "result": message})
    except Exception as e:
        logger.exception("Validation failed")
        return jsonify({"error": "Validation failed", "details": str(e)}), 500
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/health")
def health():
    return jsonify({"status": "running"})


if __name__ == "__main__":
    port = int(os.getenv("GRADING_SERVER_PORT", "5005"))
    app.run(host="0.0.0.0", port=port)
