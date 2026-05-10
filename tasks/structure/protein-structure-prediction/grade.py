import pandas as pd
import numpy as np
import json
from pathlib import Path


def tm_score(pred_coords, true_coords):
    """Simplified TM-score calculation from Cα coordinates."""
    pred = np.array(pred_coords)
    true = np.array(true_coords)

    if len(pred) != len(true) or len(pred) == 0:
        return 0.0

    L = len(true)
    d0 = 1.24 * (L - 15) ** (1.0/3) - 1.8
    if d0 < 0.5:
        d0 = 0.5

    # Superpose using Kabsch algorithm
    pred_centered = pred - pred.mean(axis=0)
    true_centered = true - true.mean(axis=0)

    H = pred_centered.T @ true_centered
    U, S, Vt = np.linalg.svd(H)

    d = np.sign(np.linalg.det(Vt.T @ U.T))
    diag = np.eye(3)
    diag[2, 2] = d
    R = Vt.T @ diag @ U.T

    pred_aligned = (R @ pred_centered.T).T
    distances = np.sqrt(np.sum((pred_aligned - true_centered) ** 2, axis=1))

    tm = np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / L
    return float(tm)


def validate_first_column_match(submission: pd.DataFrame, answers: pd.DataFrame) -> None:
    if submission.shape[1] == 0 or answers.shape[1] == 0:
        raise ValueError("Submission and answers must both contain at least one column.")

    submission_first_col = submission.columns[0]
    answers_first_col = answers.columns[0]
    if submission_first_col != answers_first_col:
        raise ValueError(
            f"Submission first column '{submission_first_col}' does not match answers first column '{answers_first_col}'."
        )

    if submission.iloc[:, 0].tolist() != answers.iloc[:, 0].tolist():
        raise ValueError(
            f"Submission first column '{submission_first_col}' must exactly match answers first column, including order."
        )


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """Grade structure prediction using mean TM-score."""
    validate_first_column_match(submission, answers)
    submission = submission.sort_values("id").reset_index(drop=True)
    answers = answers.sort_values("id").reset_index(drop=True)

    task_dir = Path(__file__).resolve().parent / "coordinates"

    scores = []
    for i in range(len(answers)):
        # Load true coordinates
        true_file = task_dir / answers.iloc[i]["coords_file"]
        if not true_file.exists():
            continue

        with open(true_file) as f:
            true_coords = json.load(f)

        # Load predicted coordinates
        pred_file = submission.iloc[i]["coords_file"]
        pred_path = Path(pred_file)
        # if not pred_path.is_absolute():
        #     pred_path = task_dir.parent.parent / pred_file

        if not pred_path.exists():
            scores.append(0.0)
            continue

        with open(pred_path) as f:
            pred_coords = json.load(f)

        scores.append(tm_score(pred_coords, true_coords))

    raw = float(np.mean(scores)) if scores else 0.0
    return raw
