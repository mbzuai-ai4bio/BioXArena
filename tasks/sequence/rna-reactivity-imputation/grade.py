import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from tasks.utils import InvalidSubmissionError
from tasks.utils import validate_submission


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
    """Compute mean per-sample Pearson correlation on non-NaN target positions."""
    validate_first_column_match(submission, answers)
    submission = validate_submission(submission, answers, id_col="id")
    answers = answers.sort_values("id").reset_index(drop=True)

    target_cols = [c for c in answers.columns if c.startswith("target_")]
    if not target_cols:
        raise InvalidSubmissionError("No target columns found in answers")

    for col in target_cols:
        if col not in submission.columns:
            raise InvalidSubmissionError(f"Missing column: {col}")

    # Compute per-sample Pearson correlation on non-NaN positions
    correlations = []
    for idx in range(len(answers)):
        y_true = np.array(
            [answers[col].iloc[idx] for col in target_cols], dtype=float
        )
        y_pred = np.array(
            [submission[col].iloc[idx] for col in target_cols], dtype=float
        )

        # Only compare at positions where both are non-NaN
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if mask.sum() < 2:
            continue

        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]

        if np.std(y_true_masked) > 0 and np.std(y_pred_masked) > 0:
            r, _ = pearsonr(y_true_masked, y_pred_masked)
            correlations.append(r)

    if not correlations:
        raise InvalidSubmissionError("No valid per-sample correlations computed")

    raw = float(np.mean(correlations))
    return raw
