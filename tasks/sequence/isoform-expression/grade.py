import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tasks.utils import InvalidSubmissionError
from tasks.utils import validate_submission

NUM_TISSUES = 30


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
    """Compute mean Spearman correlation across 30 tissue expression outputs."""
    validate_first_column_match(submission, answers)
    submission = validate_submission(submission, answers, id_col="id")
    answers = answers.sort_values("id").reset_index(drop=True)

    label_cols = [f"labels_{i}" for i in range(NUM_TISSUES)]
    for col in label_cols:
        if col not in answers.columns:
            raise InvalidSubmissionError(f"Missing column in answers: {col}")
        if col not in submission.columns:
            raise InvalidSubmissionError(f"Missing column in submission: {col}")

    correlations = []
    for col in label_cols:
        y_true = answers[col].values.astype(float)
        y_pred = submission[col].values.astype(float)
        if np.std(y_true) > 0 and np.std(y_pred) > 0:
            r, _ = spearmanr(y_true, y_pred)
            correlations.append(r)

    if not correlations:
        raise InvalidSubmissionError("No valid correlations computed")

    raw = float(np.mean(correlations))
    return raw
