import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from tasks.utils import validate_submission
from tasks.utils import InvalidSubmissionError


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
    """Average protein-wise Pearson correlation across 134 proteins."""
    validate_first_column_match(submission, answers)
    submission = validate_submission(submission, answers, id_col="id")
    answers = answers.sort_values("id").reset_index(drop=True)

    protein_cols = [c for c in answers.columns if c.startswith("protein_")]
    if not protein_cols:
        raise InvalidSubmissionError("No protein columns found in answers")

    correlations = []
    for col in protein_cols:
        if col not in submission.columns:
            raise InvalidSubmissionError(f"Missing column: {col}")
        y_true = answers[col].values.astype(float)
        y_pred = submission[col].values.astype(float)
        if np.std(y_true) > 0 and np.std(y_pred) > 0:
            r, _ = pearsonr(y_true, y_pred)
            correlations.append(r)

    if not correlations:
        raise InvalidSubmissionError("No valid correlations computed")

    raw = float(np.mean(correlations))
    return raw
