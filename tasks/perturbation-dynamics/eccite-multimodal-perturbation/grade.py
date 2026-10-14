import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
    """Mean per-protein Spearman rank correlation across test cells.

    Each `delta_protein_*` column is one surface protein. The correlation between predicted
    and true values is computed across cells for every protein and averaged. A protein whose
    predictions are constant contributes 0.
    """
    validate_first_column_match(submission, answers)
    submission = validate_submission(submission, answers, id_col="id")
    answers = answers.sort_values("id").reset_index(drop=True)

    protein_cols = [c for c in answers.columns if c.startswith("delta_protein_")]
    if not protein_cols:
        raise InvalidSubmissionError("No delta_protein columns found in answers")

    scores = []
    for col in protein_cols:
        if col not in submission.columns:
            raise InvalidSubmissionError(f"Missing column: {col}")
        y_true = answers[col].values.astype(float)
        y_pred = submission[col].values.astype(float)
        if np.ptp(y_true) == 0:
            continue
        if not np.isfinite(y_pred).all() or np.ptp(y_pred) == 0:
            scores.append(0.0)
            continue
        corr, _ = spearmanr(y_true, y_pred)
        scores.append(float(corr) if np.isfinite(corr) else 0.0)

    if not scores:
        raise InvalidSubmissionError("No valid correlations computed")

    return float(np.mean(scores))
