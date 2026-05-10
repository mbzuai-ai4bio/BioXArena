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
    """Grade genotype-to-phenotype prediction using Pearson correlation
    over donor×gene samples."""
    validate_first_column_match(submission, answers)
    submission = validate_submission(submission, answers, id_col="id", pred_cols=["expression"])
    answers = answers.sort_values("id").reset_index(drop=True)
    y_true = answers["expression"].values.astype(float)
    y_pred = submission["expression"].values.astype(float)
    if np.std(y_true) <= 0 or np.std(y_pred) <= 0:
        raise InvalidSubmissionError("No valid correlation: zero variance in prediction or target")
    r, _ = pearsonr(y_true, y_pred)
    raw = float(r)
    return raw
