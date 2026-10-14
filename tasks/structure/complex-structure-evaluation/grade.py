import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
    """Mean within-target Spearman rank correlation between predicted and true DockQ.

    The correlation is computed separately over the models of every target (answers carry
    `target_id`) and averaged, so the score reflects how well the models of one complex are
    ranked rather than which complex is easier. A target whose predictions are constant
    contributes 0.
    """
    validate_first_column_match(submission, answers)
    submission = validate_submission(submission, answers, id_col="id", pred_cols=["dockq_avg"])
    answers = answers.sort_values("id").reset_index(drop=True)
    pred = submission["dockq_avg"].astype(float).values
    true = answers["dockq_avg"].astype(float).values
    scores = []
    for _, idx in answers.groupby("target_id").indices.items():
        if len(idx) < 3 or np.ptp(true[idx]) == 0:
            continue
        if not np.isfinite(pred[idx]).all() or np.ptp(pred[idx]) == 0:
            scores.append(0.0)
            continue
        corr, _ = spearmanr(true[idx], pred[idx])
        scores.append(float(corr) if np.isfinite(corr) else 0.0)
    return float(np.mean(scores))
