import pandas as pd
import numpy as np
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
    """Mean within-drug Spearman rank correlation between predicted and true ln_ic50.

    The correlation is computed separately for every drug (answers carry `drug_id`) and
    averaged, so the score reflects which cell lines are more sensitive to a given drug
    rather than which drugs are more potent. A drug whose predictions are constant
    contributes 0.
    """
    validate_first_column_match(submission, answers)
    submission = validate_submission(submission, answers, id_col="id", pred_cols=["ln_ic50"])
    answers = answers.sort_values("id").reset_index(drop=True)
    pred = submission["ln_ic50"].astype(float).values
    true = answers["ln_ic50"].astype(float).values
    scores = []
    for _, idx in answers.groupby("drug_id").indices.items():
        if len(idx) < 3 or np.ptp(true[idx]) == 0:
            continue
        if not np.isfinite(pred[idx]).all() or np.ptp(pred[idx]) == 0:
            scores.append(0.0)
            continue
        corr, _ = spearmanr(true[idx], pred[idx])
        scores.append(float(corr) if np.isfinite(corr) else 0.0)
    return float(np.mean(scores))
