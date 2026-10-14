import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

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
    """Mean per-chain AUPRC for interface-residue prediction.

    The area under the precision-recall curve is computed separately for every chain
    (answers carry `chain_id`) and averaged, so a chain with many residues does not dominate
    and the score is not driven by differences in interface size between chains. A chain whose
    predictions are constant scores the share of interface residues of that chain, which is
    what a constant prediction earns under AUPRC.
    """
    validate_first_column_match(submission, answers)
    submission = validate_submission(submission, answers, id_col="id", pred_cols=["interface"])
    answers = answers.sort_values("id").reset_index(drop=True)

    pred = submission["interface"].astype(float).values
    true = answers["interface"].astype(int).values
    if not np.isfinite(pred).all():
        raise InvalidSubmissionError("Predictions must be finite numbers")

    scores = []
    for _, idx in answers.groupby("chain_id").indices.items():
        if len(idx) < 5 or true[idx].min() == true[idx].max():
            continue
        scores.append(float(average_precision_score(true[idx], pred[idx])))

    if not scores:
        raise InvalidSubmissionError("No chain had both interface and non-interface residues")

    return float(np.mean(scores))
