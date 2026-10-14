import pandas as pd
import numpy as np

from tasks.utils import grade_accuracy, validate_submission


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


RELATIVE_TOLERANCE = 0.10


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """Counting accuracy: fraction of images whose predicted count is within 10% of the true count.

    A prediction counts as correct if |predicted - true| <= max(0.10 * true, 1), i.e. within
    10% of the true number of cells, with a minimum tolerance of one cell.
    """
    validate_first_column_match(submission, answers)
    submission = validate_submission(submission, answers, id_col="id", pred_cols=["n_cells"])
    answers = answers.sort_values("id").reset_index(drop=True)
    pred = pd.to_numeric(submission["n_cells"], errors="coerce").values.astype(float)
    true = answers["n_cells"].values.astype(float)
    within = np.abs(pred - true) <= np.maximum(RELATIVE_TOLERANCE * true, 1.0)
    within &= np.isfinite(pred)
    hits = pd.DataFrame({"id": answers["id"].values, "within_tolerance": within})
    target = pd.DataFrame({"id": answers["id"].values, "within_tolerance": True})
    return grade_accuracy(hits, target, id_col="id", pred_col="within_tolerance", label_col="within_tolerance")
