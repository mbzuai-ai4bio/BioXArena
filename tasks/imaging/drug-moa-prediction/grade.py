import pandas as pd
from tasks.utils import grade_macro_f1


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
    # Some labels can be parsed as mixed str/float (e.g., missing values).
    # Normalize both sides to string labels to keep sklearn macro-F1 type-consistent.
    validate_first_column_match(submission, answers)
    submission = submission.copy()
    answers = answers.copy()
    submission["moa"] = submission["moa"].fillna("Unknown").astype(str)
    answers["moa"] = answers["moa"].fillna("Unknown").astype(str)
    return grade_macro_f1(
        submission, answers, id_col="id", pred_col="moa", label_col="moa"
    )
