import pandas as pd
from tasks.utils import grade_macro_roc_auc

GO_COLS = [
    "GO_0006915", "GO_0006281", "GO_0007049", "GO_0006412", "GO_0006468",
    "GO_0008150", "GO_0016310", "GO_0006355", "GO_0006950", "GO_0006259",
    "GO_0006351", "GO_0006260", "GO_0006629", "GO_0006810", "GO_0006886",
]


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
    validate_first_column_match(submission, answers)
    return grade_macro_roc_auc(submission, answers, id_col="id", label_cols=GO_COLS, missing_val=-1)
