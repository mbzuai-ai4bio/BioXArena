import pandas as pd
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
    validate_first_column_match(submission, answers)
    submission = validate_submission(
        submission, answers, id_col="id", pred_cols=["label"]
    )
    answers = answers.sort_values("id").reset_index(drop=True)

    # Normalize: lowercase, strip, sort pipe-separated multi-value answers
    def normalize(val: str) -> str:
        parts = sorted(p.strip().lower() for p in str(val).split("|"))
        return "|".join(parts)

    preds = submission["label"].apply(normalize)
    labels = answers["label"].apply(normalize)
    return float((preds == labels).mean())
