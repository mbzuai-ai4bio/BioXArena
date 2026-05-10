import pandas as pd
from tasks.utils import grade_roc_auc


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


def encode_diagnosis_labels(
    submission: pd.DataFrame, answers: pd.DataFrame, column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    submission = submission.copy()
    answers = answers.copy()

    labels = pd.Index(
        pd.concat([answers[column], submission[column]], ignore_index=True)
        .dropna()
        .astype(str)
        .unique()
    )
    if len(labels) != 2:
        raise ValueError(
            f"ROC-AUC for autism-diagnosis expects exactly 2 labels, found {len(labels)}: {labels.tolist()}"
        )

    mapping = {label: float(idx) for idx, label in enumerate(sorted(labels))}
    submission[column] = submission[column].astype(str).map(mapping)
    answers[column] = answers[column].astype(str).map(mapping)

    if submission[column].isna().any() or answers[column].isna().any():
        raise ValueError("Failed to map all diagnosis labels to numeric values.")

    return submission, answers


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """Grade autism diagnosis prediction using ROC-AUC."""
    validate_first_column_match(submission, answers)
    submission, answers = encode_diagnosis_labels(submission, answers, "diagnosis")
    return grade_roc_auc(
        submission, answers, id_col="id", pred_col="diagnosis", label_col="diagnosis"
    )
