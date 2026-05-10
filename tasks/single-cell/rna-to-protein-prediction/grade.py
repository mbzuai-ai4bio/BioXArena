import pandas as pd
import numpy as np

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
    submission = submission.sort_values("id").reset_index(drop=True)
    answers = answers.sort_values("id").reset_index(drop=True)
    protein_cols = [c for c in answers.columns if c.startswith('protein_')]
    corrs = []
    for c in protein_cols:
        corr = np.corrcoef(submission[c].values, answers[c].values)[0, 1]
        if not np.isnan(corr): corrs.append(corr)
    raw = float(np.mean(corrs)) if corrs else 0.0
    return raw if corrs else 0.0
