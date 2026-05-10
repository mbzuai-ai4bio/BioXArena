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
    """Compute mean Pearson correlation across RNA and protein delta columns.

    Evaluates RNA and protein modalities separately, then averages.
    """
    validate_first_column_match(submission, answers)
    submission = validate_submission(submission, answers, id_col="id")
    answers = answers.sort_values("id").reset_index(drop=True)

    rna_cols = [c for c in answers.columns if c.startswith("delta_rna_")]
    protein_cols = [c for c in answers.columns if c.startswith("delta_protein_")]

    if not rna_cols and not protein_cols:
        raise InvalidSubmissionError("No delta columns found in answers")

    modality_scores = []

    # RNA modality
    if rna_cols:
        rna_corrs = []
        for col in rna_cols:
            if col not in submission.columns:
                raise InvalidSubmissionError(f"Missing column: {col}")
            y_true = answers[col].values.astype(float)
            y_pred = submission[col].values.astype(float)
            if np.std(y_true) > 0 and np.std(y_pred) > 0:
                r, _ = pearsonr(y_true, y_pred)
                rna_corrs.append(r)
        if rna_corrs:
            modality_scores.append(float(np.mean(rna_corrs)))

    # Protein modality
    if protein_cols:
        prot_corrs = []
        for col in protein_cols:
            if col not in submission.columns:
                raise InvalidSubmissionError(f"Missing column: {col}")
            y_true = answers[col].values.astype(float)
            y_pred = submission[col].values.astype(float)
            if np.std(y_true) > 0 and np.std(y_pred) > 0:
                r, _ = pearsonr(y_true, y_pred)
                prot_corrs.append(r)
        if prot_corrs:
            modality_scores.append(float(np.mean(prot_corrs)))

    if not modality_scores:
        raise InvalidSubmissionError("No valid correlations computed")

    raw = float(np.mean(modality_scores))
    return raw
