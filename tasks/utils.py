"""Shared grading and validation utilities for BioXBench tasks."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_squared_error,
    roc_auc_score,
)

class InvalidSubmissionError(Exception):
    """
    A custom exception for when the agent submission cannot be graded.
    """

    pass


def validate_submission(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    id_col: str = "id",
    pred_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Validate and align submission with answers. Returns aligned submission."""
    if id_col not in submission.columns:
        raise InvalidSubmissionError(f"Submission must contain '{id_col}' column")
    if pred_cols:
        for col in pred_cols:
            if col not in submission.columns:
                raise InvalidSubmissionError(f"Submission missing column: '{col}'")
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Submission has {len(submission)} rows, expected {len(answers)}"
        )

    submission = submission.sort_values(id_col).reset_index(drop=True)
    answers = answers.sort_values(id_col).reset_index(drop=True)

    if not (submission[id_col].values == answers[id_col].values).all():
        raise InvalidSubmissionError("Submission IDs do not match answer IDs")

    return submission


def grade_roc_auc(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    id_col: str = "id",
    pred_col: str = "prediction",
    label_col: str = "label",
) -> float:
    """Binary ROC-AUC score."""
    submission = validate_submission(submission, answers, id_col, [pred_col])
    answers = answers.sort_values(id_col).reset_index(drop=True)
    return float(roc_auc_score(answers[label_col].values, submission[pred_col].values))


def grade_macro_roc_auc(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    id_col: str = "id",
    label_cols: list[str] | None = None,
    missing_val: int = -1,
) -> float:
    """Macro-averaged ROC-AUC across multiple label columns.
    Skips columns where all true values are the same or where missing_val indicates untested.
    """
    submission = validate_submission(submission, answers, id_col, label_cols)
    answers = answers.sort_values(id_col).reset_index(drop=True)

    aucs = []
    for col in label_cols:
        y_true = answers[col].values
        y_score = submission[col].values
        # Filter out missing/untested entries
        mask = y_true != missing_val
        y_true_f = y_true[mask]
        y_score_f = y_score[mask]
        if len(np.unique(y_true_f)) > 1 and len(y_true_f) > 0:
            aucs.append(roc_auc_score(y_true_f, y_score_f))
    if not aucs:
        raise InvalidSubmissionError("No valid label columns for ROC-AUC computation")
    return float(np.mean(aucs))


def grade_macro_f1(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    id_col: str = "id",
    pred_col: str = "prediction",
    label_col: str = "label",
) -> float:
    """Macro F1 score for multi-class classification."""
    submission = validate_submission(submission, answers, id_col, [pred_col])
    answers = answers.sort_values(id_col).reset_index(drop=True)
    return float(
        f1_score(answers[label_col].values, submission[pred_col].values, average="macro")
    )


def grade_accuracy(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    id_col: str = "id",
    pred_col: str = "prediction",
    label_col: str = "label",
) -> float:
    """Accuracy score."""
    submission = validate_submission(submission, answers, id_col, [pred_col])
    answers = answers.sort_values(id_col).reset_index(drop=True)
    return float(accuracy_score(answers[label_col].values, submission[pred_col].values))


def grade_spearman(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    id_col: str = "id",
    pred_col: str = "prediction",
    target_col: str = "target",
) -> float:
    """Spearman rank correlation."""
    submission = validate_submission(submission, answers, id_col, [pred_col])
    answers = answers.sort_values(id_col).reset_index(drop=True)
    corr, _ = spearmanr(answers[target_col].values, submission[pred_col].values)
    return float(corr)


def grade_pearson(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    id_col: str = "id",
    pred_col: str = "prediction",
    target_col: str = "target",
) -> float:
    """Pearson correlation."""
    submission = validate_submission(submission, answers, id_col, [pred_col])
    answers = answers.sort_values(id_col).reset_index(drop=True)
    corr, _ = pearsonr(answers[target_col].values, submission[pred_col].values)
    return float(corr)


def grade_rmse(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    id_col: str = "id",
    pred_col: str = "prediction",
    target_col: str = "target",
) -> float:
    """Root mean squared error (lower is better; returned as negative for ranking)."""
    submission = validate_submission(submission, answers, id_col, [pred_col])
    answers = answers.sort_values(id_col).reset_index(drop=True)
    return float(
        np.sqrt(mean_squared_error(answers[target_col].values, submission[pred_col].values))
    )


def grade_mse(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    id_col: str = "id",
    pred_col: str = "prediction",
    target_col: str = "target",
) -> float:
    """Mean squared error (lower is better)."""
    submission = validate_submission(submission, answers, id_col, [pred_col])
    answers = answers.sort_values(id_col).reset_index(drop=True)
    return float(mean_squared_error(answers[target_col].values, submission[pred_col].values))


def grade_auprc(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    id_col: str = "id",
    pred_col: str = "prediction",
    label_col: str = "label",
) -> float:
    """Area under precision-recall curve."""
    submission = validate_submission(submission, answers, id_col, [pred_col])
    answers = answers.sort_values(id_col).reset_index(drop=True)
    return float(
        average_precision_score(answers[label_col].values, submission[pred_col].values)
    )


def grade_c_index(
    submission: pd.DataFrame,
    answers: pd.DataFrame,
    id_col: str = "id",
    risk_col: str = "risk_score",
    time_col: str = "time",
    event_col: str = "event",
) -> float:
    """Harrell's concordance index for survival analysis."""
    submission = validate_submission(submission, answers, id_col, [risk_col])
    answers = answers.sort_values(id_col).reset_index(drop=True)

    times = answers[time_col].values
    events = answers[event_col].values
    risk_scores = submission[risk_col].values

    concordant = 0
    discordant = 0
    tied_risk = 0

    for i in range(len(times)):
        if events[i] == 0:
            continue
        for j in range(len(times)):
            if i == j:
                continue
            if times[j] > times[i]:
                if risk_scores[j] < risk_scores[i]:
                    concordant += 1
                elif risk_scores[j] > risk_scores[i]:
                    discordant += 1
                else:
                    tied_risk += 1

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5
    return float((concordant + 0.5 * tied_risk) / total)
