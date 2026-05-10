import numpy as np
import pandas as pd
from pathlib import Path

try:
    import nibabel as nib
except ImportError:
    nib = None


NUM_ORGANS = 15  # organs labeled 1-15, 0 = background


def dice_score(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """Compute Dice score for a single organ label."""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    intersection = np.sum(pred_mask & gt_mask)
    total = np.sum(pred_mask) + np.sum(gt_mask)
    if total == 0:
        return 1.0  # Both empty = perfect match
    return 2.0 * intersection / total


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
    """Grade AMOS segmentation using mean Dice score across 15 organs.

    - submission has columns: id, prediction_file
      (prediction_file points to agent-generated NIfTI masks)
    - answers has columns: id, label_file, label_dir
      (label_file + label_dir point to ground truth NIfTI masks)
    """
    validate_first_column_match(submission, answers)
    if nib is None:
        raise ImportError(
            "nibabel is required for grading segmentation tasks. "
            "Install with: pip install nibabel"
        )

    submission = submission.sort_values("id").reset_index(drop=True)
    answers = answers.sort_values("id").reset_index(drop=True)

    task_dir = Path(__file__).resolve().parent

    all_dice = []
    for i in range(len(answers)):
        ans_row = answers.iloc[i]
        sub_row = submission.iloc[i]

        # Load ground truth
        gt_path = task_dir / ans_row["label_dir"] / ans_row["label_file"]
        if not gt_path.exists():
            continue

        gt_vol = nib.load(str(gt_path)).get_fdata().astype(np.int32)

        # Load prediction
        pred_path = Path(sub_row["prediction_file"])
        # if not pred_path.is_absolute():
        #     pred_path = task_dir / pred_path
        if not pred_path.exists():
            all_dice.append(0.0)
            continue

        pred_vol = nib.load(str(pred_path)).get_fdata().astype(np.int32)

        # Compute per-organ Dice (skip background label 0)
        organ_dices = []
        for label in range(1, NUM_ORGANS + 1):
            if np.sum(gt_vol == label) > 0:  # Only score organs present in GT
                d = dice_score(pred_vol, gt_vol, label)
                organ_dices.append(d)

        if organ_dices:
            all_dice.append(np.mean(organ_dices))

    if not all_dice:
        return 0.0

    return float(np.mean(all_dice))
