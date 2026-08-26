import numpy as np
import pandas as pd
from pathlib import Path

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    from numba import njit
except ImportError:
    njit = None


NUM_ORGANS = 15  # organs labeled 1-15, 0 = background


if njit is not None:
    @njit(cache=False)
    def _count_organ_voxels(pred: np.ndarray, gt: np.ndarray):
        """Count prediction, truth and intersection voxels in one array pass."""
        pred_counts = np.zeros(NUM_ORGANS + 1, dtype=np.int64)
        gt_counts = np.zeros(NUM_ORGANS + 1, dtype=np.int64)
        intersection_counts = np.zeros(NUM_ORGANS + 1, dtype=np.int64)
        pred_flat = pred.ravel()
        gt_flat = gt.ravel()
        for index in range(gt_flat.size):
            pred_label = pred_flat[index]
            gt_label = gt_flat[index]
            if 0 <= pred_label <= NUM_ORGANS:
                pred_counts[pred_label] += 1
            if 0 <= gt_label <= NUM_ORGANS:
                gt_counts[gt_label] += 1
            if pred_label == gt_label and 0 <= gt_label <= NUM_ORGANS:
                intersection_counts[gt_label] += 1
        return pred_counts, gt_counts, intersection_counts
else:
    _count_organ_voxels = None


def dice_score(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """Compute Dice score for a single organ label."""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    intersection = np.sum(pred_mask & gt_mask)
    total = np.sum(pred_mask) + np.sum(gt_mask)
    if total == 0:
        return 1.0  # Both empty = perfect match
    return 2.0 * intersection / total


def load_label_volume(path: Path) -> np.ndarray:
    """Load integer label maps without an unnecessary float64 intermediate."""
    image = nib.load(str(path))
    data_dtype = image.get_data_dtype()
    slope = float(image.dataobj.slope)
    intercept = float(image.dataobj.inter)
    if (
        np.issubdtype(data_dtype, np.integer)
        and slope == 1.0
        and intercept == 0.0
    ):
        return np.asarray(image.dataobj, dtype=np.int32)
    return image.get_fdata().astype(np.int32)


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
    - answers has columns: id, label_file
      (label_file points to a ground-truth NIfTI mask)
    """
    validate_first_column_match(submission, answers)
    if "prediction_file" not in submission.columns:
        raise ValueError("Submission must contain a 'prediction_file' column.")
    if "label_file" not in answers.columns:
        raise ValueError("Answers must contain a 'label_file' column.")
    if nib is None:
        raise ImportError(
            "nibabel is required for grading segmentation tasks. "
            "Install with: pip install nibabel"
        )

    submission = submission.sort_values("id").reset_index(drop=True)
    answers = answers.sort_values("id").reset_index(drop=True)

    all_dice = []
    for i in range(len(answers)):
        ans_row = answers.iloc[i]
        sub_row = submission.iloc[i]

        # Load ground truth
        gt_path = Path(ans_row["label_file"])
        if not gt_path.is_file():
            raise FileNotFoundError(f"Ground-truth label file not found: {gt_path}")

        # Load prediction
        pred_path = Path(sub_row["prediction_file"])
        if not pred_path.is_file():
            all_dice.append(0.0)
            continue

        gt_vol = load_label_volume(gt_path)
        pred_vol = load_label_volume(pred_path)

        # Compute per-organ Dice (skip background label 0). The optional Numba
        # path is algebraically identical but scans each 3D volume once instead
        # of up to 45 times. Preserve the reference path for shape mismatches and
        # environments without Numba so validation/failure semantics stay stable.
        organ_dices = []
        if _count_organ_voxels is not None and pred_vol.shape == gt_vol.shape:
            pred_counts, gt_counts, intersections = _count_organ_voxels(pred_vol, gt_vol)
            for label in range(1, NUM_ORGANS + 1):
                if gt_counts[label] > 0:
                    total = pred_counts[label] + gt_counts[label]
                    organ_dices.append(2.0 * intersections[label] / total)
        else:
            for label in range(1, NUM_ORGANS + 1):
                if np.sum(gt_vol == label) > 0:  # Only score organs present in GT
                    d = dice_score(pred_vol, gt_vol, label)
                    organ_dices.append(d)

        if organ_dices:
            all_dice.append(np.mean(organ_dices))

    if not all_dice:
        return 0.0

    return float(np.mean(all_dice))
