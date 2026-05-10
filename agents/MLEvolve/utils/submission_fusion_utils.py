import os
import argparse
import json
import shutil
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Optional, Tuple


# ────────────────────── Ensemble Configuration ──────────────────────


@dataclass
class EnsembleConfig:
    """All tuneable parameters for the ensemble pipeline."""

    # -- Format detection thresholds --
    id_col_unique_ratio: float = 0.5        # min uniqueness to treat col-0 as ID
    passthrough_avg_len: int = 200           # avg str length above which a col is passthrough
    normalize_tol: float = 0.05             # max |row_sum - 1| to trigger row normalization
    normalize_zero_eps: float = 1e-10       # threshold for treating row sum as zero

    # -- Candidate selection --
    max_candidates: int = 6                 # keep at most this many top solutions
    ensemble_sizes: List[int] = field(      # sizes to sweep when > 4 candidates
        default_factory=lambda: [1, 2, 3, 4, 6]
    )
    small_candidate_threshold: int = 4      # if n_valid <= this, try all sizes 1..n

    # -- Time budget --
    max_total_time_hours: float = 9.0       # stop if cumulative exec time exceeds this

    # -- Weighting --
    weight_eps: float = 1e-6                # epsilon for inverse-metric weighting

    # -- Logging --
    cellwise_log_interval: int = 200_000    # print progress every N rows in cellwise fusion


# ────────────────────── Path Utilities ──────────────────────


def get_closest_run_dir(runs_root: str, exp_name: str) -> Optional[Path]:
    root = Path(runs_root).expanduser().resolve()
    if not root.is_dir():
        return None
    key = "_".join(exp_name.split("_", 2)[2:])
    try:
        input_ts = datetime.strptime(
            "_".join(exp_name.split("_", 2)[:2]), "%Y%m%d_%H%M%S"
        )
    except ValueError:
        return None
    candidates = []
    for d in root.iterdir():
        if not d.is_dir() or not d.name.endswith(key):
            continue
        try:
            ts = datetime.strptime("_".join(d.name.split("_", 2)[:2]), "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        candidates.append((abs((ts - input_ts).total_seconds()), d))
    if not candidates:
        return None
    return min(candidates, key=lambda t: t[0])[1].resolve()


def parse_metric(metric_file: str) -> Tuple[float, bool, float]:
    with open(metric_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    d = {}
    for ln in lines:
        if ":" in ln:
            k, v = ln.split(":", 1)
            d[k.strip()] = v.strip()
    value = float(d["Metric"])
    maximize = d["Maximize"].lower() in {"true", "1", "yes"}
    exe_time = float(d["Execution Time(s)"])
    return value, maximize, exe_time



UNSUPPORTED_TASK_TYPES = {"Detection", "Segmentation"}

_tag_cache: Optional[Dict[str, str]] = None


def _load_tags(tag_path: str) -> Dict[str, str]:
    global _tag_cache
    if _tag_cache is None:
        if os.path.isfile(tag_path):
            with open(tag_path, "r", encoding="utf-8") as f:
                _tag_cache = json.load(f)
        else:
            _tag_cache = {}
    return _tag_cache


def is_structured_output_task(task_id: str, tag_path: str) -> bool:
    """Check if a task produces structured output based on its category tag."""
    tags = _load_tags(tag_path)
    category = tags.get(task_id, "")
    return category in UNSUPPORTED_TASK_TYPES


# ────────────────────── Format Detection ──────────────────────


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def detect_format(df: pd.DataFrame, cfg: EnsembleConfig) -> Dict:
    """
    Detect submission format from actual submission values.
    Pure data-driven: no keyword lists, no per-competition config.
    """
    n_cols = len(df.columns)
    n_rows = len(df)
    col_names = list(df.columns)

    # ── Step 1: Identify ID column ──
    col0_unique_ratio = df.iloc[:, 0].nunique() / max(n_rows, 1)
    if col0_unique_ratio > cfg.id_col_unique_ratio:
        id_col_names = [col_names[0]]
    else:
        id_col_names = []

    # ── Step 2: Classify remaining columns ──
    pred_col_names = []
    passthrough_col_names = []
    start_cols = col_names[1:] if id_col_names else col_names
    for col in start_cols:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            passthrough_col_names.append(col)
            continue
        avg_len = series.str.len().mean()
        if avg_len > cfg.passthrough_avg_len:
            passthrough_col_names.append(col)
        else:
            pred_col_names.append(col)

    if not pred_col_names:
        pred_col_names = [col_names[-1]]

    # ── Step 3: Detect fusion strategy from actual values ──
    sample = df[pred_col_names[0]].dropna()
    sample_str = sample.astype(str)
    has_spaces = sample_str.str.contains(" ").any()

    all_int_tokens = False
    if has_spaces:
        first_tokens = sample_str.iloc[0].split()
        if all(_is_numeric(t) for t in first_tokens):
            # Check if all rows have the same token count (fixed-length).
            # Variable-length sequences (e.g., permutations) can't be fused per-position.
            token_counts = sample_str.str.split().str.len()
            if token_counts.nunique() == 1:
                fusion = "cellwise"
            else:
                fusion = "text_vote"
            all_int_tokens = all(float(t) == int(float(t)) for t in first_tokens)
        else:
            fusion = "text_vote"
    else:
        numeric_ok = sample_str.apply(_is_numeric).all()
        if not numeric_ok:
            fusion = "text_vote"
        else:
            nums = pd.to_numeric(sample, errors="coerce")
            all_int = (nums == nums.round()).all()
            if all_int:
                fusion = "vote"
            else:
                fusion = "average"

    # ── Step 4: Check row normalization constraint (sum ~ 1) ──
    normalize = False
    if len(pred_col_names) > 1 and fusion == "average":
        try:
            num_df = df[pred_col_names].apply(pd.to_numeric, errors="coerce")
            row_sums = num_df.sum(axis=1)
            non_zero = row_sums[row_sums.abs() > cfg.normalize_zero_eps]
            if len(non_zero) > 0 and (non_zero - 1.0).abs().max() < cfg.normalize_tol:
                normalize = True
        except Exception:
            pass

    return {
        "id_col_names": id_col_names,
        "pred_col_names": pred_col_names,
        "passthrough_col_names": passthrough_col_names,
        "fusion": fusion,
        "normalize": normalize,
        "int_tokens": all_int_tokens,
    }


# ────────────────────── Fusion Functions ──────────────────────


def fuse_average(dfs: List[pd.DataFrame], weights: np.ndarray) -> np.ndarray:
    stacked = np.stack([df.values.astype(float) for df in dfs], axis=-1)
    return np.average(stacked, axis=-1, weights=weights)


def _vote_column(col_arrays: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    n_rows = len(col_arrays[0])
    n_models = len(col_arrays)
    all_same = col_arrays[0]
    unanimous = np.ones(n_rows, dtype=bool)
    for m in range(1, n_models):
        unanimous &= (col_arrays[m] == all_same)
    if unanimous.all():
        return all_same
    mat = np.column_stack(col_arrays)
    result = np.empty(n_rows, dtype=object)
    for r in range(n_rows):
        if unanimous[r]:
            result[r] = all_same[r]
        else:
            scores: Dict[str, float] = {}
            for m in range(n_models):
                v = mat[r, m]
                scores[v] = scores.get(v, 0.0) + weights[m]
            result[r] = max(scores, key=scores.get)
    return result


def fuse_vote(dfs: List[pd.DataFrame], weights: np.ndarray) -> np.ndarray:
    n_rows, n_cols = dfs[0].shape
    result = np.empty((n_rows, n_cols), dtype=object)
    for c in range(n_cols):
        col_arrays = [df.iloc[:, c].astype(str).values for df in dfs]
        result[:, c] = _vote_column(col_arrays, weights)
    return result


def fuse_text_vote(dfs: List[pd.DataFrame], weights: np.ndarray) -> np.ndarray:
    n_rows, n_cols = dfs[0].shape
    result = np.empty((n_rows, n_cols), dtype=object)
    for c in range(n_cols):
        col_arrays = [df.iloc[:, c].fillna("").astype(str).str.strip().values for df in dfs]
        result[:, c] = _vote_column(col_arrays, weights)
    return result


def fuse_cellwise(dfs: List[pd.DataFrame], weights: np.ndarray,
                  log_interval: int = 200_000) -> np.ndarray:
    n_rows, n_cols = dfs[0].shape
    result = np.empty((n_rows, n_cols), dtype=object)
    for c in range(n_cols):
        all_strs = [df.iloc[:, c].fillna("").astype(str).values for df in dfs]
        unanimous = np.ones(n_rows, dtype=bool)
        for m in range(1, len(all_strs)):
            unanimous &= (all_strs[m] == all_strs[0])
        result[:, c] = all_strs[0].copy()
        disagree_idx = np.where(~unanimous)[0]
        n_disagree = len(disagree_idx)
        if n_disagree == 0:
            print(f"  Column {c + 1}/{n_cols}: all {n_rows} rows unanimous, skipped.")
            continue
        print(f"  Column {c + 1}/{n_cols}: voting on {n_disagree}/{n_rows} disagreeing rows...")
        for i, r in enumerate(disagree_idx):
            token_lists = [s[r].split() for s in all_strs]
            max_len = max(len(tl) for tl in token_lists)
            for tl in token_lists:
                tl.extend([""] * (max_len - len(tl)))
            fused = []
            for pos in range(max_len):
                scores: Dict[str, float] = {}
                for m in range(len(token_lists)):
                    v = token_lists[m][pos]
                    if v:
                        scores[v] = scores.get(v, 0.0) + weights[m]
                if scores:
                    fused.append(max(scores, key=scores.get))
            result[r, c] = " ".join(fused)
            if log_interval > 0 and (i + 1) % log_interval == 0:
                print(f"    {i + 1}/{n_disagree} rows done...")
    return result


# ────────────────────── Weighting ──────────────────────


def get_weights(metrics: List[float], maximize_flags: List[bool],
                cfg: EnsembleConfig) -> np.ndarray:
    n = len(metrics)
    raw = []
    for i, (m, maxim) in enumerate(zip(metrics, maximize_flags)):
        score = m if maxim else 1.0 / (m + cfg.weight_eps)
        score *= n - i
        raw.append(score)
    raw = np.array(raw, dtype=float)
    if raw.sum() == 0:
        return np.ones(n) / n
    return raw / raw.sum()


# ────────────────────── Main Ensemble ──────────────────────


def ensemble(args):
    cfg = EnsembleConfig()
    exp_name = get_closest_run_dir(args.runs_root, args.exp_name)
    if exp_name is None:
        raise FileNotFoundError(f"No matching run directory found for exp_name={args.exp_name}")
    print(f"Experiment: {exp_name}")
    base_dir = f"{exp_name}/workspace/"

    # Select solution directory
    if args.use_llm_selection and os.path.exists(os.path.join(base_dir, "top_solution_llm/")):
        top_dir = os.path.join(base_dir, "top_solution_llm/")
    else:
        top_dir = os.path.join(base_dir, "top_solution/")
    print(f"Solution dir: {top_dir}")

    # ── 1. Discover top{i} directories ──
    all_dirs = []
    i = 1
    while True:
        path = os.path.join(top_dir, f"top{i}")
        if not os.path.isdir(path):
            break
        all_dirs.append(path)
        i += 1

    if not all_dirs:
        print("[WARN] No top{i} dirs found. Using best_submission as top1.")
        best_sub = os.path.join(base_dir, "best_submission/submission.csv")
        if not os.path.isfile(best_sub):
            raise FileNotFoundError(f"Missing: {best_sub}")
        top1_path = os.path.join(top_dir, "top1")
        os.makedirs(top1_path, exist_ok=True)
        shutil.copy(best_sub, os.path.join(top1_path, "submission.csv"))
        with open(os.path.join(top1_path, "metric.txt"), "w") as f:
            f.write("Metric: 1.0\nMaximize: True\nExecution Time(s): 1.0\n")
        all_dirs = [Path(top1_path)]

    print(f"Found {len(all_dirs)} candidate solutions")

    # ── 2. Skip Detection & Segmentation tasks ──
    if is_structured_output_task(args.task_id, args.tag_path):
        print(f"[SKIP] '{args.task_id}' is Detection/Segmentation, using top1 only.")
        _copy_top1_as_ensemble(base_dir)
        return

    # ── 3. Auto-detect format from first submission ──
    first_sub = pd.read_csv(os.path.join(all_dirs[0], "submission.csv"))
    fmt = detect_format(first_sub, cfg)
    pred_col_names = fmt["pred_col_names"]
    id_col_names = fmt["id_col_names"]
    print(f"Auto-detected format:")
    print(f"  ID cols:          {id_col_names}")
    print(f"  Pred cols:        {pred_col_names}")
    print(f"  Passthrough cols: {fmt['passthrough_col_names']}")
    print(f"  Fusion strategy:  {fmt['fusion']}")
    print(f"  Normalize (sum1): {fmt['normalize']}")

    # ── 4. Decide ensemble sizes to try ──
    all_dirs = all_dirs[:cfg.max_candidates]
    n_valid = len(all_dirs)
    if n_valid <= cfg.small_candidate_threshold:
        num_use = list(range(1, n_valid + 1))
    else:
        num_use = [x for x in cfg.ensemble_sizes if x <= n_valid]

    # ── 5. Run ensemble for each size ──
    t_max = cfg.max_total_time_hours * 3600
    for select_num in num_use:
        use_dirs = all_dirs[:select_num]
        k = len(use_dirs)

        metrics, maximize_flags, pred_dfs = [], [], []
        t_total = 0
        ref_df = None

        for td in use_dirs:
            metric_file = os.path.join(td, "metric.txt")
            sub_file = os.path.join(td, "submission.csv")
            if not (os.path.isfile(metric_file) and os.path.isfile(sub_file)):
                raise FileNotFoundError(f"Missing metric.txt or submission.csv in {td}")

            m, maxim, t_new = parse_metric(metric_file)
            t_total += t_new
            metrics.append(m)
            maximize_flags.append(maxim)

            df = pd.read_csv(sub_file)
            if ref_df is None:
                ref_df = df
            else:
                df = _align_submission(df, ref_df, id_col_names)

            available = [c for c in pred_col_names if c in df.columns]
            pred_dfs.append(df[available].copy())

        if t_total > t_max:
            print(f"[WARN] Total time {t_total}s > {t_max}s limit, stopping.")
            break

        weights = get_weights(metrics, maximize_flags, cfg)
        print(f"\nEnsembling {k} models:")
        for i, (w, m) in enumerate(zip(weights, metrics), 1):
            print(f"  Top{i}: weight={w:.4f}, metric={m:.5f}")

        # ── 6. Fuse predictions ──
        if k == 1:
            result_df = ref_df.copy()
        else:
            fusion_fn = {
                "average": fuse_average,
                "vote": fuse_vote,
                "text_vote": fuse_text_vote,
                "cellwise": lambda dfs, w: fuse_cellwise(dfs, w, cfg.cellwise_log_interval),
            }[fmt["fusion"]]

            ensemble_pred = fusion_fn(pred_dfs, weights)

            # ── 7. Reassemble output ──
            result_df = ref_df.copy()
            fused_cols = pred_dfs[0].columns.tolist()
            for j, col_name in enumerate(fused_cols):
                result_df[col_name] = ensemble_pred[:, j] if ensemble_pred.ndim > 1 else ensemble_pred

            # ── 8. Post-process ──
            if fmt["fusion"] == "cellwise" and fmt["int_tokens"]:
                for col_name in fused_cols:
                    result_df[col_name] = result_df[col_name].astype(str).apply(
                        lambda cell: " ".join(
                            str(int(float(tok))) if tok.replace(".", "", 1).replace("-", "", 1).isdigit()
                            else tok
                            for tok in cell.split()
                        ) if cell and cell != "nan" else cell
                    )

            if fmt["normalize"]:
                row_sums = result_df[fused_cols].sum(axis=1).replace(0, np.nan)
                result_df[fused_cols] = result_df[fused_cols].div(row_sums, axis=0).fillna(0)

        # ── 9. Restore ID column dtype ──
        for col in id_col_names:
            if col in result_df.columns and col in ref_df.columns:
                result_df[col] = result_df[col].astype(ref_df[col].dtype)

        # ── 10. Save ──
        save_dir = os.path.join(base_dir, "ensembles_csv/")
        os.makedirs(save_dir, exist_ok=True)
        t_h = round(t_total / 3600, 2)
        out_file = os.path.join(save_dir, f"top{k}ens-total_run_time{t_h}h.csv")
        result_df.to_csv(out_file, index=False)
        print(f"Saved: {out_file}")
        print(f"  Preview: {result_df.iloc[0, :3].to_dict()}")


def _copy_top1_as_ensemble(base_dir: str):
    top1_sub = os.path.join(base_dir, "top_solution/top1/submission.csv")
    if not os.path.isfile(top1_sub):
        top1_sub = os.path.join(base_dir, "best_submission/submission.csv")
    if not os.path.isfile(top1_sub):
        print(f"[WARN] No submission found to copy.")
        return
    save_dir = os.path.join(base_dir, "ensembles_csv/")
    os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(save_dir, "top1ens-total_run_time0.0h.csv")
    shutil.copy(top1_sub, out_file)
    print(f"Copied top1 to: {out_file}")


def _align_submission(
    df: pd.DataFrame, ref_df: pd.DataFrame, id_col_names: List[str]
) -> pd.DataFrame:
    n_ref = len(ref_df)

    if id_col_names and id_col_names[0] in df.columns:
        id_name = id_col_names[0]
        try:
            ref_ids = ref_df[id_name]
            aligned = df.set_index(id_name).reindex(ref_ids.values).reset_index()
            if len(aligned) == n_ref:
                return aligned
        except Exception:
            pass

    # Fallback: positional alignment
    if len(df) >= n_ref:
        return df.iloc[:n_ref].reset_index(drop=True)
    # df shorter than ref: pad with NaN
    return df.reindex(range(n_ref)).reset_index(drop=True)


# ────────────────────── CLI ──────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rule-ensemble")
    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--runs_root", type=str,
                        default="./runs/")
    parser.add_argument("--tag_path", type=str,
                        default="engine/coldstart/competition_tag_classified.json",
                        help="Path to task category JSON")
    parser.add_argument("--use_llm_selection", action="store_true")
    args = parser.parse_args()
    ensemble(args)
