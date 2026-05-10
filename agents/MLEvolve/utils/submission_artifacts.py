"""Helpers for materializing submission CSV sidecar artifacts.

Some tasks store predictions in local files or directories referenced by
`submission.csv` instead of placing the values directly in the CSV. These
helpers snapshot those artifacts per node and export self-contained bundles for
best/top submissions and final task outputs.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

logger = logging.getLogger("MLEvolve")

_PATH_COLUMN_SUFFIXES = ("_file", "_path", "_dir")


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def _normalize_raw_path(value: object) -> str:
    return str(value).strip().replace("\\", "/")


def _looks_like_artifact_reference(value: object, column_name: str) -> bool:
    if _is_blank(value):
        return False

    raw = _normalize_raw_path(value)
    candidate = Path(raw)
    if candidate.is_absolute():
        return True

    if "/" in raw:
        return True

    column_name = column_name.lower()
    if column_name.endswith(_PATH_COLUMN_SUFFIXES):
        return True

    return False


def _resolve_existing_artifact_path(
    value: object,
    column_name: str,
    search_roots: tuple[Path, ...],
) -> Path | None:
    if not _looks_like_artifact_reference(value, column_name):
        return None

    raw = _normalize_raw_path(value)
    candidate = Path(raw)

    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    for root in search_roots:
        resolved = root / candidate
        if resolved.exists():
            return resolved

    return None


def _split_name_and_suffixes(name: str) -> tuple[str, str]:
    suffixes = "".join(Path(name).suffixes)
    stem = name[: -len(suffixes)] if suffixes else name
    return stem, suffixes


def _inject_node_suffix(relative_path: Path, node_id: str) -> Path:
    suffix = f"_{node_id}"
    parts = list(relative_path.parts)
    if not parts:
        return Path(f"artifacts{suffix}")

    if len(parts) == 1:
        stem, file_suffixes = _split_name_and_suffixes(parts[0])
        if stem.endswith(suffix):
            return relative_path
        return Path(f"{stem}{suffix}{file_suffixes}")

    if parts[0].endswith(suffix):
        return relative_path

    parts[0] = f"{parts[0]}{suffix}"
    return Path(*parts)


def _strip_node_suffix(relative_path: Path, node_id: str) -> Path:
    suffix = f"_{node_id}"
    parts = list(relative_path.parts)
    if not parts:
        return relative_path

    if len(parts) == 1:
        stem, file_suffixes = _split_name_and_suffixes(parts[0])
        if not stem.endswith(suffix):
            return relative_path
        return Path(f"{stem[:-len(suffix)]}{file_suffixes}")

    if parts[0].endswith(suffix):
        parts[0] = parts[0][:-len(suffix)]

    return Path(*parts)


def _absolute_artifact_snapshot_path(source_path: Path, node_id: str) -> Path:
    parent_name = source_path.parent.name or "root"
    return Path(f"artifacts_{node_id}") / parent_name / source_path.name


def _value_to_relative_path(value: object, source_path: Path, node_id: str) -> Path:
    raw = _normalize_raw_path(value)
    candidate = Path(raw)
    if candidate.is_absolute():
        return _absolute_artifact_snapshot_path(source_path, node_id)
    return _inject_node_suffix(candidate, node_id)


def _value_to_export_path(value: object, source_path: Path, node_id: str | None) -> Path:
    raw = _normalize_raw_path(value)
    candidate = Path(raw)
    if candidate.is_absolute():
        return Path("artifacts") / (source_path.parent.name or "root") / source_path.name
    if node_id is None:
        return candidate
    return _strip_node_suffix(candidate, node_id)


def _copy_path(source_path: Path, target_path: Path) -> None:
    if source_path.is_dir():
        if target_path.exists():
            shutil.rmtree(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_path, target_path)
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def materialize_node_submission_artifacts(
    submission_csv: Path,
    workspace_dir: Path,
    node_id: str | int,
) -> None:
    """Rewrite a node submission CSV to point to node-scoped artifact copies."""
    submission_csv = Path(submission_csv)
    workspace_dir = Path(workspace_dir)
    node_id = str(node_id)

    if not submission_csv.exists():
        return

    df = pd.read_csv(submission_csv, dtype=str, keep_default_na=False)
    search_roots = (submission_csv.parent, workspace_dir)
    dirty = False
    copied_targets: set[Path] = set()

    for row_idx in range(len(df)):
        for column_name in df.columns[1:]:
            raw_value = df.at[row_idx, column_name]
            source_path = _resolve_existing_artifact_path(raw_value, column_name, search_roots)
            if source_path is None:
                continue

            target_rel = _value_to_relative_path(raw_value, source_path, node_id)
            target_abs = submission_csv.parent / target_rel

            if source_path.resolve() != target_abs.resolve() and target_abs not in copied_targets:
                _copy_path(source_path, target_abs)
                copied_targets.add(target_abs)

            rewritten_value = target_rel.as_posix()
            if raw_value != rewritten_value:
                df.at[row_idx, column_name] = rewritten_value
                dirty = True

    if dirty:
        df.to_csv(submission_csv, index=False)
        logger.info("Materialized submission sidecar artifacts for node %s", node_id)


def export_submission_bundle(
    source_submission_csv: Path,
    destination_dir: Path,
    submission_filename: str = "submission.csv",
    node_id: str | int | None = None,
) -> None:
    """Export a self-contained submission bundle into `destination_dir`."""
    source_submission_csv = Path(source_submission_csv)
    destination_dir = Path(destination_dir)
    node_id = None if node_id is None else str(node_id)

    df = pd.read_csv(source_submission_csv, dtype=str, keep_default_na=False)
    search_roots = (
        source_submission_csv.parent,
        source_submission_csv.parent.parent,
    )
    copied_targets: set[Path] = set()

    for row_idx in range(len(df)):
        for column_name in df.columns[1:]:
            raw_value = df.at[row_idx, column_name]
            source_path = _resolve_existing_artifact_path(raw_value, column_name, search_roots)
            if source_path is None:
                continue

            target_rel = _value_to_export_path(raw_value, source_path, node_id)
            target_abs = destination_dir / target_rel

            if target_abs not in copied_targets:
                _copy_path(source_path, target_abs)
                copied_targets.add(target_abs)

            df.at[row_idx, column_name] = target_rel.as_posix()

    destination_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination_dir / submission_filename, index=False)
