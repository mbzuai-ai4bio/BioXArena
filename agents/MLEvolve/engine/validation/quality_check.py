"""Submission file quality check and format fix."""

import json
import logging
import re
from pathlib import Path

import pandas as pd

from llm import generate
from config import Config
from engine.validation.format_client import call_validate

logger = logging.getLogger("MLEvolve")


def submission_format_fix_prompt(
    submission_path: Path,
    sample_path: Path | None = None,
    head_rows: int = 20,
) -> str | None:
    """Build prompt for LLM to suggest submission column renames. Returns None if column count differs or names match; does not call LLM or modify files."""
    import numpy as np

    try:
        submission_path = Path(submission_path)
        if not submission_path.exists():
            logger.warning(f"Submission file not found: {submission_path}, skipping format fix")
            return None

        workspace_dir = submission_path.parent.parent
        input_dir = workspace_dir / "input"

        if sample_path is None:
            if not input_dir.exists():
                logger.warning(f"Input directory not found: {input_dir}, skipping format fix")
                return None

            candidates = [
                p for p in input_dir.glob("**/*")
                if p.is_file() and p.suffix.lower() == ".csv" and "sample" in p.name.lower()
            ]
            if not candidates:
                logger.info("No sample submission found under input, skipping format fix")
                return None
            sample_path = sorted(candidates)[0]
        else:
            sample_path = Path(sample_path)

        if not sample_path.exists():
            logger.warning(f"Sample submission file not found: {sample_path}, skipping format fix")
            return None

        sample_df = pd.read_csv(sample_path, dtype=str)
        submission_df = pd.read_csv(submission_path, dtype=str)

        sample_cols = sample_df.columns.tolist()
        submission_cols = submission_df.columns.tolist()

        if all(isinstance(col, (int, np.integer)) for col in sample_cols):
            logger.info("Sample submission CSV has no header, skipping format fix")
            return None
        if all(isinstance(col, (int, np.integer)) for col in submission_cols):
            logger.info("Submission CSV has no header, skipping format fix")
            return None

        if len(sample_cols) != len(submission_cols):
            logger.info(f"Column count mismatch (sample={len(sample_cols)}, submission={len(submission_cols)}), skipping format fix")
            return None

        if sample_cols == submission_cols:
            logger.info("Column names and order already match, no fix needed")
            return None

        def _format_preview(df: pd.DataFrame, title: str) -> str:
            preview_df = df.head(head_rows)
            preview_str = preview_df.to_string(index=False)
            return f"### {title}\n```\n{preview_str}\n```\n"

        prompt_parts = []

        prompt_parts.append(
            "You are a data format fixer. Compare the two CSV files below and determine which sample_submission column name should be assigned to each current submission column."
            " Do not change the data order; you may only rename columns list. You must response with Response Format "
        )

        prompt_parts.append(_format_preview(sample_df, "Sample Submission (standard format, first 20 rows)"))
        prompt_parts.append(_format_preview(submission_df, "Current Submission (to fix, first 20 rows)"))

        prompt_parts.append(
            "## Instructions\n"
            "- The goal is to repair the current submission header so it exactly uses the sample_submission column names.\n"

            f"Sample submission columns (gold order): {sample_cols}\n"
            f"Current submission columns (current order): {submission_cols}\n\n"
            "- Use every sample column name exactly once; the output array length must equal the number of submission columns.\n"
            "- If current column names already convey the same meaning as sample_submission columns (e.g., only differs in case, spacing, underscores, or similar meaning) and the data type / value patterns also align in preview, please directly output the sample column names in their original order.\n"
            "- If names differ substantially, use the previewed values (data type, ranges, patterns) to infer the best match. Because the submission data order must remain unchanged, reorder the sample column names to align with the current submission column data, and output ordered sample column names.\n"
            "\n"
            "## Your OUTPUT Format\n"
            "Your response should contain ONLY a single markdown code block (wrapped in ```) with a JSON array, without any other text.\n"
            "The JSON array format: ```json\n[\"col_name_1\", \"col_name_2\", ...]\n```\n"
            "The i-th element must be the sample column name that should replace the current name of submission column i.\n"
            "\n"
            "Example 1 (names already align): sample submission = [\"id\", \"target\"], current submission = [\"SUBId\", \"Target_value\"].\n"
            "Output:\n"
            "```json\n"
            "[\"id\", \"target\"]\n"
            "```\n"
            "\n"
            "Example 2 (names reordered): sample submission = [\"a1\", \"b1\", \"c\"], current submission = [\"B1\", \"C\", \"A\"].\n"
            "Because you infer that current submission column 1 maps to sample column c, column 2 to a1, and column 3 to b1.\n"
            "Output:\n"
            "```json\n"
            "[\"c\", \"a1\", \"b1\"]\n"
            "```\n"
            "\n"
            "There should be no additional headings or text in your response. MUST the markdown code block. "
        )

        return "\n\n".join(prompt_parts)

    except Exception as e:
        logger.warning(f"Failed to build submission format fix prompt: {e}, skip auto-fix")
        return None


def validate_submission_content_quality(
    submission_path: Path,
    sample_path: Path | None = None,
    constant_threshold: float = 0.95,
) -> tuple[bool, str]:
    """Local check for submission content quality (placeholder/constant filling). Returns (is_valid, error_message)."""
    import numpy as np
    try:
        submission_path = Path(submission_path)
        if not submission_path.exists():
            return False, f"Submission file not found: {submission_path}"
        if sample_path is None:
            workspace_dir = submission_path.parent.parent
            input_dir = workspace_dir / "input"
            if input_dir.exists():
                candidates = [
                    p for p in input_dir.glob("**/*")
                    if p.is_file() and p.suffix.lower() == ".csv" and "sample" in p.name.lower()
                ]
                if candidates:
                    sample_path = sorted(candidates)[0]
        df_sub = pd.read_csv(submission_path, dtype=str, keep_default_na=False)
        if sample_path is not None and Path(sample_path).exists():
            df_sample = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
            target_cols = df_sample.columns[1:]
        else:
            logger.info("No sample submission found, checking last column for constant values")
            if len(df_sub.columns) >= 2:
                target_cols = [df_sub.columns[-1]]
            else:
                logger.warning("Submission has only 1 column, skipping content quality check")
                return True, ""
        for col in target_cols:
            if col not in df_sub.columns:
                continue
            values = df_sub[col].astype(str)
            # Skip all checks for binary classification columns (both 0 and 1 present)
            unique_vals = set(values.unique())
            if unique_vals == {"0", "1"}:
                logger.info(f"Column '{col}' is binary (0/1) — skipping quality check")
                continue
            empty_mask = values.apply(lambda x: len(str(x).strip()) == 0)
            empty_ratio = empty_mask.sum() / len(values) if len(values) > 0 else 0
            if empty_ratio > constant_threshold:
                return False, (
                    f"Column '{col}' contains {empty_ratio*100:.1f}% empty values (threshold: {constant_threshold*100}%). "
                    f"The agent must generate real model predictions for all test samples, not leave them empty."
                )
            if len(values) > 2:
                value_counts = values.value_counts()
                if len(value_counts) > 0:
                    most_common_ratio = value_counts.iloc[0] / len(values)
                    if most_common_ratio > constant_threshold:
                        most_common_value = value_counts.index[0]
                        return False, (
                            f"Column '{col}': {most_common_ratio*100:.1f}% of values are '{most_common_value}'. "
                            f"This indicates the agent filled the submission with constants instead of running real model inference on each test sample. "
                            f"This is a serious lazy behavior that violates model inference integrity requirements."
                        )
            # Check 3: Common placeholder patterns
            placeholder_patterns = {
                r'^0+$': 'all zeros',
                r'^1+$': 'all ones',
                r'^0(\s+0)+$': 'space-separated zeros',
                r'^1(\s+1)+$': 'space-separated ones',
                r'^\d+(\s+[01]\.0+)+(\s+[01])+$': 'placeholder like "14 1.0 0 0 1 1"',
            }
            for pattern, pattern_name in placeholder_patterns.items():
                matches = values.str.match(pattern, na=False)
                match_ratio = matches.sum() / len(values) if len(values) > 0 else 0
                if match_ratio > 0.5:
                    sample_values = values[matches].head(3).tolist()
                    return False, (
                        f"Column '{col}': {match_ratio*100:.1f}% of values match placeholder pattern: {pattern_name}. "
                        f"Sample values: {sample_values}. "
                        f"The agent used placeholders instead of real model predictions."
                    )
        return True, ""
    except Exception as e:
        logger.warning(f"Content quality check encountered error: {e}, skipping check")
        return True, ""


def _validate_submission_with_retry(
    exp_id: str,
    submission_path: Path,
    cfg: Config,
    max_attempts: int = 2,
    sample_path: Path | None = None,
) -> tuple[bool, dict]:
    """Validate submission with retries; optionally try format fix between attempts."""
    if not getattr(cfg, "use_grading_server", True):
        logger.info("Grading server disabled (use_grading_server=False); skipping format validation.")
        return True, {"is_valid": True, "result": "Skipped (no grading server)."}

    status = False
    res = None

    for attempt in range(max_attempts):
        status, res = call_validate(exp_id=exp_id, submission_path=submission_path)
        if not status:
            return status, res

        if res and res.get("is_valid", False):
            return status, res

        logger.warning(
            f"Submission validation failed on attempt {attempt + 1}/{max_attempts} for file {submission_path}."
        )

        if attempt == max_attempts - 1:
            break

        fix_success = try_fix_submission_format(
            submission_path=submission_path,
            cfg=cfg,
            sample_path=sample_path,
            head_rows=20,
        )

        if not fix_success:
            logger.info("Auto-format fix not performed or failed. Stopping further retries.")
            break

        logger.info("Auto-format fix applied. Retrying submission validation.")

    return status, res


def try_fix_submission_format(
    submission_path: Path,
    cfg: Config,
    sample_path: Path | None = None,
    head_rows: int = 20,
) -> bool:
    """Attempt to fix submission.csv header using LLM suggestions."""

    try:
        submission_path = Path(submission_path)
        if not submission_path.exists():
            logger.warning(f"Submission file not found: {submission_path}")
            return False

        suggested_columns = llm_suggest_submission_columns(
            submission_path=submission_path,
            sample_path=sample_path,
            head_rows=head_rows,
            cfg=cfg,
        )

        if not suggested_columns:
            logger.info("LLM did not return a valid column mapping suggestion")
            return False

        df = pd.read_csv(submission_path, dtype=str)

        if len(df.columns) != len(suggested_columns):
            logger.warning(
                "Suggested columns length %s does not match current submission columns %s",
                len(suggested_columns),
                len(df.columns),
            )
            return False

        df.columns = suggested_columns
        df.to_csv(submission_path, index=False)
        logger.info("Submission columns updated via LLM suggestion")
        return True

    except Exception as exc:
        logger.warning(f"Failed to apply LLM-based submission fix: {exc}")
        return False


def _extract_json_array(text: str) -> list[str] | None:
    """Extract and parse the first JSON array from text."""

    code_block_patterns = [
        r"```json\s*(\[[\s\S]*?\])\s*```",  # ```json [...] ```
        r"```\s*(\[[\s\S]*?\])\s*```",      # ``` [...] ```
    ]

    for pattern in code_block_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                    return parsed
            except json.JSONDecodeError:
                continue

    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('```'):
            continue
        if line.startswith('[') and line.endswith(']'):
            try:
                parsed = json.loads(line)
                if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                    return parsed
            except json.JSONDecodeError:
                continue

    array_pattern = r"\[(?:[^\[\]]+|\[[^\]]*\])*?\]"
    matches = re.finditer(array_pattern, text)
    for match in matches:
        json_str = match.group(0)
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                return parsed
        except json.JSONDecodeError:
            continue

    return None


def llm_suggest_submission_columns(
    submission_path: Path,
    cfg: Config,
    sample_path: Path | None = None,
    head_rows: int = 20,
    temperature: float = 0.0,
    max_tokens: int = 20000,
) -> list[str] | None:
    """Use LLM to suggest corrected submission column names."""

    prompt = submission_format_fix_prompt(
        submission_path=submission_path,
        sample_path=sample_path,
        head_rows=head_rows,
    )

    if not prompt:
        return None

    try:
        response = generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            cfg=cfg,
        )
    except Exception as exc:
        logger.warning(f"LLM call failed during submission column fix: {exc}")
        return None

    suggested_columns = _extract_json_array(response)
    if suggested_columns is None:
        logger.warning("LLM response did not contain a valid JSON array of column names")
        return None

    return suggested_columns
