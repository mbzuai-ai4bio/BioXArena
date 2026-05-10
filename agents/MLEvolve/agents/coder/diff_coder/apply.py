"""Diff patch application with retry logic.

Provides the shared apply-and-retry pipeline used by all diff-based
code generation modes (debug / improve / evolution / fusion).
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Any, Tuple, Optional

from .patcher import SearchReplacePatcher

logger = logging.getLogger("MLEvolve")


def apply_diff_with_retry(
    diff_response: str,
    original_code: str,
    max_retries: int = 3,
    regenerate_fn=None,
) -> Tuple[Optional[str], int, str]:
    current_code = original_code
    total_applied = 0
    retry_note = ""
    current_response = diff_response

    for attempt in range(max_retries):
        try:
            logger.info(f"Applying diff patches... (attempt {attempt + 1}/{max_retries})")

            if current_response and (
                "<<<<<<< SEARCH" in current_response
                or "< SEARCH" in current_response
                or "<<<<<<<" in current_response
            ):
                if "<<<<<<< SEARCH" in current_response:
                    search_markers = current_response.count("<<<<<<< SEARCH")
                    replace_markers = current_response.count(">>>>>>> REPLACE")
                elif "< SEARCH" in current_response:
                    search_markers = current_response.count("< SEARCH")
                    replace_markers = current_response.count("> REPLACE")
                else:
                    search_markers = 1
                    replace_markers = 0
                has_incomplete_block = search_markers > replace_markers

                patcher = SearchReplacePatcher()
                updated_code, count = patcher.apply_patch(current_response, current_code, strict=False)
                if count > 0 and updated_code and updated_code != current_code:
                    current_code = updated_code
                    total_applied += count

                if total_applied > 0 and current_code != original_code and not has_incomplete_block:
                    logger.info(f"Successfully applied {total_applied} diff patch(es)")
                    return current_code, total_applied, ""
                else:
                    if has_incomplete_block and (count > 0 or total_applied > 0):
                        retry_note = (
                            "Your previous diff output appears truncated/incomplete "
                            "(missing closing '>>>>>>> REPLACE'). "
                            f"I have already applied {total_applied} patch(es). "
                            "Please continue and provide ONLY the remaining patches."
                        )
                    else:
                        retry_note = (
                            "Your previous diff did not apply cleanly to the current code. "
                            "Please generate minimal SEARCH/REPLACE blocks that match the CURRENT code exactly."
                        )

                    logger.warning(
                        f"Diff attempt {attempt + 1}/{max_retries}: "
                        f"count={count}, total_applied={total_applied}, "
                        f"code_changed={current_code != original_code}, "
                        f"search_markers={search_markers}, replace_markers={replace_markers}, "
                        f"has_incomplete_block={has_incomplete_block}"
                    )

                    if attempt < max_retries - 1 and regenerate_fn:
                        logger.info("Regenerating diff...")
                        current_response = regenerate_fn(current_code, retry_note)
                        continue
                    else:
                        if total_applied > 0:
                            return current_code, total_applied, retry_note
                        return None, 0, retry_note
            else:
                retry_note = (
                    "Your previous output did not contain valid SEARCH/REPLACE blocks. "
                    "Output ONLY complete SEARCH/REPLACE blocks (no other text)."
                )
                logger.warning(
                    f"Diff attempt {attempt + 1}/{max_retries}: "
                    "Response does not contain SEARCH/REPLACE format"
                )

                if attempt < max_retries - 1 and regenerate_fn:
                    logger.info("Regenerating diff...")
                    current_response = regenerate_fn(current_code, retry_note)
                    continue
                else:
                    return None, 0, retry_note

        except Exception as e:
            logger.warning(f"Diff attempt {attempt + 1}/{max_retries} failed with exception: {e}")
            retry_note = (
                f"Your previous diff failed to apply due to an error: {e}. "
                "Please output minimal SEARCH/REPLACE blocks that match the CURRENT code exactly, "
                "and ensure every block is complete."
            )
            if attempt < max_retries - 1 and regenerate_fn:
                logger.info("Regenerating diff...")
                try:
                    current_response = regenerate_fn(current_code, retry_note)
                except Exception as retry_e:
                    logger.error(f"Failed to regenerate diff: {retry_e}")
                continue
            else:
                if total_applied > 0:
                    return current_code, total_applied, retry_note
                return None, 0, retry_note

    return None, 0, retry_note


def format_planning_result_for_plan(planning_result: Dict[str, Any]) -> str:
    """Serialize planning result for node plan storage."""
    try:
        return json.dumps(planning_result, ensure_ascii=True)
    except (TypeError, ValueError):
        return str(planning_result)
