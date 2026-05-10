"""SEARCH/REPLACE patch engine.

Pure utility class for applying SEARCH/REPLACE diff patches to source code.
No business logic, no prompts, no logging dependencies.
"""

from __future__ import annotations

import re
import difflib
from typing import List, Tuple, Optional


class SearchReplacePatcher:
    """Apply SEARCH/REPLACE diff patches to source code."""

    class PatchError(RuntimeError):
        """Custom exception raised when a SEARCH pattern cannot be applied."""
        pass

    PATCH_PATTERN = re.compile(
        r"<{7}\s*SEARCH\s*\n(.*?)\n\s*={7}\s*\n(.*?)\n\s*>{7}\s*REPLACE\s*",
        re.DOTALL,
    )

    @staticmethod
    def _strip_trailing_whitespace(text: str) -> str:
        return "\n".join(line.rstrip() for line in text.splitlines())

    @staticmethod
    def _find_indented_match(search_text: str, original_text: str) -> Tuple[str, int]:
        if not search_text.strip():
            return "", -1

        # exact match first
        pos = original_text.find(search_text)
        if pos != -1:
            return search_text, pos

        search_lines = search_text.splitlines()
        first_search_line = search_lines[0].strip()
        original_lines = original_text.splitlines()
        for i, line in enumerate(original_lines):
            if line.strip() == first_search_line:
                line_indent = len(line) - len(line.lstrip())
                indent_str = line[:line_indent]

                indented_search_lines = []
                for j, search_line in enumerate(search_lines):
                    if j == 0:
                        indented_search_lines.append(indent_str + search_line.strip())
                    else:
                        search_line_indent = len(search_line) - len(search_line.lstrip())
                        if search_line.strip():
                            indented_search_lines.append(
                                indent_str + " " * search_line_indent + search_line.strip()
                            )
                        else:
                            indented_search_lines.append("")
                indented_search = "\n".join(indented_search_lines)
                indented_pos = original_text.find(indented_search)
                if indented_pos != -1:
                    return indented_search, indented_pos
        return "", -1

    @staticmethod
    def _apply_indentation_to_replace(replace_text: str, indent_str: str) -> str:
        if not replace_text.strip():
            return replace_text
        replace_lines = replace_text.splitlines()
        indented_replace_lines = []
        for line in replace_lines:
            if line.strip():
                line_indent = len(line) - len(line.lstrip())
                indented_replace_lines.append(indent_str + " " * line_indent + line.strip())
            else:
                indented_replace_lines.append("")
        return "\n".join(indented_replace_lines)

    @staticmethod
    def _find_best_match_with_diff(search_text: str, original_text: str) -> Optional[Tuple[List[str], int, List[str]]]:
        search_lines = search_text.strip().splitlines()
        if not search_lines:
            return None
        original_lines = original_text.splitlines()
        search_len = len(search_lines)
        best_match = None
        best_ratio = 0.0
        best_start_line = 0
        for i in range(max(0, len(original_lines) - search_len + 1)):
            candidate_lines = original_lines[i : i + search_len]
            candidate_text = "\n".join(candidate_lines)
            search_block = "\n".join(search_lines)
            ratio = difflib.SequenceMatcher(None, search_block, candidate_text).ratio()
            if ratio > best_ratio and ratio > 0.6:
                best_ratio = ratio
                best_match = candidate_lines
                best_start_line = i + 1
        if best_match is None:
            return None
        search_prefixed = [f"  {l}" for l in search_lines]
        match_prefixed = [f"  {l}" for l in best_match]
        diff_lines = list(
            difflib.unified_diff(
                search_prefixed,
                match_prefixed,
                fromfile="Search Pattern",
                tofile=f"Actual Code (line {best_start_line})",
                lineterm="",
                n=0,
            )
        )
        clean_diff = [ln for ln in diff_lines if not (ln.startswith("---") or ln.startswith("+++") or ln.startswith("@@"))]
        return best_match, best_start_line, clean_diff

    def apply_patch(self, patch_text: str, original_text: str, strict: bool = True) -> Tuple[str, int]:
        """Apply SEARCH/REPLACE blocks to original_text."""
        new_text = original_text
        num_applied = 0
        patch_text = self._strip_trailing_whitespace(patch_text)

        for block in self.PATCH_PATTERN.finditer(patch_text):
            search, replace = block.group(1), block.group(2)
            search = self._strip_trailing_whitespace(search)
            replace = self._strip_trailing_whitespace(replace)

            # Insertion if empty SEARCH
            if not search.strip():
                new_text = new_text.rstrip() + "\n\n" + replace + "\n"
                num_applied += 1
                continue

            matched_search, pos = self._find_indented_match(search, new_text)
            if pos == -1:
                if strict:
                    best = self._find_best_match_with_diff(search, new_text)
                    if best:
                        _, start_line, diff_lines = best
                        msg = [
                            "SEARCH pattern not found. Closest match at line " + str(start_line) + ":",
                            *diff_lines,
                        ]
                        raise self.PatchError("\n".join(msg))
                    else:
                        raise self.PatchError("SEARCH pattern not found (no close match).")
                else:
                    continue

            if matched_search != search:
                matched_lines = matched_search.splitlines()
                if matched_lines:
                    first_matched_line = matched_lines[0]
                    indent_len = len(first_matched_line) - len(first_matched_line.lstrip())
                    indent_str = first_matched_line[:indent_len]
                    replace = self._apply_indentation_to_replace(replace, indent_str)

            new_text = new_text.replace(matched_search, replace, 1)
            num_applied += 1
        return new_text, num_applied
