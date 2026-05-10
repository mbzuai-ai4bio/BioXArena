"""Prompt templates for diff-based code generation modes.

All prompt text that instructs LLMs on the SEARCH/REPLACE diff format
and diff-mode-specific requirements.
"""

from __future__ import annotations


# ============ SEARCH/REPLACE format template ============
# Tells the LLM what output format to use

DIFF_SYS_FORMAT = """
***You are required to explain your plan in detail and clearly indicate the code improvements using the exact SEARCH/REPLACE diff format specified below (ensure the format is strictly followed without deviations).***

# Code format:
<<<<<<< SEARCH
# Exact original code snippet to replace (must match exactly, including indentation)
=======
# New replacement code snippet
>>>>>>> REPLACE

Example of a valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

* Every block's SEARCH section must be copied **verbatim** from the current file. Including indentation.
  - **CRITICAL**: The SEARCH pattern must match the original code EXACTLY, including:
    * Exact number of leading spaces (do not reduce or increase indentation)
    * Exact whitespace characters (spaces vs tabs)
    * Exact line breaks and empty lines
  - **Common mistake**: Using 2 spaces when the original code uses 4 spaces (or vice versa) will cause the diff to fail.
  - **How to avoid**: Copy the code directly from the provided code block above, do not retype it.
* **CRITICAL**: Every SEARCH must have exactly one REPLACE (paired in the same block). Do not write a SEARCH without its corresponding REPLACE.
* You can propose multiple independent edits. SEARCH/REPLACE blocks follow one after another. DO NOT ADD ANY OTHER TEXT BETWEEN THESE BLOCKS.
* Make sure the file still runs after your changes.
* Please refrain from modifying the entire file; only revise the required sections."""


# ============ Diff mode instruction builders ============

def build_base_diff_instructions(learning_guidance: str = "") -> str:
    learning_line = f"    - {learning_guidance}\n" if learning_guidance else ""

    base_instructions = f"""    # Diff Mode Instructions

    ## Plan Execution Requirements

    - **You must implement the improvement plan provided above.** Only modify the components and aspects specified in the plan.
    {learning_line}    - **Preserve all interfaces, variable names, and compatibility constraints** as mentioned in the plan.
    - **Make minimal, targeted changes** - focus only on the enhancements outlined in the plan.

    ## CRITICAL: SEARCH Pattern Requirements

    **When creating SEARCH patterns, you MUST:**
    1. **Copy the EXACT code from the provided code above, including:**
       - Exact indentation (spaces/tabs) - count spaces carefully!
       - Exact whitespace (leading/trailing spaces)
       - Exact line breaks
       - Exact comments and formatting

    2. **DO NOT modify the indentation in SEARCH patterns** - they must match the original code character-by-character.

    3. **To ensure accuracy:**
       - Copy the code block directly from the provided code (do not retype it)
       - Count spaces carefully if needed (Python uses 4 spaces per indentation level typically)
       - Verify the SEARCH pattern matches exactly before creating the REPLACE section
       - If the code is inside a function, preserve the function's indentation level

    4. **Common mistakes to avoid:**
       - Do not reduce indentation (e.g., using 2 spaces when original has 4)
       - Do not add extra indentation
       - Do not normalize whitespace - preserve exactly as shown

    - **You must use the SEARCH/REPLACE diff format specified below to modify the code.** Follow the format requirements exactly.
    """
    return base_instructions


def build_diff_format_suffix() -> str:
    suffix = (
        "\n\n🔴 **CRITICAL REQUIREMENT**: You MUST provide code modifications using SEARCH/REPLACE format. "
        "Do NOT return unchanged code or full code blocks. "
        "Keep each SEARCH snippet minimal (ONLY the lines to be replaced, plus tiny context if needed). "
        "Prefer multiple small SEARCH/REPLACE blocks instead of one large block.\n"
        "You MUST use the exact SEARCH/REPLACE diff format specified below.\n"
        "⚠️ **Your response will be automatically rejected if it does not contain SEARCH/REPLACE format markers.**\n\n"
        f"Response format: {DIFF_SYS_FORMAT}"
        "\n\n🔴 **FINAL REMINDER**: Your response MUST start with a SEARCH/REPLACE block. "
        "Do NOT output full code or explanations without diff format."
    )
    return suffix
