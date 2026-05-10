"""Diff-based code generation package.

Provides shared utilities for all diff modes (debug / improve / evolution / fusion):
- patcher: SEARCH/REPLACE patch engine (pure utility)
- prompts: Prompt templates and instruction builders
- apply: Diff patch application with retry logic
- diff_generate: Unified generate + apply pipeline for all agents
"""

from .patcher import SearchReplacePatcher  # noqa: F401
from .prompts import (  # noqa: F401
    DIFF_SYS_FORMAT,
    build_base_diff_instructions,
    build_diff_format_suffix,
)
from .apply import (  # noqa: F401
    apply_diff_with_retry,
    format_planning_result_for_plan,
)
from .diff_generate import diff_generate_and_apply  # noqa: F401
