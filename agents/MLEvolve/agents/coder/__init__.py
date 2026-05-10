"""Code generation modes package.

Each module implements a different code generation strategy:
- base_coder: Single-shot plan + code generation (simplest / fallback mode)
- stepwise_coder: Multi-agent stepwise generation (data prep -> model -> training)
- diff_coder: Diff-based code generation (SEARCH/REPLACE patch modes)
"""

from .base_coder import plan_and_code_query, RESPONSE_FORMAT  # noqa: F401
from .stepwise_coder import stepwise_plan_and_code_query  # noqa: F401
from .diff_coder import (  # noqa: F401
    SearchReplacePatcher,
    DIFF_SYS_FORMAT,
    build_base_diff_instructions,
    build_diff_format_suffix,
    apply_diff_with_retry,
    format_planning_result_for_plan,
    diff_generate_and_apply,
)
