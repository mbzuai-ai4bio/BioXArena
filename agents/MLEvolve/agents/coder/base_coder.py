"""Base code generation mode (single-shot plan + code).

The simplest generation strategy: one LLM call produces a natural language
plan followed by a complete code block. Used as the default / fallback mode
when diff or stepwise generation is not enabled or fails.
"""

from __future__ import annotations

import logging
from typing import Tuple

from llm import generate
from utils.response import extract_code, extract_text_up_to_code

logger = logging.getLogger("MLEvolve")


# ============ Response format prompt (rewrite mode specific) ============

RESPONSE_FORMAT = {
    "Response format": (
        "Your response should be a brief outline/sketch of your proposed solution in natural language, "
        "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
        "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
    )
}


def plan_and_code_query(
    agent_instance,
    prompt,
    retries: int = 3,
) -> Tuple[str, str]:
    """Generate plan + code in one LLM call; returns (nl_text, code). On failure returns ("", raw_completion_text)."""
    completion_text = None
    for _ in range(retries):
        completion_text = generate(
            prompt=prompt,
            temperature=agent_instance.acfg.code.temp,
            cfg=agent_instance.cfg,
        )
        code = extract_code(completion_text)
        nl_text = extract_text_up_to_code(completion_text)

        if code and nl_text:
            return nl_text, code

        logger.debug("Extraction retry...")

    logger.warning("Code extraction failed after retries")
    return "", completion_text  # type: ignore
