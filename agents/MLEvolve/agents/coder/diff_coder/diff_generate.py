"""Unified diff generation and application pipeline.

All diff-based agents (improve / evolution / fusion / multi_fusion)
call ``diff_generate_and_apply()`` with stage-specific text parameters.
Prompt construction, LLM call, and patch application are fully shared.

Each agent provides only:
- ``introduction``       : system-level intro describing the stage
- ``extra_context``      : additional assistant context (trajectory, references, etc.)
- ``extra_user_sections``: additional user prompt sections
- ``learning_guidance``  : extra instruction for diff guidelines
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Tuple, Union

from llm import generate
from .prompts import build_base_diff_instructions, build_diff_format_suffix, DIFF_SYS_FORMAT
from .apply import apply_diff_with_retry, format_planning_result_for_plan

logger = logging.getLogger("MLEvolve")


# ============ Public API ============

def diff_generate_and_apply(
    agent_instance,
    planning_result: Dict[str, Any],
    parent_code: str,
    data_preview: str,
    execution_output: str,
    introduction: str,
    extra_context: str = "",
    extra_user_sections: str = "",
    learning_guidance: str = "",
    max_diff_retries: int = 3,
) -> Tuple[str, str]:
    model_name = agent_instance.acfg.code.model

    plan_text = _format_plan_text(planning_result)

    base_instructions = build_base_diff_instructions(learning_guidance)
    diff_format = build_diff_format_suffix()
    diff_instructions = (
        f"{base_instructions}\n\n"
        f"{diff_format}\n\n"
        f"Response format: {DIFF_SYS_FORMAT}"
    )

    user_prompt_parts = [f"\n# Improvement Plan\n\n{plan_text}\n"]
    if extra_user_sections:
        user_prompt_parts.append(f"\n{extra_user_sections}\n")
    user_prompt_parts.append(f"\n{diff_instructions}\n")
    user_prompt = "".join(user_prompt_parts)

    assistant_text = _build_assistant_text(
        data_preview, parent_code, execution_output, extra_context,
    )

    diff_prompt = _build_diff_model_prompt(model_name, introduction, user_prompt, assistant_text)

    logger.info("Calling LLM to generate diff...")
    diff_response = generate(
        prompt=diff_prompt,
        temperature=agent_instance.acfg.code.temp,
        cfg=agent_instance.cfg,
    )

    def regenerate_fn(current_code: str, retry_note: str) -> str:
        new_assistant = _build_assistant_text(
            data_preview, current_code, execution_output, extra_context,
        )
        new_prompt = _build_diff_model_prompt(
            model_name, introduction, user_prompt, new_assistant,
        )
        if retry_note:
            if isinstance(new_prompt, dict) and "user" in new_prompt:
                new_prompt["user"] = retry_note + "\n\n" + new_prompt["user"]
            elif isinstance(new_prompt, str):
                new_prompt = retry_note + "\n\n" + new_prompt
        return generate(
            prompt=new_prompt,
            temperature=agent_instance.acfg.code.temp,
            cfg=agent_instance.cfg,
        )

    final_code, total_applied, _ = apply_diff_with_retry(
        diff_response, parent_code,
        max_retries=max_diff_retries,
        regenerate_fn=regenerate_fn,
    )

    plan_str = format_planning_result_for_plan(planning_result)

    if final_code:
        logger.info(f"Diff completed: applied {total_applied} patch(es)")
        return plan_str, final_code
    else:
        logger.warning("All diff attempts failed, returning original code")
        return plan_str, parent_code


# ============ Internal helpers ============

def _format_plan_text(planning_result: Dict[str, Any]) -> str:
    modules_to_modify = planning_result.get("module", [])
    plans = planning_result.get("plan", {})

    text = f"## Improvement Analysis\n\n{planning_result.get('reason', 'Improve the solution')}\n\n"

    if modules_to_modify:
        text += f"## Target Components: {', '.join(modules_to_modify)}\n\n"
        if plans and len(plans) > 0:
            text += "## Detailed Enhancement Plans\n\n"
            for module_name in modules_to_modify:
                plan_detail = plans.get(module_name, "No plan provided")
                text += f"**{module_name}:**\n{plan_detail}\n\n"
        else:
            text += (
                f"## Improvement Guidance\n\n"
                f"Based on the analysis above, implement targeted improvements "
                f"to the following components: {', '.join(modules_to_modify)}\n"
                f"Focus on enhancing test set performance while preserving "
                f"existing functionality and interfaces.\n\n"
            )
    return text


def _build_assistant_text(
    data_preview: str,
    code: str,
    execution_output: str,
    extra_context: str = "",
) -> str:
    wrapped_code = f"```python\n{code}\n```"
    wrapped_output = f"```\n{execution_output or '[No execution output provided]'}\n```"

    parts = [
        "Let me approach this systematically.",
        f"First, I'll examine the dataset:\n{data_preview}",
        f"According to the improvement plan provided above, I will implement the specified enhancements "
        f"to improve test set performance.",
        f"Regarding this task, I previously made attempts with the following code:\n{wrapped_code}",
        f"The execution of this code yielded the following results:\n{wrapped_output}",
    ]

    if extra_context:
        parts.append(extra_context)

    parts.append("I will now implement the improvements according to the plan.")

    return "\n".join(parts)


def _build_diff_model_prompt(
    model_name: str,
    introduction: str,
    user_prompt: str,
    assistant_text: str,
) -> Union[str, Dict[str, str]]:
    if (model_name or "").lower().startswith("gemini"):
        return f"{introduction}\n\n{user_prompt}\n\n{assistant_text}"
    return {
        "system": introduction,
        "user": user_prompt,
        "assistant": assistant_text,
    }
