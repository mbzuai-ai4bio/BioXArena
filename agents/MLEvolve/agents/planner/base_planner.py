"""Generic planning agent for module selection.

Provides the shared planner pipeline used by all diff-based agents
(improve / evolution / fusion / multi_fusion). Each agent provides
stage-specific text (your_task_section, assistant_suffix) and calls
run_planner() to get a structured planning result.

Main entry: run_planner()
"""

from __future__ import annotations

import re
import json
import logging
from typing import Dict, Any, Union

from llm import generate, compile_prompt_to_md
from llm.model_profiles import thinking_json_incompatible

logger = logging.getLogger("MLEvolve")


# ============ Planning constants ============

PLANNING_ALLOWED_MODULES = [
    "data_processing_and_feature_engineering",
    "model_design",
    "training_evaluation",
]

PLANNING_JSON_FORMAT = f"""{{
  "reason": "The reason why you chose these components to modify. Based on the current status and execution results: 1) why these components are the most promising ones to focus on, 2) why these components are the right places to apply your modifications.",
  "module": ["module_name_from_allowed_list"],
  // CRITICAL: "module" must be an array (1-3 elements). Each element must be from: [{', '.join([f"'{m}'" for m in PLANNING_ALLOWED_MODULES])}]
  // Choose 1-3 modules based on actual analysis, not from this example
  "plan": {{
    "module_name_from_allowed_list": "Detailed modification plan for this module: 1) what to change, 2) how to change it, 3) expected results..."
  }}
  // CRITICAL: Keys in "plan" MUST be exactly the same as module names in the "module" array
}}"""

PLANNING_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "description": "The reason why you chose these components to modify. Based on the current status and execution results: 1) why these components are the most promising ones to focus on, 2) why these components are the right places to apply your modifications. Should be 1-4 sentences including root cause analysis."
        },
        "module": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": PLANNING_ALLOWED_MODULES
            },
            "description": f"An array of 1-3 module names to modify. Each element MUST be one of: {', '.join(PLANNING_ALLOWED_MODULES)}. Choose 1-3 modules based on actual analysis, not from examples.",
            "minItems": 1,
            "maxItems": 3
        },
        "plan": {
            "type": "object",
            "title": "Module Modification Plans",
            "description": "A JSON object where keys are module names (must match exactly with names in the 'module' array) and values are detailed modification plan strings. CRITICAL REQUIREMENT: For EVERY module name listed in the 'module' array, you MUST create a corresponding key-value pair in this 'plan' object. The number of keys in 'plan' MUST equal the number of elements in 'module'. Each value must be a non-empty string containing the modification plan.",
            "additionalProperties": {
                "type": "string",
                "title": "Modification Plan",
                "description": "A detailed modification plan (2-5 sentences) as a string. Must include: 1) WHAT to change (specific technical modification with category), 2) WHY this change (root cause analysis, why THIS TASK needs it), 3) HOW to implement (specific implementation approach), 4) Interface constraints (variable names, function signatures to preserve). Maintain existing variable names to ensure compatibility with other components. This string must not be empty."
            },
            "required": []
        }
    },
    "required": ["reason", "module", "plan"],
    "additionalProperties": False
}


# ============ Component descriptions ============

def get_component_descriptions() -> Dict[str, str]:
    """Get module name → description mapping from StepAgent definitions."""
    from agents.coder.stepwise_coder import create_default_step_agents  # lazy to avoid circular import
    step_agents = create_default_step_agents()
    return {agent.name: agent.description for agent in step_agents}


# ============ Model-specific prompt formatting ============

def build_model_prompt(
    model_name: str,
    introduction: str,
    user_prompt: str,
    assistant_suffix: str,
):
    """Build prompt for model. Gemini: plain text. Qwen/OpenAI: chat dict {system, user, assistant}."""
    if (model_name or "").lower().startswith("gemini"):
        return f"{introduction}\n\n{user_prompt}\n\n{assistant_suffix}"
    return {
        "system": introduction,
        "user": user_prompt,
        "assistant": assistant_suffix,
    }


def build_chat_prompt_for_model(
    model_name: str,
    introduction: str,
    user_prompt: str,
    assistant_prefix: str,
):
    """Build prompt for agents (draft/improve/fusion/etc). Gemini: plain text. Qwen/OpenAI: chat dict."""
    if (model_name or "").lower().startswith("gemini"):
        return f"{introduction}\n\n{user_prompt}\n\n{assistant_prefix}"
    return {
        "system": introduction,
        "user": user_prompt,
        "assistant": assistant_prefix,
    }


# ============ Response parser ============

def parse_planning_response(response: Union[str, dict]) -> Dict[str, Any]:
    if isinstance(response, dict):
        result = response.copy()
        _normalize_planning_result(result)
        result["parse_success"] = True
        result["raw_response"] = str(response)
        return result

    response_text = response if isinstance(response, str) else str(response)

    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response_text.strip()

    try:
        json_str_cleaned = _clean_json_control_chars(json_str)
        decoder = json.JSONDecoder()
        result, idx = decoder.raw_decode(json_str_cleaned.lstrip())

        if not isinstance(result, dict):
            raise ValueError("Response is not a JSON object")

        _normalize_planning_result(result)
        result["parse_success"] = True
        result["raw_response"] = response_text
        return result

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"❌ Error parsing planning response: {e}")
        logger.error(f"Response type: {type(response).__name__}")
        logger.error(f"Response preview (first 500 chars): {response_text[:500]}...")
        if len(response_text) > 500:
            logger.error(f"Response length: {len(response_text)} chars (truncated)")

        return {
            "reason": f"Failed to parse JSON response: {str(e)}. Raw analysis is available below.",
            "module": [],
            "plan": {},
            "raw_response": response_text,
            "parse_success": False,
        }


def _normalize_planning_result(result: Dict[str, Any]) -> None:
    if "reason" not in result:
        result["reason"] = "No reason provided"
    if "module" not in result:
        result["module"] = []
    if "plan" not in result:
        result["plan"] = {}

    if not isinstance(result["module"], list):
        logger.warning("'module' is not a list, converting to list")
        result["module"] = [result["module"]] if result["module"] else []

    all_modules = []
    for module in result["module"]:
        if module not in PLANNING_ALLOWED_MODULES:
            logger.warning(f"⚠️ Module name '{module}' is not in allowed list, but will be kept for diff generation.")
        all_modules.append(module)
    result["module"] = all_modules

    raw_plan = result.get("plan", {})
    if isinstance(raw_plan, dict):
        result["plan"] = raw_plan
    elif isinstance(raw_plan, str):
        plan_text = raw_plan.strip()
        if plan_text:
            reason = (result.get("reason") or "").strip()
            result["reason"] = (reason + "\n\n" + plan_text).strip() if reason else plan_text
            result["plan"] = {m: plan_text for m in (result.get("module") or [])} if result.get("module") else {}
        else:
            result["plan"] = {}
    else:
        result["plan"] = {}

    if len(all_modules) > 0 and len(result["plan"]) == 0:
        logger.warning(
            f"⚠️ Planning returned modules {all_modules} but empty 'plan' field. "
            f"This may indicate a schema compliance issue."
        )


def _clean_json_control_chars(json_text: str) -> str:
    def replace_control_in_string(match):
        string_content = match.group(1)
        cleaned = string_content
        cleaned = re.sub(r'(?<!\\)\n', r'\\n', cleaned)
        cleaned = re.sub(r'(?<!\\)\t', r'\\t', cleaned)
        cleaned = re.sub(r'(?<!\\)\r', r'\\r', cleaned)
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', cleaned)
        return f'"{cleaned}"'

    pattern = r'"((?:[^"\\]|\\.)*)"'
    return re.sub(pattern, replace_control_in_string, json_text)


# ============ Shared planner text builders ============

_DEFAULT_SELECTION_GUIDELINES = """\
2. **Prefer selecting 1-2 components** (or at most 3) for targeted enhancement.
3. Choose components where modifications are most likely to improve performance.
4. Avoid changing the evaluation metric or submission/output format.
5. Avoid breaking interfaces or variable names used across components.
6. Consider execution output, performance bottlenecks, or stability issues when selecting components.
7. **Focus on generalizable enhancements** that improve test performance, not just validation performance.
8. **Follow the scientific approach structure** when writing plans - clearly state WHAT, WHY, HOW, and interface constraints.
"""


def build_planner_task(stage_intro: str, extra_guidelines: str = "") -> str:
    parts = ["# Your Plan\n", stage_intro, "\n\n**Selection Guidelines:**"]
    if extra_guidelines:
        parts.append(f"\n{extra_guidelines}")
    parts.append(f"\n{_DEFAULT_SELECTION_GUIDELINES}")
    return "".join(parts)


def build_planner_suffix(
    prompt_base: Dict[str, Any],
    data_preview: str,
    context: Dict[str, Any],
    extra_text: str = "",
) -> str:
    from utils.response import wrap_code  # lazy import to avoid circular

    code = prompt_base.get("Previous solution", {}).get("Code", "")
    execution_output = context.get("execution_output", "")

    suffix = (
        f"Let me approach this systematically.\n"
        f"First, I'll examine the dataset:\n{data_preview}\n"
        f"Regarding this task, I previously made attempts with the following code:\n{code}\n"
        f"The execution of this code yielded the following results:\n{wrap_code(execution_output, lang='')}\n"
    )
    if extra_text:
        suffix += extra_text + "\n"
    suffix += "Now I will output my analysis in JSON format only (no additional text):"
    return suffix


# ============ Generic planner entry ============

def run_planner(
    agent_instance,
    prompt_base: Dict[str, Any],
    data_preview: str,
    context: Dict[str, Any],
    your_task_section: str,
    assistant_suffix: str,
    extra_prompt_sections: Dict[str, str] | None = None,
    max_retries: int = 3,
    stage_name: str = "Planning",
) -> Dict[str, Any]:
    component_descriptions = get_component_descriptions()
    component_desc_parts = [f"- **{name}**: {desc}" for name, desc in component_descriptions.items()]
    component_desc_text = "\n".join(component_desc_parts)

    planning_prompt_dict = prompt_base.copy()

    planning_prompt_dict["Instructions"]["Response Format"] = [
        "",
        "**CRITICAL: You must output your analysis in JSON format only (no additional text).**",
        "",
        "Output format:",
        PLANNING_JSON_FORMAT,
        "",
        "**CRITICAL Requirements:**",
        f"- `reason`: 1-4 sentences explaining why these components are the best targets now, including root cause analysis.",
        f"- `module`: An **array** of module names. Each element **MUST be one of**: {', '.join([repr(m) for m in PLANNING_ALLOWED_MODULES])}",
        "- `plan`: An object where:",
        "    * Keys **MUST be exactly the same** as module names in the `module` array",
        "    * Values are **detailed modification plans** (2-5 sentences) following the scientific approach structure:",
        "        - WHAT to change (specific technical modification with category)",
        "        - WHY this change (root cause analysis, why THIS TASK needs it)",
        "        - HOW to implement (specific implementation approach)",
        "        - Interface constraints (variable names, function signatures to preserve)",
        "    * Include any **interface/variable constraints** to keep other components working",
        "- Return **ONLY** the JSON object, no markdown code blocks, no explanations before or after",
    ]

    planning_prompt_dict["Available Components"] = component_desc_text

    instructions = "\n# Instructions\n\n"
    instructions += compile_prompt_to_md(planning_prompt_dict["Instructions"], 2)

    memory_section = ""
    memory_text = planning_prompt_dict.get("Memory", "")
    if memory_text and str(memory_text).strip():
        memory_section = f"# Memory\nBelow is a record of previous improvement attempts and their outcomes:\n {memory_text}"

    user_prompt_parts = [
        f"\n# Task description\n{planning_prompt_dict['Task description']}\n",
        f"{memory_section}\n" if memory_section else "",
    ]
    if extra_prompt_sections:
        for section_text in extra_prompt_sections.values():
            if section_text:
                user_prompt_parts.append(f"{section_text}\n")

    user_prompt_parts.extend([
        f"# Available Components\n{component_desc_text}\n\n",
        f"{instructions}\n\n",
        f"{your_task_section}",
    ])
    user_prompt = "".join(user_prompt_parts)

    model_name = agent_instance.acfg.code.model.lower()
    planning_prompt_complete = build_model_prompt(
        model_name=model_name,
        introduction=planning_prompt_dict.get("Introduction", ""),
        user_prompt=user_prompt,
        assistant_suffix=assistant_suffix,
    )

    planning_result = None
    for attempt in range(max_retries):
        logger.info(f"Calling {stage_name} Agent to analyze which modules to modify... (attempt {attempt + 1}/{max_retries})")

        json_schema = PLANNING_JSON_SCHEMA

        planning_response = generate(
            prompt=planning_prompt_complete,
            temperature=agent_instance.acfg.code.temp,
            cfg=agent_instance.cfg,
            json_schema=json_schema,
        )

        planning_result = parse_planning_response(planning_response)
        parse_success = planning_result.get("parse_success", True)

        if not parse_success:
            logger.error(f"❌ {stage_name} attempt {attempt + 1}: JSON parsing failed")
            if attempt < max_retries - 1:
                logger.info(f"Retrying {stage_name} Agent (JSON parsing failed)...")
                continue
            raw = planning_result.get("raw_response", "")
            if raw and thinking_json_incompatible(agent_instance.acfg.code.model):
                logger.warning("Qwen thinking mode: using raw response as fallback plan")
                return {
                    "reason": raw,
                    "module": [],
                    "plan": {},
                    "parse_success": True,
                    "raw_response": raw,
                }
            logger.error(f"All {stage_name} attempts failed due to JSON parsing errors")
            return planning_result

        modules = planning_result.get("module", [])
        plans = planning_result.get("plan", {})
        has_plan = isinstance(plans, dict) and len(plans) > 0

        if len(modules) > 0:
            if len(plans) == 0:
                logger.warning(
                    f"⚠️ {stage_name} returned modules {modules} but empty 'plan' field."
                )
                if attempt < max_retries - 1:
                    logger.info(f"Retrying {stage_name} Agent (modules selected but plan is empty)...")
                    continue
                logger.warning("Proceeding with modules + reason only on last attempt.")
            elif len(plans) < len(modules):
                missing = set(modules) - set(plans.keys())
                logger.warning(f"⚠️ {stage_name} returned modules {modules} but 'plan' missing keys: {missing}")

            logger.info(f"✅ {stage_name} Result: {planning_result['reason']}")
            logger.info(f"Modules to modify: {modules}")
            if plans:
                logger.info(f"Plans provided for: {list(plans.keys())}")
            return planning_result

        elif has_plan:
            plan_modules = list(plans.keys())
            logger.info(f"{stage_name}: No modules in 'module' field, but found {len(plan_modules)} modules in 'plan': {plan_modules}")
            planning_result["module"] = plan_modules
            logger.info(f"✅ {stage_name} Result: {planning_result['reason']}")
            return planning_result

        else:
            logger.warning(f"⚠️ {stage_name} attempt {attempt + 1}: No modules selected and no plan provided")
            if parse_success:
                logger.warning(f"Reason provided: {planning_result.get('reason', 'N/A')}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying {stage_name} Agent (no modules selected)...")
                continue

    logger.warning(f"❌ {stage_name} Agent failed after all retries, returning empty result (will trigger fallback)")
    raw = planning_result.get("raw_response", "") if planning_result else ""
    if raw and thinking_json_incompatible(agent_instance.acfg.code.model):
        logger.warning(f"Qwen thinking mode: using raw response as fallback plan")
        return {
            "reason": raw,
            "module": [],
            "plan": {},
            "parse_success": True,
            "raw_response": raw,
        }
    return {
        "reason": f"{stage_name} Agent failed after retries (invalid responses), falling back to full rewrite",
        "module": [],
        "plan": {},
        "parse_success": False,
        "raw_response": raw,
    }
