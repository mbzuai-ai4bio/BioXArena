"""Memory-enhanced two-stage planning.

Stage 1: generate_initial_plan  — produce a free-text plan (optionally guided
         by dissimilar historical records from GlobalMemoryLayer).
Stage 2: refine_plan_to_json   — retrieve similar success/fail records from
         memory, then convert the text plan into structured JSON.

Used by improve_agent when global memory is available.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

from llm import generate, compile_prompt_to_md
from llm.model_profiles import thinking_json_incompatible
from utils.response import wrap_code
from .base_planner import (
    PLANNING_ALLOWED_MODULES,
    PLANNING_JSON_FORMAT,
    PLANNING_JSON_SCHEMA,
    get_component_descriptions,
    build_model_prompt,
    parse_planning_response,
)

logger = logging.getLogger("MLEvolve")


# ============ Stage 1: Generate initial text plan ============

def generate_initial_plan(
    agent_instance,
    prompt_base: Dict[str, Any],
    data_preview: str,
    context: Dict[str, Any],
) -> str:
    initial_prompt_dict = prompt_base.copy()

    component_descriptions = get_component_descriptions()
    component_desc_parts = [f"- **{name}**: {desc}" for name, desc in component_descriptions.items()]
    component_desc_text = "\n".join(component_desc_parts)
    initial_prompt_dict["Available Components"] = component_desc_text

    initial_prompt_dict["Instructions"]["Output Format"] = [
        "",
        "**Output Requirements:**",
        "- Identify key components (at most 4) that need modification to improve performance.",
        "- Write a detailed paragraph explaining:",
        "  * WHAT specific components/aspects you plan to change",
        "  * WHY these changes are needed (based on execution results and task requirements)",
        "  * WHAT approach you will take (high-level strategy)",
        "- Be specific about which components (from the Available Components list above) are the focus.",
        "- Your plan should be comprehensive and actionable, following the improvement guidelines above.",
        "- Output your plan in **plain text format**.",
        "- **Plan only**: DO NOT output any complete code snippets, markdown code blocks, or execution logs.",
    ]

    instructions = "\n# Instructions\n\n"
    instructions += compile_prompt_to_md(initial_prompt_dict["Instructions"], 2)

    if "Available Components" in initial_prompt_dict:
        instructions += f"\n\n# Available Components\n\n{initial_prompt_dict['Available Components']}\n"

    introduction = (
        "You are a Kaggle grandmaster attending a competition. "
        "Based on the current solution, generate an improvement plan."
    )

    memory_section = f"# Memory\n{prompt_base.get('Memory', '')}"

    user_prompt = (
        f"\n# Task description\n{prompt_base.get('Task description', '')}\n\n"
        f"{memory_section}\n\n"
        f"{instructions}\n"
    )

    assistant_suffix = (
        f"Okay! I will analyze the current solution and propose an improvement plan.\n"
        f"Dataset information:\n{data_preview}\n"
        f"Previous code:\n{prompt_base['Previous solution']['Code']}\n"
        f"Execution results:\n{wrap_code(context.get('execution_output', ''), lang='')}\n"
        f"Based on the above, I believe there is room for improvement. "
        f"I will give a plan (only a plan no complete code or markdown code blocks)."
    )

    model_name = agent_instance.acfg.code.model.lower()
    prompt_complete = build_model_prompt(
        model_name=model_name,
        introduction=introduction,
        user_prompt=user_prompt,
        assistant_suffix=assistant_suffix,
    )

    planning_response = generate(
        prompt=prompt_complete,
        temperature=agent_instance.acfg.code.temp,
        cfg=agent_instance.cfg,
        json_schema=None,
    )

    planning_text = planning_response.strip() if isinstance(planning_response, str) else str(planning_response).strip()
    logger.info(f"[InitialPlan] Generated text plan (length: {len(planning_text)} chars)")
    return planning_text


# ============ Stage 2: Refine plan to JSON with memory ============

def refine_plan_to_json(
    agent_instance,
    initial_plan_text: str,
    prompt_base: Dict[str, Any],
    data_preview: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    query_text = initial_plan_text

    similar_success_records = agent_instance.global_memory.retrieve_similar_records(
        query_text=query_text,
        top_k=2,
        alpha=0.5,
        dissimilar=False,
        label_filter=1,
    )

    similar_fail_records = agent_instance.global_memory.retrieve_similar_records(
        query_text=query_text,
        top_k=2,
        alpha=0.5,
        dissimilar=False,
        label_filter=-1,
    )

    refinement_guidance = _build_refinement_guidance(similar_success_records, similar_fail_records)

    component_descriptions = get_component_descriptions()
    component_desc_parts = [f"- **{name}**: {desc}" for name, desc in component_descriptions.items()]
    component_desc_text = "\n".join(component_desc_parts)

    user_prompt = _build_refine_user_prompt(
        prompt_base, initial_plan_text, refinement_guidance, component_desc_text,
    )

    if refinement_guidance:
        introduction = (
            "You are refining an improvement plan with historical reference. "
            "Convert the initial text plan into a structured JSON format with specific module-level modifications, "
            "optimizing it based on similar historical experiments while staying true to the initial plan's direction."
        )
        assistant_suffix = (
            f"Okay! I will refine the initial plan into a structured JSON format, optimizing it based on similar historical experiments.\n"
            f"Dataset information:\n{data_preview}\n"
            f"Current code:\n{prompt_base['Previous solution']['Code']}\n"
            f"Execution results:\n{wrap_code(context.get('execution_output', ''), lang='')}\n"
            f"Based on the initial plan and similar historical experiments (what worked and what failed), "
            f"I will refine it into specific module-level modifications while staying true to the initial plan's core direction."
        )
    else:
        introduction = (
            "You are converting an improvement plan into structured format. "
            "Convert the initial text plan into a structured JSON format with specific module-level modifications, "
            "following the initial plan's direction closely."
        )
        assistant_suffix = (
            f"Okay! I will convert the initial plan into a structured JSON format.\n"
            f"Dataset information:\n{data_preview}\n"
            f"Current code:\n{prompt_base['Previous solution']['Code']}\n"
            f"Execution results:\n{wrap_code(context.get('execution_output', ''), lang='')}\n"
            f"Based on the initial plan, task requirements, and execution results, "
            f"I will convert it into specific module-level modifications."
        )

    model_name = agent_instance.acfg.code.model.lower()
    planning_prompt_complete = build_model_prompt(
        model_name=model_name,
        introduction=introduction,
        user_prompt=user_prompt,
        assistant_suffix=assistant_suffix,
    )

    json_schema = PLANNING_JSON_SCHEMA
    max_retries = 3
    planning_result = None

    for attempt in range(max_retries):
        logger.info(f"[RefinePlan] Calling LLM to generate JSON plan... (attempt {attempt + 1}/{max_retries})")

        planning_response = generate(
            prompt=planning_prompt_complete,
            temperature=agent_instance.acfg.code.temp,
            cfg=agent_instance.cfg,
            json_schema=json_schema,
        )

        if not planning_response or (isinstance(planning_response, str) and not planning_response.strip()):
            logger.warning(f"⚠️ RefinePlan attempt {attempt + 1}: Empty response from LLM")
            if attempt < max_retries - 1:
                continue
            else:
                return {
                    "reason": "RefinePlan failed after retries (empty responses from LLM)",
                    "module": [], "plan": {},
                    "parse_success": False, "raw_response": "",
                }

        planning_result = parse_planning_response(planning_response)
        parse_success = planning_result.get("parse_success", True)

        if not parse_success:
            logger.error(f"❌ RefinePlan attempt {attempt + 1}: JSON parsing failed")
            if attempt < max_retries - 1:
                continue
            else:
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
                return planning_result

        modules = planning_result.get("module", [])
        plans = planning_result.get("plan", {})
        has_plan = isinstance(plans, dict) and len(plans) > 0

        if len(modules) > 0:
            if len(plans) == 0:
                logger.warning(f"⚠️ RefinePlan returned modules {modules} but empty 'plan' field.")
                if attempt < max_retries - 1:
                    continue
                logger.warning("Proceeding with modules + reason only on last attempt.")
            elif len(plans) < len(modules):
                missing = set(modules) - set(plans.keys())
                logger.warning(f"⚠️ RefinePlan returned modules {modules} but 'plan' missing keys: {missing}")

            logger.info(f"[RefinePlan] Generated JSON plan with {len(modules)} modules")
            return planning_result

        elif has_plan:
            plan_modules = list(plans.keys())
            planning_result["module"] = plan_modules
            logger.info(f"[RefinePlan] Generated JSON plan with {len(plan_modules)} modules (from plan keys)")
            return planning_result

        else:
            logger.warning(f"⚠️ RefinePlan attempt {attempt + 1}: No modules selected and no plan provided")
            if attempt < max_retries - 1:
                continue

    logger.warning("❌ RefinePlan failed after all retries, returning empty result (will trigger fallback)")
    return {
        "reason": "RefinePlan failed after retries (invalid responses), falling back to full rewrite",
        "module": [], "plan": {},
        "parse_success": False,
        "raw_response": planning_result.get("raw_response", "") if planning_result else "",
    }


# ============ Internal helpers ============

def _build_refinement_guidance(similar_success_records, similar_fail_records) -> str:
    """Build guidance text from retrieved similar records."""
    if not similar_success_records and not similar_fail_records:
        return ""

    guidance_parts = [
        "## Historical Experience of Similar Experiments: ",
        "",
    ]

    if similar_success_records:
        guidance_parts.append("**✅ Successful Similar Experiments:**")
        for idx, (record, score) in enumerate(similar_success_records, 1):
            guidance_parts.append(f"{idx}. Plan: {record.description}")
            guidance_parts.append(f"   Method: {record.method}")
        guidance_parts.append("")

    if similar_fail_records:
        guidance_parts.append("**❌ Failed Similar Experiments (Avoid):**")
        for idx, (record, score) in enumerate(similar_fail_records, 1):
            guidance_parts.append(f"{idx}. Plan: {record.description}")
            guidance_parts.append(f"   Method: {record.method}")
        guidance_parts.append("")

    logger.info(
        f"[RefinePlan] Retrieved {len(similar_success_records)} success "
        f"and {len(similar_fail_records)} fail records"
    )
    return "\n".join(guidance_parts)


def _build_refine_user_prompt(
    prompt_base: Dict[str, Any],
    initial_plan_text: str,
    refinement_guidance: str,
    component_desc_text: str,
) -> str:
    """Build the user prompt for the refine stage."""
    parts = [
        "# Task description",
        prompt_base.get("Task description", ""),
        "",
        "# Initial Plan (Stage 1)",
        initial_plan_text,
        "",
    ]

    if refinement_guidance:
        parts.extend([refinement_guidance, ""])

    parts.extend([
        "# Available Components",
        component_desc_text,
        "",
        "# Task",
        "",
    ])

    if refinement_guidance:
        parts.extend([
            "Refine the initial plan above into a structured JSON format. This is a refinement step with historical reference:",
            "",
            "**Key Points:**",
            "- Convert the initial plan into specific module-level modifications",
            "- **Reference similar historical experiments** above to optimize the plan (learn from what worked and avoid what failed)",
            "- **DO NOT deviate** from the initial plan's core direction - use historical experiences to refine and enhance, not replace",
            "- Consider the task requirements, current code state, and execution results (provided in assistant message)",
            "",
            "**Requirements:**",
            f"- `reason`: Explain why these components are selected, considering the initial plan and how similar historical experiments inform the refinement.",
            f"- `module`: An **array** of module names (1-3 elements). Each element **MUST be one of**: {', '.join([repr(m) for m in PLANNING_ALLOWED_MODULES])}",
            "- `plan`: An object where:",
            "    * Keys **MUST be exactly the same** as module names in the `module` array",
            "    * Values are **detailed modification plans** (2-5 sentences) following:",
            "        - WHAT to change (specific technical modification, refined based on historical experiences)",
            "        - WHY this change (based on initial plan and how similar cases inform the decision)",
            "        - HOW to implement (specific approach, informed by similar successful/failed cases)",
            "        - Interface constraints (preserve variable names and function signatures)",
        ])
    else:
        parts.extend([
            "Convert the initial plan above into a structured JSON format. This is a direct conversion step:",
            "",
            "**Key Points:**",
            "- Convert the text plan into specific module-level modifications",
            "- Follow the initial plan's direction closely",
            "- Consider the task requirements, current code state, and execution results (provided in assistant message)",
            "",
            "**Requirements:**",
            f"- `reason`: Explain why these components are selected based on the initial plan.",
            f"- `module`: An **array** of module names (1-3 elements). Each element **MUST be one of**: {', '.join([repr(m) for m in PLANNING_ALLOWED_MODULES])}",
            "- `plan`: An object where:",
            "    * Keys **MUST be exactly the same** as module names in the `module` array",
            "    * Values are **detailed modification plans** (2-5 sentences) following:",
            "        - WHAT to change (specific technical modification)",
            "        - WHY this change (based on initial plan and execution results)",
            "        - HOW to implement (specific approach)",
            "        - Interface constraints (preserve variable names and function signatures)",
        ])

    parts.extend([
        "",
        "**Output format:**",
        PLANNING_JSON_FORMAT,
        "",
        "**CRITICAL:** Return **ONLY** the JSON object, no markdown code blocks, no explanations before or after.",
    ])

    return "\n".join(parts)
