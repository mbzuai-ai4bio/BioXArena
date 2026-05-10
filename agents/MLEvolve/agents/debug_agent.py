import logging
from typing import Any, List, Tuple

from llm import compile_prompt_to_md, generate
from engine.search_node import SearchNode
from agents.coder import plan_and_code_query
from utils.response import extract_plan_from_diff_response, wrap_code
from agents.prompts import (
    ROBUSTNESS_GENERALIZATION_STRATEGY,
    get_internet_clarification,
    get_impl_guideline_from_agent,
)

from agents.coder.diff_coder import SearchReplacePatcher, DIFF_SYS_FORMAT
from agents.planner import build_chat_prompt_for_model
from agents.triggers import register_node

logger = logging.getLogger("MLEvolve")


def _format_debug_memory_guidance(agent, similar_fixes: List[Tuple]) -> str:
    if not similar_fixes:
        return ""

    guidance_parts = [
        "## Historical Debug Experience",
        "",
        "The following similar errors have been successfully fixed in previous attempts:",
        ""
    ]

    case_idx = 0
    for record, score in similar_fixes:
        if not record.description or not record.description.strip():
            continue

        case_idx += 1
        guidance_parts.append(f"**Case {case_idx}:**")

        if record.record_id in agent.global_memory.node_metadata_map:
            metadata = agent.global_memory.node_metadata_map[record.record_id]
            parent_error = metadata.get("parent_error", "")
            if parent_error:
                error_preview = parent_error[:200] + ("..." if len(parent_error) > 200 else "")
                guidance_parts.append(f"- Similar Error: {error_preview}")

        guidance_parts.append(f"- Fix Strategy: {record.description}")
        guidance_parts.append("")

    if case_idx == 0:
        return ""

    guidance_parts.append("**Note**: Consider applying similar fix strategies if applicable.")
    guidance_parts.append("")

    return "\n".join(guidance_parts)


def run(agent, parent_node: SearchNode) -> SearchNode:
    debugging_standards = (
        "🔧 Debug SYSTEMATICALLY: Read error → Identify root cause → Apply minimal, targeted fix.\n\n"
        "**Do**: Fix root cause, preserve solution intent, maintain code quality.\n"
        "**Don't**: Random changes, delete large sections, replace model with dummy predictions, take shortcuts.\n\n"
    )

    full_code_requirement = (
        "\n\n"
        "🔴 **CRITICAL REQUIREMENT - Read Carefully:**\n"
        "Your response MUST contain a COMPLETE, SELF-CONTAINED, EXECUTABLE Python script from start to finish.\n"
        "❌ DO NOT provide partial code snippets or modifications only\n"
        "❌ DO NOT assume previous code context exists\n"
        "❌ DO NOT use placeholder comments like '# ... rest of training code ...'\n"
        "✅ DO provide the ENTIRE solution including:\n"
        "   • All imports at the top\n"
        "   • All data loading and preprocessing\n"
        "   • Complete model definition and training\n"
        "   • Complete validation metric calculation\n"
        "   • Complete test inference and submission.csv generation\n"
        "   • Every line of code needed to run from beginning to end\n\n"
        "Your response format:\n"
        "1. A brief implementation outline (2-3 sentences) explaining the bugfix\n"
        "2. A single markdown code block containing the COMPLETE executable solution with the bugfix applied\n\n"
    )

    bug_description = "Your previous solution encountered an issue — it either failed during execution, did not generate the required output files, or produced output in an incorrect format"

    introduction_base = (
        debugging_standards +
        f"{bug_description}, "
        "so based on the information below, you should revise it in order to fix this. "
        "\n\n"
        "Remember: The code will be executed in a fresh Python environment. It must be 100% self-contained."
    )

    introduction = introduction_base

    prompt: Any = {
        "Introduction": introduction,
        "Task description": agent.task_desc,
        "Previous (buggy) implementation": wrap_code(parent_node.code),
        "Execution output": wrap_code(parent_node.term_out, lang=""),
        "Instructions": {},
    }
    prompt["Instructions"] |= {
        "Bugfix improvement sketch guideline": [
            "- You should write a brief natural language description (2-3 sentences) of how the issue in the previous implementation can be fixed.\n",
            "- Don't suggest to do EDA.\n",
            "- Most libraries are stable and available. The bug is not caused by the library version mismatch. **Don't suggest to reinstall the core libraries.** (like pip install torch, pip upgrade transformers, !pip install tensorflow, subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers', 'accelerate', 'pandas', 'torch', 'torchvision']))\n",
        ],
    }
    prompt["Instructions"] |= get_impl_guideline_from_agent(agent)
    prompt["Instructions"] |= ROBUSTNESS_GENERALIZATION_STRATEGY

    internet_clarification = get_internet_clarification(getattr(agent.cfg, "pretrain_model_dir", ""))
    prompt["Instructions"]["Implementation guideline"].extend(internet_clarification)

    debug_memory_guidance = ""
    if agent.global_memory and len(agent.global_memory.records) > 0:
        current_error = parent_node.term_out or getattr(parent_node, 'execution_output', '')

        if current_error and current_error.strip():
            try:
                logger.debug(f"[Debug] Retrieving similar errors, query_length={len(current_error)}, memory_records={len(agent.global_memory.records)}")
                similar_fixes = agent.global_memory.retrieve_similar_records(
                    query_text=current_error,
                    top_k=2,
                    alpha=0.5,
                    dissimilar=False,
                    label_filter=1,
                    stage_filter="debug",
                )

                if similar_fixes:
                    debug_memory_guidance = _format_debug_memory_guidance(agent, similar_fixes)
                    logger.info(f"[Debug] Found {len(similar_fixes)} similar errors with successful fixes from memory")
            except Exception as e:
                logger.warning(f"[Debug] Failed to retrieve memory for debug: {e}")
        else:
            logger.warning(f"[Debug] No current error found for debug")
    else:
        logger.debug(f"[Debug] No global memory")

    if debug_memory_guidance:
        prompt["Instructions"]["Historical Debug Experience"] = [debug_memory_guidance]

    base_instructions = "\n# Instructions\n\n"
    base_instructions += compile_prompt_to_md(prompt["Instructions"], 2)

    def build_prompt_complete(instructions_with_format, use_full_code_requirement=False):
        current_introduction = introduction_base + (full_code_requirement if use_full_code_requirement else "")
        user_prompt = f"\n# Task description\n{prompt['Task description']}\n{instructions_with_format}"
        assistant_prefix = f"Let me approach this systematically.\nFirst, I'll review the dataset:\n{agent.data_preview}\nThe code that needs fixing:\n{prompt['Previous (buggy) implementation']}\nThe error/issue encountered:\n{prompt['Execution output']}\nAnalyzing the root cause: {parent_node.analysis}\nI'll now fix this issue."
        return build_chat_prompt_for_model(agent.acfg.code.model, current_introduction, user_prompt, assistant_prefix)

    parent_node.add_expected_child_count()

    plan, code = None, None
    prompt_complete = None
    max_diff_retries = 3

    if agent.acfg.use_diff_mode:
        diff_instructions = base_instructions + f"\n\n🔴 **IMPORTANT**: There is a bug that MUST be fixed. "
        diff_instructions += f"You MUST provide code modifications using SEARCH/REPLACE format. "
        diff_instructions += (
            "Do NOT return unchanged code. "
            "Keep each SEARCH snippet minimal (ONLY the lines to be replaced, plus tiny context if needed). "
            "Prefer multiple small SEARCH/REPLACE blocks instead of one large block.\n\n"
        )
        diff_instructions += f"Response format: {DIFF_SYS_FORMAT}"

        current_code = parent_node.code
        total_applied = 0
        retry_note = ""
        for retry_idx in range(max_diff_retries):
            try:
                logger.info(f"Attempting diff method (retry {retry_idx + 1}/{max_diff_retries}) for node {parent_node.id}")
                try:
                    prompt["Previous (buggy) implementation"] = current_code
                except Exception:
                    pass

                diff_instructions_retry = diff_instructions
                if retry_note:
                    diff_instructions_retry += (
                        "\n\n🔁 **RETRY NOTE (IMPORTANT)**:\n"
                        f"{retry_note}\n"
                        "Now output ONLY the remaining SEARCH/REPLACE blocks needed to finish. "
                        "Do NOT repeat already-applied blocks. "
                        "Keep SEARCH minimal and ensure every block is complete.\n"
                    )

                prompt_with_diff = build_prompt_complete(diff_instructions_retry)

                response = generate(
                    prompt=prompt_with_diff,
                    temperature=agent.acfg.code.temp,
                    cfg=agent.cfg
                )

                if response and ("<<<<<<< SEARCH" in response or "< SEARCH" in response or "<<<<<<<" in response):
                    if "<<<<<<< SEARCH" in response:
                        search_markers = response.count("<<<<<<< SEARCH")
                        replace_markers = response.count(">>>>>>> REPLACE")
                    elif "< SEARCH" in response:
                        search_markers = response.count("< SEARCH")
                        replace_markers = response.count("> REPLACE")
                    else:
                        search_markers = 1
                        replace_markers = 0
                    has_incomplete_block = search_markers > replace_markers

                    patcher = SearchReplacePatcher()
                    updated_code, count = patcher.apply_patch(response, current_code, strict=False)
                    if count > 0 and updated_code and updated_code != current_code:
                        current_code = updated_code
                        total_applied += count

                    if total_applied > 0 and current_code and current_code != parent_node.code and not has_incomplete_block:
                        plan = extract_plan_from_diff_response(response).strip()
                        if not plan:
                            error_parts = []
                            parent_error = getattr(parent_node, "exc_type", None)
                            parent_analysis = getattr(parent_node, "analysis", None)
                            if parent_error:
                                error_parts.append(f"Parent error: {parent_error}")
                            if parent_analysis:
                                error_parts.append(f"Parent analysis: {parent_analysis}")
                            if not error_parts:
                                error_parts.append("I will debug the code to fix the bug.")
                            plan = " | ".join(error_parts)

                        code = current_code
                        prompt_complete = prompt_with_diff
                        logger.info(
                            f"Successfully applied {total_applied} diff patch(es) for node {parent_node.id} "
                            f"(last attempt applied={count}, retry {retry_idx + 1}/{max_diff_retries})"
                        )
                        break
                    else:
                        if has_incomplete_block and (count > 0 or total_applied > 0):
                            retry_note = (
                                "Your previous diff output appears truncated/incomplete (missing closing '>>>>>>> REPLACE'). "
                                f"I have already applied {total_applied} patch(es) to the code. "
                                "Please continue and provide ONLY the remaining patches."
                            )
                        else:
                            retry_note = (
                                "Your previous diff did not apply cleanly to the current code. "
                                "Please generate minimal SEARCH/REPLACE blocks that match the CURRENT code exactly."
                            )
                        logger.warning(
                            f"Diff patch attempt {retry_idx + 1}/{max_diff_retries}: "
                            f"count={count}, total_applied={total_applied}, "
                            f"code_changed={current_code != parent_node.code if current_code else False}, "
                            f"search_markers={search_markers}, replace_markers={replace_markers}, "
                            f"has_incomplete_block={has_incomplete_block}"
                        )
                        if retry_idx < max_diff_retries - 1:
                            logger.info(f"Retrying diff method...")
                        else:
                            logger.warning(f"All {max_diff_retries} diff attempts failed, will fallback to full rewrite")
                else:
                    logger.warning(
                        f"Diff attempt {retry_idx + 1}/{max_diff_retries}: "
                        f"Response does not contain SEARCH/REPLACE format"
                    )
                    retry_note = (
                        "Your previous output did not contain valid SEARCH/REPLACE blocks. "
                        "Output ONLY complete SEARCH/REPLACE blocks (no other text)."
                    )
                    if retry_idx < max_diff_retries - 1:
                        logger.info(f"Retrying diff method...")
                    else:
                        logger.warning(f"All {max_diff_retries} diff attempts failed, will fallback to full rewrite")
            except Exception as e:
                logger.warning(f"Diff attempt {retry_idx + 1}/{max_diff_retries} failed with exception: {e}")
                retry_note = (
                    f"Your previous diff failed to apply due to an error: {e}. "
                    "Please output minimal SEARCH/REPLACE blocks that match the CURRENT code exactly, "
                    "and ensure every block is complete."
                )
                if retry_idx < max_diff_retries - 1:
                    logger.info(f"Retrying diff method...")
                else:
                    logger.warning(f"All {max_diff_retries} diff attempts failed, will fallback to full rewrite")

        if code is None and total_applied > 0:
            code = current_code
            if plan is None:
                plan = "Partial diff patches applied; continuing with partially fixed code."

    if code is None:
        logger.info(f"Falling back to full code rewrite debugging method for node {parent_node.id}")
        prompt_complete = build_prompt_complete(base_instructions, use_full_code_requirement=True)
        plan, code = plan_and_code_query(agent, prompt_complete)

    from_topk = getattr(parent_node, '_topk_triggered', False)

    new_node = SearchNode(plan=plan, code=code, parent=parent_node, stage="debug",
                        local_best_node=parent_node.local_best_node, from_topk=from_topk)
    register_node(agent, new_node, prompt_complete, parent_node=parent_node)

    logger.info(f"[debug] {parent_node.id} → node {new_node.id}")
    return new_node
