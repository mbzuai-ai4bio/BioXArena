"""Improve Agent: generate improved plan/code from a successful parent node (diff or full mode)."""

import logging
import time
from typing import Any

from llm import compile_prompt_to_md
from engine.search_node import SearchNode
from utils.response import wrap_code
from agents.triggers import get_patience_counter, register_node
from agents.prompts import (
    ROBUSTNESS_GENERALIZATION_STRATEGY,
    prompt_leakage_prevention,
    prompt_resp_fmt,
    get_internet_clarification,
    get_impl_guideline_from_agent,
)
from agents.planner import run_planner, generate_initial_plan, refine_plan_to_json, build_planner_task, build_planner_suffix, build_chat_prompt_for_model
from agents.coder import plan_and_code_query
from agents.coder.diff_coder import diff_generate_and_apply

logger = logging.getLogger("MLEvolve")


def run(agent, parent_node: SearchNode) -> SearchNode:
    improvement_standards = (
        "🎯 As a Grandmaster, make MEANINGFUL improvements that boost leaderboard performance.\n\n"
        "**Acceptable**: Advanced architectures, ensemble techniques, feature engineering, hyperparameter optimization, improved pipelines.\n"
        "**NOT Acceptable**: Cosmetic changes, minor tweaks without justification, breaking functionality.\n\n"
    )

    introduction = (
        improvement_standards +
        "You are provided with a previously developed solution below and should improve it "
        "in order to further increase the (test time) performance. "
        "For this you should first outline a brief plan in natural language for how the solution can be improved and "
        "then implement this improvement in Python based on the provided previous solution."
    )

    prompt: Any = {
        "Introduction": introduction,
        "Task description": agent.task_desc,
        "Memory": parent_node.fetch_child_memory(include_code=False),
        "Instructions": {},
    }
    prompt["Previous solution"] = {
        "Code": wrap_code(parent_node.code),
    }

    success_patience, total_patience, branch_best_score = get_patience_counter(agent, parent_node)
    use_magnitude_prompt = (success_patience >= 2) or (total_patience >= 5)

    if use_magnitude_prompt:
        trigger_reason = []
        if success_patience >= 2:
            trigger_reason.append(f"success_patience={success_patience}>=2")
        if total_patience >= 5:
            trigger_reason.append(f"total_patience={total_patience}>=5")
        logger.warning(f"🔥 PLATEAU DETECTED! Triggered by: {' AND '.join(trigger_reason)}, using Magnitude-Based prompt")
        if branch_best_score is None:
            best_score_str = "N/A (no successful nodes yet)"
        else:
            best_score_str = f"{branch_best_score:.4f}"
        prompt["Instructions"] |= {
            "🔥 Improvement Strategy: Magnitude-Based Reasoning": [
                "",
                "⚠️ **CRITICAL: The current approach has hit a plateau.**",
                "",
                "Do NOT just tweak parameters unless we are very close to the target.",
                "Classify your thoughts into these 3 Tiers based on the **Magnitude of Change**:",
                "",
                "**Tier 1: Optimization (The \"How\")**",
                "- Definition: Keep the model architecture and data fixed. Only change *how* we train.",
                "- Scope: Hyperparameters, Learning Rate Schedulers, Random Seeds, Post-processing thresholds.",
                "- *When to use: We are fine-tuning a working solution.*",
                "",
                "**Tier 2: Representation & Components (The \"What\")**",
                "- Definition: Change specific modules of the pipeline, but keep the overall paradigm.",
                "- Scope:",
                "    - **Data**: New feature engineering, different augmentations, input normalization.",
                "    - **Model**: Swapping the backbone (e.g., larger model), changing the loss function, adding regularization layers (Dropout/BN).",
                "- *When to use: The current model is underfitting or overfitting.*",
                "",
                "**Tier 3: Systemic Paradigm Shift (The \"Architecture\")**",
                "- Definition: Fundamentally change the approach. The old code structure might need a rewrite.",
                "- Scope:",
                "    - **Paradigm**: Switching from GBDT to Neural Net (or vice versa), Single Model -> Ensemble.",
                "    - **Objective**: Changing from Regression to Classification (binning), Multi-task learning.",
                "    - **Data Flow**: Pseudo-labeling, Self-supervised pre-training.",
                "- *When to use: The current approach has hit a hard ceiling (plateau).*",
                "",
                "**Current Status**:",
                f"- Best Score: {best_score_str}",
                f"- Successful nodes without improvement: {success_patience}",
                f"- Total nodes since best: {total_patience} (including failed attempts)",
                "",
                "**⚠️ CRITICAL INSTRUCTION**:",
                f"The branch has stagnated (success_patience={success_patience}, total_patience={total_patience}).",
                "You MUST propose a **Tier 2 or Tier 3** change to break the plateau.",
                "Do NOT propose Tier 1 (hyperparameter tuning). The current approach needs a more fundamental change.",
                "",
                "You can refer to the expert technique suggestions above, which are distilled from the kaggle award-winning solutions.",
                "",
                "After deciding your Tier 2/3 strategy, briefly describe:",
                "- Which Tier you're using and why",
                "- What specific components will change",
                "- Why this addresses the root cause of the plateau",
            ],
        }
    else:
        prompt["Instructions"] |= {
            "🔬 Critical: Scientific Approach to Optimization": [
                "",
                "⚠️ **MANDATORY FORMAT REQUIREMENT**",
                "You MUST structure your plan using the following EXACT format:",
                "",
                "CHANGES (list ALL modifications, one or multiple):",
                "",
                "Change #1: [Category: Data Augmentation / Model Architecture / Loss Function / Optimization / Regularization / Training Strategy]",
                "- What: [Describe the SPECIFIC technical modification you will make]",
                "- Why: [Explain why THIS TASK needs this specific change]",
                "",
                "Change #2 (if applicable): [Category]",
                "- What: [Describe the SPECIFIC technical modification]",
                "- Why: [Explain why THIS TASK needs this specific change]",
                "",
                "[Add more changes if needed, but keep them focused and related]",
                "",
                "---",
                "",
                "WHY current solution limited:",
                "- Root cause: [Specific analysis, not just 'low performance']",
                "- Evidence: [Data/observation that supports your diagnosis]",
                "",
                "HOW these changes address it:",
                "- Mechanism: [Theoretical justification of WHY this should work]",
                "- Expected improvement: [Concrete prediction]",
                "- Synergy (if multiple changes): [How changes work together, if applicable]",
                "",
                "KEEP UNCHANGED (must explicitly list):",
                "- Random seed: [specify value, e.g., 42]",
                "- Data split: [must be identical to parent]",
                "- [List other key components that remain unchanged]",
                "",
                "⚠️ Plans that do not follow this structure will be considered invalid.",
                "",
                "---",
                "",
                "**Guidelines on Number of Changes**:",
                "",
                "- **Single change (Recommended for most cases)**: Best for establishing clear causality",
                "  Example: \"Add [specific augmentation technique]\" → easy to attribute performance change",
                "",
                "- **Multiple related changes (Acceptable)**: When changes naturally work together",
                "  Example: \"Change model architecture + adjust corresponding hyperparameters\"",
                "  (architecture changes often require optimizer/lr adjustments)",
                "",
                "- **Fusion scenario (Acceptable)**: Combining proven improvements from Memory",
                "  Example: \"Integrate [technique from Attempt #X] + [technique from Attempt #Y]\"",
                "  (both already validated separately in Memory)",
                "",
                "⚠️ **Key Principle**: Whether single or multiple changes, you MUST:",
                "1. Clearly list each specific change",
                "2. Explain the rationale for each",
                "3. Specify what stays the same for proper baseline comparison",
                "",
                "---",
                "",
                "**Explanation of Requirements**:",
                "",
                "1. **WHY is the current solution limited?**",
                "   - Not just 'performance is low' - what is the ROOT CAUSE?",
                "   - What EVIDENCE supports your diagnosis?",
                "",
                "2. **HOW will your changes address this root cause?**",
                "   - Not just 'try method X' - explain the MECHANISM",
                "   - Why should this work? What is the theoretical justification?",
                "",
                "3. **WHAT will you change, and what will stay the same?**",
                "   - List ALL changes explicitly (even if multiple)",
                "   - Keep other things identical for proper baseline comparison",
                "   - This enables understanding WHAT led to performance changes",
                "",
                "---",
                "",
                "⚠️ This structured format enables proper performance tracking and knowledge accumulation.",
                "Random trial-and-error without clear documentation is not acceptable.",
                "Others will learn from your reasoning and can replicate your improvements.",
            ],
        }

    prompt["Instructions"] |= {
        "Solution improvement guidelines": [
            "- Propose a single, specific, actionable improvement (atomic change for controlled experiment).\n",
            "- Your improvement must be distinctly different from existing attempts in the Memory section.\n",
            "",
            "⚠️ **IMPORTANT: Depth of Improvement**",
            "Consider TWO types of improvements (both are valid, but think about which is more appropriate):",
            "",
            "**Type A: Architectural Deepening (Often More Powerful)**",
            "- ADD components to existing model (e.g., add relevant mechanisms for this task)",
            "- MODIFY internal structure (e.g., enhance feature extraction for task characteristics)",
            "- DESIGN task-specific modules based on domain knowledge and data patterns",
            "Example: 'Keep current backbone, but ADD [mechanism] to address [specific task challenge]'",
            "",
            "**Type B: Model/Method Replacement (Simpler but may miss potential)**",
            "- REPLACE entire model/algorithm with a different approach",
            "- This is valid when current architecture is fundamentally unsuitable for this task",
            "- But ask yourself: Could I improve the current approach by adding/modifying instead of replacing?",
            "",
            "- Your plan should be concise but comprehensive: Must address WHY/HOW/WHAT (2-4 sentences each). Avoid verbosity - every sentence should add new insight. Natural length: around 8-12 sentences for a complete reasoning process.\n",
            "- Don't suggest to do EDA.\n",
        ],
    }

    prompt["Instructions"] |= get_impl_guideline_from_agent(agent)
    prompt["Instructions"] |= prompt_leakage_prevention()
    internet_clarification = get_internet_clarification(getattr(agent.cfg, "pretrain_model_dir", ""))
    prompt["Instructions"]["Implementation guideline"].extend(internet_clarification)
    prompt["Instructions"] |= ROBUSTNESS_GENERALIZATION_STRATEGY

    output = wrap_code(parent_node.term_out, lang="")

    if not agent.acfg.use_diff_mode:
        prompt["Instructions"] |= prompt_resp_fmt()

    instructions = "\n# Instructions\n\n"
    instructions += compile_prompt_to_md(prompt["Instructions"], 2)

    memory_section = ""
    if prompt.get("Memory", "").strip():
        memory_section = f"\n# Memory\nBelow is a record of previous improvement attempts and their outcomes:\n {prompt['Memory']}\n"

    user_prompt = f"\n# Task description\n{prompt['Task description']}{memory_section}\n{instructions}"
    assistant_prefix = f"Let me approach this systematically.\nFirst, I'll review the dataset:\n{agent.data_preview}\nThe current solution uses the following code:\n{prompt['Previous solution']['Code']}\nIts output was:\n{output}\nBuilding on this, I'll develop an improved approach."
    prompt_complete = build_chat_prompt_for_model(agent.acfg.code.model, introduction, user_prompt, assistant_prefix)

    parent_node.add_expected_child_count()

    if agent.acfg.use_diff_mode:
        try:
            logger.info(f"Using diff improve for node {parent_node.id}")
            plan, code = _diff_improve(agent, prompt, agent.data_preview, parent_node)
        except Exception as e:
            logger.warning(f"Diff improve failed: {e}, falling back to full rewrite")
            plan, code = plan_and_code_query(agent, prompt_complete)
    else:
        plan, code = plan_and_code_query(agent, prompt_complete)

    from_topk = getattr(parent_node, '_topk_triggered', False)

    new_node = SearchNode(plan=plan, code=code, parent=parent_node, stage="improve",
                        local_best_node=parent_node.local_best_node, from_topk=from_topk)
    register_node(agent, new_node, prompt_complete, parent_node=parent_node)

    if hasattr(parent_node, '_topk_triggered'):
        parent_node._topk_triggered = False

    logger.info(f"[improve] {parent_node.id} → node {new_node.id}")
    return new_node


# ============ Diff improve pipeline ============

_IMPROVE_STAGE_INTRO = (
    "Based on the task requirements, data characteristics, and execution results, carefully analyze "
    "the current solution to identify improvement opportunities that will enhance the final test set "
    "performance. Then select which component(s) to modify and provide detailed, actionable modification plans."
)
_IMPROVE_EXTRA_GUIDELINE = (
    "1. **Analyze the task description and data type carefully** before proposing enhancements. "
    "Your improvements must be based on the current task."
)
_IMPROVE_PLANNER_TASK = build_planner_task(_IMPROVE_STAGE_INTRO, _IMPROVE_EXTRA_GUIDELINE)

_IMPROVE_DIFF_INTRODUCTION = (
    "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
    "solution and a detailed improvement plan. Your task is to implement the improvement plan to enhance "
    "the solution's test set performance."
)


_IMPROVE_SUFFIX_EXTRA = (
    "Building on the current solution, I'll develop an improved approach "
    "that addresses identified limitations while preserving what works well."
)


def _diff_improve(agent, prompt_base, data_preview, parent_node):
    context = {
        "stage": "improve",
        "memory": prompt_base["Memory"],
        "previous_code": parent_node.code,
        "previous_code_summary": parent_node.code_summary if hasattr(parent_node, 'code_summary') and parent_node.code_summary else None,
        "execution_output": parent_node.term_out if hasattr(parent_node, 'term_out') else "",
        "parent_node": parent_node,
    }

    use_memory = (
        getattr(agent.acfg, 'use_global_memory', False)
        and agent.global_memory is not None
        and len(agent.global_memory.records) > 0
    )

    if use_memory:
        logger.info("[DiffImprove] Using two-stage planning with memory")
        initial_plan = generate_initial_plan(agent, prompt_base, data_preview, context)
        planning_result = refine_plan_to_json(agent, initial_plan, prompt_base, data_preview, context)
    else:
        logger.info("[DiffImprove] Using direct planner (memory disabled or empty)")
        planning_result = run_planner(
            agent_instance=agent,
            prompt_base=prompt_base,
            data_preview=data_preview,
            context=context,
            your_task_section=_IMPROVE_PLANNER_TASK,
            assistant_suffix=build_planner_suffix(prompt_base, data_preview, context, extra_text=_IMPROVE_SUFFIX_EXTRA),
            stage_name="ImprovePlanning",
        )

    modules = planning_result.get('module', [])
    plans = planning_result.get('plan', {})

    if not planning_result.get("parse_success", False):
        raise RuntimeError("Planner returned empty result after retries, triggering outer fallback")

    if not modules and plans:
        planning_result['module'] = list(plans.keys())

    return diff_generate_and_apply(
        agent_instance=agent,
        planning_result=planning_result,
        parent_code=parent_node.code,
        data_preview=data_preview,
        execution_output=context["execution_output"],
        introduction=_IMPROVE_DIFF_INTRODUCTION,
    )
