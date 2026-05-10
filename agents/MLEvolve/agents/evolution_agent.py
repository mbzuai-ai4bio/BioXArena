"""Evolution Agent: improve using branch evolution trajectory."""

import re
import logging

from typing import Any

from llm import compile_prompt_to_md
from engine.search_node import SearchNode
from utils.response import wrap_code
from agents.triggers import get_patience_counter
from agents.prompts import (
    ROBUSTNESS_GENERALIZATION_STRATEGY,
    prompt_resp_fmt,
    get_impl_guideline_from_agent,
)
from agents.improve_agent import run as run_improve
from agents.planner import run_planner, build_planner_task, build_planner_suffix, build_chat_prompt_for_model
from agents.coder import plan_and_code_query
from agents.coder.diff_coder import diff_generate_and_apply
from agents.triggers import register_node

logger = logging.getLogger("MLEvolve")


def _get_branch_trajectory_for_evolution(parent_node: SearchNode) -> str:
    if not hasattr(parent_node, 'branch_id') or not parent_node.branch_id:
        return "No branch trajectory available."

    trajectory = parent_node.get_root_to_current_trajectory(max_steps=10)

    if not trajectory or trajectory.strip() == "":
        return "No evolution history available in current branch."

    step_count = len(re.findall(r'^Step \d+:', trajectory, re.MULTILINE))

    if step_count < 2:
        return f"Insufficient evolution history (only {step_count} step(s)). Need at least 2 steps to learn from trajectory."

    return f"Your Past Evolution trajectory:\n{trajectory}"


def run(agent, parent_node: SearchNode) -> SearchNode:
    branch_trajectory = _get_branch_trajectory_for_evolution(parent_node)

    if "Insufficient evolution history" in branch_trajectory or "No evolution history" in branch_trajectory or "No branch trajectory" in branch_trajectory:
        logger.info(f"Insufficient trajectory history for evolution, falling back to normal improve for node {parent_node.id}")
        return run_improve(agent, parent_node)

    introduction = (
        "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
        "solution below and should improve it in order to further increase the (test time) performance. "
        "For this you should first outline a brief plan in natural language for how the solution can be improved and "
        "then implement this improvement in Python based on the provided previous solution. "
    )

    prompt: Any = {
        "Introduction": introduction,
        "Task description": agent.task_desc,
        "Memory": parent_node.fetch_child_memory(),
        "Branch Evolution History": branch_trajectory,
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
        logger.warning(f"🔥 EVOLUTION PLATEAU DETECTED! Triggered by: {' AND '.join(trigger_reason)}, using Magnitude-Based prompt")
        if branch_best_score is None:
            best_score_str = "N/A (no successful nodes yet)"
        else:
            best_score_str = f"{branch_best_score:.4f}"

        prompt["Instructions"] |= {
            "🔥 Evolution Strategy: Magnitude-Based Reasoning": [
                "",
                "⚠️ **CRITICAL: The evolution trajectory shows a plateau.**",
                "",
                "Do NOT just tweak parameters. Classify your change into these 3 Tiers:",
                "",
                "**Tier 1: Optimization (The \"How\")**",
                "- Keep model architecture and data fixed. Only change training details.",
                "- Examples: Hyperparameters, learning rate schedules, random seeds.",
                "",
                "**Tier 2: Representation & Components (The \"What\")**",
                "- Change specific modules while keeping the overall paradigm.",
                "- Examples: Swap backbone, change loss function, add regularization.",
                "",
                "**Tier 3: Systemic Paradigm Shift (The \"Architecture\")**",
                "- Fundamentally change the approach based on trajectory insights.",
                "- Examples: Paradigm shift (GBDT→NN), ensemble model, pseudo-labeling.",
                "",
                "**Current Status**:",
                f"- Best Score: {best_score_str}",
                f"- Successful nodes without improvement: {success_patience}",
                f"- Total nodes since best: {total_patience} (including failed attempts)",
                "",
                "**⚠️ CRITICAL INSTRUCTION**:",
                f"The branch has stagnated (success_patience={success_patience}, total_patience={total_patience}).",
                "You MUST propose a **Tier 2 or Tier 3** change to break the plateau.",
                "Analyze the evolution trajectory to identify WHY progress stopped, then make a fundamental change.",
                "",
                "Briefly describe:",
                "- Which Tier you're using",
                "- What trajectory pattern led to this choice",
                "- What specific change will break the plateau",
            ],
        }
    else:
        prompt["Instructions"] |= {
            "🔬 Critical: Scientific Approach to Evolution": [
                "",
                "⚠️ **MANDATORY FORMAT REQUIREMENT**",
                "You MUST structure your plan using the following EXACT format:",
                "",
                "```",
                "CHANGES (list ALL modifications, one or multiple):",
                "",
                "Change #1: [Category: Data Augmentation / Model Architecture / Loss Function / Optimization / Regularization / Training Strategy]",
                "- What: [Describe the SPECIFIC technical modification you will make]",
                "- Why: [Based on trajectory, explain why THIS TASK needs this specific change]",
                "",
                "Change #2 (if applicable): [Category]",
                "- What: [Describe the SPECIFIC technical modification]",
                "- Why: [Based on trajectory, explain why THIS TASK needs this specific change]",
                "",
                "[Add more changes if needed, but keep them focused and related]",
                "",
                "---",
                "",
                "WHY current solution limited (based on trajectory analysis):",
                "- Root cause: [What patterns from evolution history reveal the limitation?]",
                "- Evidence: [Specific steps/metrics from trajectory that support this diagnosis]",
                "",
                "HOW these changes build on learned patterns:",
                "- Mechanism: [How does this apply lessons from successful/failed steps in trajectory?]",
                "- Expected improvement: [Concrete prediction based on trajectory patterns]",
                "- Trajectory insight: [What specific lesson from evolution history guides this choice?]",
                "",
                "KEEP UNCHANGED (must explicitly list):",
                "- Random seed: [specify value, e.g., 42]",
                "- Data split: [must be identical to parent]",
                "- Successful elements from trajectory: [list what worked and should be preserved]",
                "```",
                "",
                "⚠️ Plans that do not follow this structure will be considered invalid.",
                "",
                "---",
                "",
                "**Guidelines on Learning from Trajectory**:",
                "",
                "- **Identify patterns**: Look for what worked (e.g., \"Step 2→3: augmentation +5%\") and what didn't",
                "- **Build on success**: If technique X improved performance, consider enhancing it further",
                "- **Avoid repeated failures**: If approach Y failed multiple times, try a different direction",
                "- **Extract root causes**: Why did certain changes work? Apply that understanding to new changes",
                "",
                "⚠️ **Key Principle**: Evolution means learning from history, not random mutation.",
                "Use trajectory evidence to guide your changes, not gut feeling.",
                "",
                "---",
                "",
                "⚠️ This structured format enables proper performance tracking and knowledge accumulation.",
                "Your reasoning should clearly show how you learned from the evolution trajectory.",
            ],
        }

    prompt["Instructions"] |= {
        "Solution improvement sketch guideline": [
            "- Propose a single, specific, actionable improvement (atomic change for controlled experiment).\n",
            "- When proposing the design, take the Memory section into account.\n",
            "- Your improvement must be distinctly different from existing attempts in the Memory section.\n",
            "- Pay special attention to the Branch Evolution History section, which shows the evolution path of your current approach. From this historical trajectory, extract both successful patterns and failed experiences to guide your improvement strategy.\n",
            "- Your plan should be concise but comprehensive, naturally reflecting your reasoning process (WHY previous changes worked/failed, HOW you'll build on that, WHAT you'll change).\n",
            "- Don't suggest to do EDA.\n",
        ],
    }
    prompt["Instructions"] |= ROBUSTNESS_GENERALIZATION_STRATEGY
    prompt["Instructions"] |= get_impl_guideline_from_agent(agent)
    output = wrap_code(parent_node.term_out, lang="")

    if not agent.acfg.use_diff_mode:
        prompt["Instructions"] |= prompt_resp_fmt()

    instructions = "\n# Instructions\n\n"
    instructions += compile_prompt_to_md(prompt["Instructions"], 2)

    memory_section = ""
    if prompt.get("Memory", "").strip():
        memory_section = f"\n# Memory\nBelow is a record of previous improvement attempts and their outcomes:\n {prompt['Memory']}\n"

    user_prompt = f"\n# Task description\n{prompt['Task description']}{memory_section}{prompt['Branch Evolution History']}\n\n{instructions}"
    assistant_prefix = f"Let me approach this systematically.\nFirst, I'll review the dataset:\n{agent.data_preview}\nThe current solution uses the following code:\n{prompt['Previous solution']['Code']}\nIts output was:\n{output}\nBuilding on this and my evolution trajectory, I'll develop an improved approach."
    prompt_complete = build_chat_prompt_for_model(agent.acfg.code.model, introduction, user_prompt, assistant_prefix)

    parent_node.add_expected_child_count()

    if agent.acfg.use_diff_mode:
        try:
            logger.info(f"Using diff evolution for node {parent_node.id}")
            plan, code = _diff_evolution(agent, prompt, agent.data_preview, parent_node)
        except Exception as e:
            logger.warning(f"Diff evolution failed: {e}, falling back to full evolution")
            plan, code = plan_and_code_query(agent, prompt_complete)
    else:
        plan, code = plan_and_code_query(agent, prompt_complete)

    from_topk = getattr(parent_node, '_topk_triggered', False)

    new_node = SearchNode(plan=plan, code=code, parent=parent_node, stage="evolution",
                        local_best_node=parent_node.local_best_node, from_topk=from_topk)
    register_node(agent, new_node, prompt_complete, parent_node=parent_node)

    if hasattr(parent_node, '_topk_triggered'):
        parent_node._topk_triggered = False

    logger.info(f"[evolution] {parent_node.id} → node {new_node.id}")
    return new_node


# ============ Diff evolution pipeline ============

_EVOLUTION_STAGE_INTRO = (
    "Based on the task requirements, data characteristics, execution results, and **evolution trajectory**, "
    "analyze the current solution to identify improvement opportunities. Learn from the historical trajectory "
    "(successful patterns and failed approaches) to make more informed improvement decisions."
)
_EVOLUTION_EXTRA_GUIDELINE = (
    "1. **Analyze the evolution trajectory carefully** to understand what has been tried and what worked/failed. "
    "Build on successful patterns and avoid repeating failed approaches."
)
_EVOLUTION_PLANNER_TASK = build_planner_task(_EVOLUTION_STAGE_INTRO, _EVOLUTION_EXTRA_GUIDELINE)

_EVOLUTION_DIFF_INTRODUCTION = (
    "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
    "solution and a detailed improvement plan based on evolution trajectory analysis. Your task is to "
    "implement the improvement plan, learning from the evolution trajectory to enhance the solution's "
    "test set performance."
)


def _build_evolution_suffix_extra(context):
    branch_history = context.get("branch_evolution_history", "")
    if not branch_history:
        return ""
    return (
        f"I also have access to my evolution trajectory:\n{branch_history}\n"
        f"I will analyze this trajectory to identify successful patterns and failed approaches, "
        f"then use these insights to guide my improvement plan."
    )


def _diff_evolution(agent, prompt_base, data_preview, parent_node):
    branch_history = prompt_base.get("Branch Evolution History", "")

    context = {
        "stage": "evolution",
        "memory": prompt_base.get("Memory", ""),
        "previous_code": parent_node.code,
        "execution_output": parent_node.term_out if hasattr(parent_node, 'term_out') else "",
        "branch_evolution_history": branch_history,
    }

    planning_result = run_planner(
        agent_instance=agent,
        prompt_base=prompt_base,
        data_preview=data_preview,
        context=context,
        your_task_section=_EVOLUTION_PLANNER_TASK,
        assistant_suffix=build_planner_suffix(prompt_base, data_preview, context, extra_text=_build_evolution_suffix_extra(context)),
        stage_name="EvolutionPlanning",
    )

    modules = planning_result.get('module', [])
    plans = planning_result.get('plan', {})

    if not planning_result.get("parse_success", False):
        raise RuntimeError("Evolution planner returned empty result, triggering outer fallback")

    if not modules and plans:
        planning_result['module'] = list(plans.keys())

    extra_context = ""
    if branch_history:
        extra_context = (
            f"I also have access to my past evolution trajectory:\n{branch_history}\n"
            f"I will analyze this trajectory to identify successful patterns and failed approaches, "
            f"then use these insights to make more informed improvements."
        )

    return diff_generate_and_apply(
        agent_instance=agent,
        planning_result=planning_result,
        parent_code=parent_node.code,
        data_preview=data_preview,
        execution_output=context["execution_output"],
        introduction=_EVOLUTION_DIFF_INTRODUCTION,
        extra_context=extra_context,
        learning_guidance="Learn from evolution trajectory - use the evolution history to guide your changes. Build on successful patterns and avoid repeating failed approaches from the trajectory.",
    )
