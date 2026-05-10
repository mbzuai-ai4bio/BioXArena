"""Fusion Agent: merge solutions from other branches (fallback to improve if no candidates)."""

import logging
from typing import Any, List

from llm import compile_prompt_to_md
from engine.search_node import SearchNode
from utils.response import wrap_code
from agents.prompts import prompt_resp_fmt, get_impl_guideline_from_agent
from agents.improve_agent import run as run_improve
from agents.planner import run_planner, build_planner_task, build_chat_prompt_for_model
from agents.coder import plan_and_code_query
from agents.coder.diff_coder import diff_generate_and_apply
from engine import solution_manager
from agents.triggers import register_node

logger = logging.getLogger("MLEvolve")


def _get_fusion_candidates(agent, parent_node: SearchNode) -> List[SearchNode]:
    candidates = []

    for branch_id in agent.branch_successful_nodes.keys():
        if branch_id != parent_node.branch_id:
            branch_candidates = solution_manager.get_branch_top_nodes(agent,branch_id, top_k=2)
            candidates.extend(branch_candidates)

    if not candidates:
        current_branch_candidates = solution_manager.get_branch_top_nodes(agent,parent_node.branch_id, top_k=2)
        candidates = [node for node in current_branch_candidates if node.id != parent_node.id]

    logger.info(f"Found {len(candidates)} fusion candidates for node {parent_node.id}")
    return candidates


def fuse_two_nodes(agent, source_node: SearchNode, target_node: SearchNode) -> SearchNode:
    introduction = (
        "You are a Kaggle grandmaster attending a competition. "
        "You are provided with a successful reference solution from another approach below. "
        "Your task is to analyze this reference solution and improve your current solution by drawing inspiration from its strengths. "
        "First, outline a brief plan in natural language for how you will incorporate the best ideas, "
        "then implement this improved solution in Python."
    )

    reference_trajectory = target_node.generate_node_trajectory(need_code=True)

    prompt: Any = {
        "Introduction": introduction,
        "Task description": agent.task_desc,
        "Current Solution": {
            "Plan": source_node.plan,
            "Code": wrap_code(source_node.code),
            "Performance": source_node.metric.value if source_node.metric else 'N/A',
            "Analysis": source_node.analysis if source_node.analysis else 'N/A'
        },
        "Reference Solution": reference_trajectory,
        "Instructions": {},
    }

    prompt["Instructions"] |= {
        "🔬 Critical: Scientific Approach to Fusion": [
            "",
            "⚠️ **MANDATORY FORMAT REQUIREMENT**",
            "You MUST structure your plan using the following EXACT format:",
            "",
            "```",
            "CHANGES (list ALL modifications, one or multiple):",
            "",
            "Change #1: [Category: Data Augmentation / Model Architecture / Loss Function / Optimization / Regularization / Training Strategy]",
            "- What: [Describe the SPECIFIC technique you will incorporate from reference]",
            "- Why: [Based on reference analysis, explain why THIS TASK needs this specific change]",
            "",
            "Change #2 (if applicable): [Category]",
            "- What: [Describe the SPECIFIC technique]",
            "- Why: [Based on reference analysis, explain why THIS TASK needs this specific change]",
            "",
            "[Add more changes if needed, but keep them focused and related]",
            "",
            "---",
            "",
            "WHY current solution limited (and reference succeeded):",
            "- Root cause: [What limitation does your solution have that reference addressed?]",
            "- Evidence: [Specific comparison showing reference's advantage]",
            "",
            "HOW reference techniques apply to MY solution:",
            "- Mechanism: [Why should their technique work in your context?]",
            "- Compatibility: [How does it fit with your existing architecture?]",
            "- Expected improvement: [Concrete prediction based on reference's success]",
            "",
            "KEEP UNCHANGED (must explicitly list):",
            "- Random seed: [specify value, e.g., 42]",
            "- Data split: [must be identical to parent]",
            "- Core architecture: [your base model/framework that stays intact]",
            "```",
            "",
            "⚠️ Plans that do not follow this structure will be considered invalid.",
            "",
            "---",
            "",
            "**Guidelines on Reference-Based Fusion**:",
            "",
            "- **Selective adoption**: Don't copy everything - choose techniques that address YOUR limitations",
            "- **Understand mechanisms**: Why did it work for them? Will it work for you?",
            "- **Preserve your strengths**: Keep what's working in your solution",
            "- **Avoid blind combination**: Fusion ≠ pasting their code into yours",
            "",
            "⚠️ **Key Principle**: Fusion means understanding WHY techniques work, not blindly copying.",
            "Reference solutions provide ideas, not templates to copy.",
            "",
            "---",
            "",
            "⚠️ This structured format enables proper performance tracking and knowledge accumulation.",
            "Your reasoning should clearly show how you analyzed the reference and adapted it.",
        ],
    }

    prompt["Instructions"] |= {
        "Two-solution improvement guideline": [
            "- Analyze the reference solution and identify its key strengths and successful techniques.",
            "- Focus on understanding WHY the reference solution succeeded and HOW those techniques apply to your context.",
            "- Create an improved solution that builds upon your current approach while selectively incorporating ONE promising technique from the reference solution.",
            "- The improvement should be thoughtful and selective, choosing only techniques that address a specific limitation in your solution rather than simply combining approaches.",
            "- Your plan should be concise but comprehensive, naturally reflecting your reasoning process (WHY reference succeeded, HOW it applies to you, WHAT you'll incorporate).",
            "- The final code should be a single, runnable Python script.",
            "- Do not suggest to do EDA.",
        ],
    }
    prompt["Instructions"] |= get_impl_guideline_from_agent(agent)

    if not agent.acfg.use_diff_mode:
        prompt["Instructions"] |= prompt_resp_fmt()

    instructions = "\n# Instructions\n\n"
    instructions += compile_prompt_to_md(prompt["Instructions"], 2)

    user_prompt = f"\n# Task description\n{prompt['Task description']}\n\n# Reference Solution\n{prompt['Reference Solution']}\n\n{instructions}"
    assistant_prefix = f"Let me approach this systematically.\nFirst, I'll review the dataset:\n{agent.data_preview}\nMy current solution:\nPlan: {prompt['Current Solution']['Plan']}\nCode: {prompt['Current Solution']['Code']}\nPerformance: {prompt['Current Solution']['Performance']}\nAnalysis: {prompt['Current Solution']['Analysis']}\nI'll now analyze the reference solution and selectively incorporate its best ideas."
    prompt_complete = build_chat_prompt_for_model(agent.acfg.code.model, introduction, user_prompt, assistant_prefix)

    if agent.acfg.use_diff_mode:
        try:
            logger.info(f"Using diff fusion for node {source_node.id} with reference {target_node.id}")
            plan, code = _diff_fusion(agent, prompt, agent.data_preview, source_node)
        except Exception as e:
            logger.warning(f"Diff fusion failed: {e}, falling back to full fusion")
            plan, code = plan_and_code_query(agent, prompt_complete)
    else:
        plan, code = plan_and_code_query(agent, prompt_complete)

    from_topk = getattr(source_node, '_topk_triggered', False)

    fused_node = SearchNode(
        plan=plan,
        code=code,
        parent=source_node,
        stage="fusion",
        local_best_node=source_node.local_best_node,
        from_topk=from_topk
    )
    register_node(agent, fused_node, prompt_complete, parent_node=source_node)

    if hasattr(source_node, '_topk_triggered'):
        source_node._topk_triggered = False

    logger.info(f"[fusion] {source_node.id} → node {fused_node.id}")
    return fused_node


def _fuse_with_multiple_references(
    agent, parent_node: SearchNode, reference_nodes: List[SearchNode]
) -> SearchNode:
    introduction = (
        "You are a Kaggle grandmaster attending a competition. "
        "You are provided with multiple successful solutions from different approaches below. "
        "Your task is to analyze these reference solutions and improve your current solution by drawing inspiration from their strengths. "
        "First, outline a brief plan in natural language for how you will incorporate the best ideas, "
        "then implement this improved solution in Python."
    )

    reference_summaries = []
    for node in reference_nodes:
        trajectory = node.generate_node_trajectory(need_code=False)
        reference_summaries.append(trajectory)
    reference_memory = "\n-------------------------------\n".join(reference_summaries)

    prompt: Any = {
        "Introduction": introduction,
        "Task description": agent.task_desc,
        "Current Solution": {
            "Plan": parent_node.plan,
            "Code": wrap_code(parent_node.code),
            "Performance": parent_node.metric.value if parent_node.metric else 'N/A',
            "Analysis": parent_node.analysis if parent_node.analysis else 'N/A'
        },
        "Reference Solutions": reference_memory,
        "Instructions": {},
    }

    prompt["Instructions"] |= {
        "🔬 Critical: Scientific Approach to Multi-Reference Fusion": [
            "",
            "⚠️ **MANDATORY FORMAT REQUIREMENT**",
            "You MUST structure your plan using the following EXACT format:",
            "",
            "```",
            "CHANGES (list ALL modifications, one or multiple):",
            "",
            "Change #1: [Category: Data Augmentation / Model Architecture / Loss Function / Optimization / Regularization / Training Strategy]",
            "- What: [Describe the SPECIFIC technique from references you will incorporate]",
            "- Why: [Based on multi-reference analysis, explain why THIS TASK needs this specific change]",
            "- Source: [Which reference(s) inspired this change]",
            "",
            "Change #2 (if applicable): [Category]",
            "- What: [Describe the SPECIFIC technique]",
            "- Why: [Based on multi-reference analysis, explain why THIS TASK needs this specific change]",
            "- Source: [Which reference(s) inspired this change]",
            "",
            "[Add more changes if needed, but keep them focused and related]",
            "",
            "---",
            "",
            "WHY current solution limited (and which reference addressed it best):",
            "- Root cause: [What limitation does your solution have?]",
            "- Evidence: [Comparison across references showing which approach works best]",
            "- Best reference: [Which reference most effectively addressed this limitation]",
            "",
            "HOW selected techniques apply to MY solution:",
            "- Mechanism: [Why should this technique work in your context?]",
            "- Compatibility: [How does it fit with your existing architecture?]",
            "- Expected improvement: [Concrete prediction based on references' success]",
            "",
            "KEEP UNCHANGED (must explicitly list):",
            "- Random seed: [specify value, e.g., 42]",
            "- Data split: [must be identical to parent]",
            "- Core architecture: [your base model/framework that stays intact]",
            "```",
            "",
            "⚠️ Plans that do not follow this structure will be considered invalid.",
            "",
            "---",
            "",
            "**Guidelines on Multi-Reference Fusion**:",
            "",
            "- **Comparative analysis**: Why did different references succeed? Which approach is most relevant?",
            "- **Best technique selection**: More references → better choice, not more techniques to combine",
            "- **Avoid feature combination**: Don't try to use techniques from all references",
            "- **Focus on YOUR needs**: Which reference best addresses YOUR specific limitation?",
            "",
            "⚠️ **Key Principle**: More references means better understanding of which ONE technique to adopt.",
            "Synthesize knowledge to make the best choice, don't combine everything.",
            "",
            "---",
            "",
            "⚠️ This structured format enables proper performance tracking and knowledge accumulation.",
            "Your reasoning should clearly show how you compared references and chose the best approach.",
        ],
    }

    prompt["Instructions"] |= {
        "Multi-solution improvement guideline": [
            "- Analyze all reference solutions and identify their key strengths and successful techniques.",
            "- Focus on understanding WHY each reference succeeded and HOW those techniques apply to your context.",
            "- Create an improved solution that builds upon your current approach while selectively incorporating ONE promising technique from the most relevant reference.",
            "- The improvement should be thoughtful and selective, choosing the single technique that best addresses a specific limitation in your solution rather than combining multiple approaches.",
            "- Your plan should be concise but comprehensive, naturally reflecting your reasoning process (WHY each reference succeeded, HOW they apply to you, WHAT you'll incorporate).",
            "- The final code should be a single, runnable Python script.",
            "- Do not suggest to do EDA.",
        ],
    }
    prompt["Instructions"] |= get_impl_guideline_from_agent(agent)

    if not agent.acfg.use_diff_mode:
        prompt["Instructions"] |= prompt_resp_fmt()

    instructions = "\n# Instructions\n\n"
    instructions += compile_prompt_to_md(prompt["Instructions"], 2)

    user_prompt = f"\n# Task description\n{prompt['Task description']}\n\n# Reference Solutions\n{prompt['Reference Solutions']}\n\n{instructions}"
    assistant_prefix = f"Let me approach this systematically.\nFirst, I'll review the dataset:\n{agent.data_preview}\nMy current solution:\nPlan: {prompt['Current Solution']['Plan']}\nCode: {prompt['Current Solution']['Code']}\nPerformance: {prompt['Current Solution']['Performance']}\nAnalysis: {prompt['Current Solution']['Analysis']}\nI'll now analyze the reference solutions and selectively incorporate the best ideas."
    prompt_complete = build_chat_prompt_for_model(agent.acfg.code.model, introduction, user_prompt, assistant_prefix)

    if agent.acfg.use_diff_mode:
        try:
            logger.info(f"Using diff multi-fusion for node {parent_node.id} with {len(reference_nodes)} references")
            plan, code = _diff_multi_fusion(agent, prompt, agent.data_preview, parent_node)
        except Exception as e:
            logger.warning(f"Diff multi-fusion failed: {e}, falling back to full rewrite")
            plan, code = plan_and_code_query(agent, prompt_complete)
    else:
        plan, code = plan_and_code_query(agent, prompt_complete)

    from_topk = getattr(parent_node, '_topk_triggered', False)

    fused_node = SearchNode(
        plan=plan,
        code=code,
        parent=parent_node,
        stage="fusion",
        local_best_node=parent_node.local_best_node,
        from_topk=from_topk
    )
    register_node(agent, fused_node, prompt_complete, parent_node=parent_node)

    if hasattr(parent_node, '_topk_triggered'):
        parent_node._topk_triggered = False

    logger.info(f"[fusion] {parent_node.id} → node {fused_node.id}")
    return fused_node


def run(agent, parent_node: SearchNode) -> SearchNode:
    candidates = _get_fusion_candidates(agent, parent_node)

    if not candidates:
        logger.info(f"No fusion candidates found for node {parent_node.id}, falling back to normal improve")
        return run_improve(agent, parent_node)

    if len(candidates) == 1:
        fused_node = fuse_two_nodes(agent, parent_node, candidates[0])
        parent_node.add_expected_child_count()
        return fused_node

    elif len(candidates) <= 5:
        fused_node = _fuse_with_multiple_references(agent, parent_node, candidates)
        parent_node.add_expected_child_count()
        return fused_node

    else:
        top_5_candidates = candidates[:5]
        fused_node = _fuse_with_multiple_references(agent, parent_node, top_5_candidates)
        parent_node.add_expected_child_count()
        return fused_node


# ============ Diff fusion pipeline ============

_FUSION_STAGE_INTRO = (
    "Based on the task requirements, data characteristics, and the **reference solution**, analyze your current "
    "solution to identify improvement opportunities. Learn from the reference solution's strengths to make "
    "targeted enhancements."
)
_FUSION_EXTRA_GUIDELINE = (
    "1. **Analyze the reference solution carefully** to understand what techniques made it successful. "
    "Focus on selective adoption - choose techniques that address YOUR specific limitations."
)
_FUSION_PLANNER_TASK = build_planner_task(_FUSION_STAGE_INTRO, _FUSION_EXTRA_GUIDELINE)

_MULTI_FUSION_STAGE_INTRO = (
    "Based on the task requirements, data characteristics, and **multiple reference solutions**, analyze your "
    "current solution to identify improvement opportunities. Compare the references to understand which "
    "technique is most relevant to YOUR specific limitations."
)
_MULTI_FUSION_EXTRA_GUIDELINE = (
    "1. **Compare all reference solutions** to understand why different approaches succeeded. "
    "Best technique selection - more references means better understanding, not more techniques to combine."
)
_MULTI_FUSION_PLANNER_TASK = build_planner_task(_MULTI_FUSION_STAGE_INTRO, _MULTI_FUSION_EXTRA_GUIDELINE)

_FUSION_DIFF_INTRODUCTION = (
    "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
    "solution and a detailed improvement plan based on analysis of a reference solution. Your task is to "
    "implement the improvement plan, selectively incorporating the best ideas from the reference."
)

_MULTI_FUSION_DIFF_INTRODUCTION = (
    "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
    "solution and a detailed improvement plan based on comparative analysis of multiple reference solutions. "
    "Your task is to implement the improvement plan, selectively incorporating the best ideas from the "
    "most relevant reference."
)


def _build_fusion_planner_suffix(prompt_base, data_preview, context):
    current = prompt_base.get("Current Solution", {})
    return (
        f"Let me approach this systematically.\n"
        f"First, I'll review the dataset:\n{data_preview}\n"
        f"My current solution:\n"
        f"Plan: {current.get('Plan', 'N/A')}\n"
        f"Code: {current.get('Code', 'N/A')}\n"
        f"Performance: {current.get('Performance', 'N/A')}\n"
        f"Analysis: {current.get('Analysis', 'N/A')}\n"
        f"I'll analyze the reference solution and selectively incorporate the best ideas. "
        f"Now I will output my analysis in JSON format only (no additional text):"
    )


def _build_multi_fusion_planner_suffix(prompt_base, data_preview, context):
    current = prompt_base.get("Current Solution", {})
    return (
        f"Let me approach this systematically.\n"
        f"First, I'll review the dataset:\n{data_preview}\n"
        f"My current solution:\n"
        f"Plan: {current.get('Plan', 'N/A')}\n"
        f"Code: {current.get('Code', 'N/A')}\n"
        f"Performance: {current.get('Performance', 'N/A')}\n"
        f"Analysis: {current.get('Analysis', 'N/A')}\n"
        f"I'll compare the reference solutions and selectively incorporate the best ideas. "
        f"Now I will output my analysis in JSON format only (no additional text):"
    )


def _diff_fusion(agent, prompt_base, data_preview, source_node):
    reference_solution = prompt_base.get("Reference Solution", "")

    context = {
        "stage": "fusion",
        "current_code": source_node.code,
        "current_plan": source_node.plan,
        "current_performance": source_node.metric.value if source_node.metric else 'N/A',
        "current_analysis": source_node.analysis if source_node.analysis else 'N/A',
        "reference_solution": reference_solution,
    }

    planning_result = run_planner(
        agent_instance=agent,
        prompt_base=prompt_base,
        data_preview=data_preview,
        context=context,
        your_task_section=_FUSION_PLANNER_TASK,
        assistant_suffix=_build_fusion_planner_suffix(prompt_base, data_preview, context),
        stage_name="FusionPlanning",
    )

    modules = planning_result.get('module', [])
    plans = planning_result.get('plan', {})

    if not planning_result.get("parse_success", False):
        raise RuntimeError("Fusion planner returned empty result, triggering outer fallback")

    if not modules and plans:
        planning_result['module'] = list(plans.keys())

    extra_context = ""
    if reference_solution:
        extra_context = (
            f"I also have access to a reference solution from another successful approach:\n"
            f"{reference_solution}\n"
            f"I will selectively incorporate the best ideas from this reference."
        )

    return diff_generate_and_apply(
        agent_instance=agent,
        planning_result=planning_result,
        parent_code=source_node.code,
        data_preview=data_preview,
        execution_output="",
        introduction=_FUSION_DIFF_INTRODUCTION,
        extra_context=extra_context,
    )


def _diff_multi_fusion(agent, prompt_base, data_preview, parent_node):
    reference_solutions = prompt_base.get("Reference Solutions", "")

    context = {
        "stage": "multi_fusion",
        "current_code": parent_node.code,
        "current_plan": parent_node.plan,
        "current_performance": parent_node.metric.value if parent_node.metric else 'N/A',
        "current_analysis": parent_node.analysis if parent_node.analysis else 'N/A',
        "reference_solutions": reference_solutions,
    }

    planning_result = run_planner(
        agent_instance=agent,
        prompt_base=prompt_base,
        data_preview=data_preview,
        context=context,
        your_task_section=_MULTI_FUSION_PLANNER_TASK,
        assistant_suffix=_build_multi_fusion_planner_suffix(prompt_base, data_preview, context),
        stage_name="MultiFusionPlanning",
    )

    modules = planning_result.get('module', [])
    plans = planning_result.get('plan', {})

    if not planning_result.get("parse_success", False):
        raise RuntimeError("Multi-fusion planner returned empty result, triggering outer fallback")

    if not modules and plans:
        planning_result['module'] = list(plans.keys())

    extra_context = ""
    if reference_solutions:
        extra_context = (
            f"I also have access to multiple reference solutions from different successful approaches:\n"
            f"{reference_solutions}\n"
            f"I will compare them and selectively incorporate the best ideas from the most relevant reference."
        )

    return diff_generate_and_apply(
        agent_instance=agent,
        planning_result=planning_result,
        parent_code=parent_node.code,
        data_preview=data_preview,
        execution_output="",
        introduction=_MULTI_FUSION_DIFF_INTRODUCTION,
        extra_context=extra_context,
    )
