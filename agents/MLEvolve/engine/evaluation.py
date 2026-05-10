"""Node evaluation: backpropagate, check_improvement, get_node_reward."""

import logging
import time
import random

from engine.search_node import SearchNode

logger = logging.getLogger("MLEvolve")


def backpropagate(node: SearchNode, value: float, add_to_tree=True):
    """Propagate reward up the tree; update debug_success, continue_improve, lock."""
    logger.info(f"[backprop] node {node.id}, reward={value}")
    while node is not None:
        if node.parent and node.is_buggy is False and node.parent.is_buggy is True:
            node.parent.is_debug_success = True
        elif node.parent and node.is_buggy is True and node.is_debug_success is True and node.parent.is_buggy is True:
            node.parent.is_debug_success = True
        if node.parent and node.parent.stage != "root":
            node.parent.continue_improve = node.continue_improve
        if node.stage in ["draft", "fusion_draft"] and node.lock:
            node.lock = False
        if node.improve_failure_depth > 0:
            node.improve_failure_depth = 0
        node.update(value, add_to_tree)
        node = node.parent


def get_node_reward(agent, node: SearchNode):
    reward = 0

    if node.is_buggy is True or node.is_buggy is None:
        reward = -1
    elif node.is_buggy is False and node.metric.value is None:
        reward = -1
    else:
        if node.metric.value is not None and agent.best_metric is not None:
            improvement = node.metric.value - agent.best_metric if node.metric.maximize else agent.best_metric - node.metric.value
            if improvement > 0:
                logger.info(f"Node {node.id} is better than the best node {agent.best_node.id} now!")
                reward += 1.5

        if node.parent and node.parent.stage != "root":
            if node.parent.is_buggy is True:
                reward += 1.5
            else:
                reward += 1
    return reward


def check_improvement(agent, cur_node: SearchNode, parent_node: SearchNode):

    improvement = 0
    should_backpropagate = False

    if (agent.search_start_time and
        cur_node.stage != "root" and
        cur_node.branch_id is not None):

        time_elapsed = time.time() - agent.search_start_time
        time_progress = time_elapsed / agent.acfg.time_limit

        if not hasattr(agent, 'branch_node_count'):
            agent.branch_node_count = {}

        branch_id = cur_node.branch_id
        agent.branch_node_count[branch_id] = agent.branch_node_count.get(branch_id, 0) + 1
        current_count = agent.branch_node_count[branch_id]

        force_backprop = False

        scfg = agent.scfg

        if time_progress >= scfg.force_backprop_late_threshold:
            if random.random() < scfg.force_backprop_late_prob:
                force_backprop = True
                logger.info(f"[Force Backprop] Late stage ({time_progress:.1%}), "
                        f"node {cur_node.id} (stage={cur_node.stage}, branch={branch_id}, #{current_count})")

        elif time_progress >= scfg.force_backprop_mid_threshold and current_count % scfg.force_backprop_mid_modulo == 0:
            force_backprop = True
            logger.info(f"[Force Backprop] Mid stage ({time_progress:.1%}), "
                       f"branch {branch_id} node #{current_count}, "
                       f"node {cur_node.id} (stage={cur_node.stage})")

        if force_backprop:
            skip_force_backprop = False

            if (not cur_node.is_buggy and
                cur_node.metric is not None and
                cur_node.metric.value is not None):

                recent_window = scfg.recent_best_window
                recent_nodes = [n for n in agent.journal[-recent_window:]
                               if (not n.is_buggy and n.metric and n.metric.value is not None)]

                if recent_nodes:
                    if cur_node.metric.maximize:
                        recent_best = max(recent_nodes, key=lambda n: n.metric.value)
                        is_recent_best = cur_node.metric.value >= recent_best.metric.value
                    else:
                        recent_best = min(recent_nodes, key=lambda n: n.metric.value)
                        is_recent_best = cur_node.metric.value <= recent_best.metric.value

                    if is_recent_best:
                        logger.info(f"[Smart Backprop] Node {cur_node.id} is recent best "
                                  f"(metric={cur_node.metric.value:.4f}), skip force backprop to continue improvement chain")
                        skip_force_backprop = True

            if not skip_force_backprop:
                if (not cur_node.is_buggy and
                    cur_node.metric is not None and
                    cur_node.metric.value is not None):

                    local_best = cur_node.local_best_node
                    if local_best and local_best.metric and local_best.metric.value is not None:
                        if agent.metric_maximize:
                            is_better = cur_node.metric.value > local_best.metric.value
                        else:
                            is_better = cur_node.metric.value < local_best.metric.value

                        if is_better:
                            cur_node.local_best_node = cur_node
                            logger.info(f"  └─ Updated local_best: {cur_node.metric.value:.4f} "
                                      f"(prev: {local_best.metric.value:.4f})")
                    else:
                        cur_node.local_best_node = cur_node
                        logger.info(f"  └─ Set as local_best: {cur_node.metric.value:.4f}")

                reward = get_node_reward(agent, cur_node)
                backpropagate(cur_node, reward)
                return True

    local_best_node = cur_node.local_best_node
    local_best_metric = local_best_node.metric.value

    if cur_node.is_buggy is False:
        new_metric = cur_node.metric.value
        if parent_node.is_buggy:
            logger.info(f"[eval] debug success for {parent_node.id}")
            if new_metric:
                if local_best_metric:
                    debug_improvement = new_metric - local_best_metric if agent.metric_maximize else local_best_metric - new_metric
                    if debug_improvement > 0:
                        cur_node.local_best_node = cur_node
                    cur_node.continue_improve = True
                    should_backpropagate = False
                else:
                    cur_node.local_best_node = cur_node
                    cur_node.continue_improve = True
                    should_backpropagate = False
            else:
                should_backpropagate = True

        if new_metric is not None and local_best_metric is not None:
            improvement = new_metric - local_best_metric if agent.metric_maximize else local_best_metric - new_metric
            if improvement < agent.scfg.metric_improvement_threshold and local_best_node.improve_failure_depth < agent.scfg.max_improve_failure:
                local_best_node.improve_failure_depth += 1
                action = "continue"
                cur_node.continue_improve = True
            elif improvement < agent.scfg.metric_improvement_threshold and local_best_node.improve_failure_depth >= agent.scfg.max_improve_failure:
                action = "terminal"
                cur_node.continue_improve = False
                should_backpropagate = True
                cur_node.is_terminal = True
            else:
                action = "continue"
                cur_node.local_best_node = cur_node
                cur_node.continue_improve = True
            logger.info(f"[eval] node {cur_node.id}: improvement={improvement:.6f}, action={action}")
        elif new_metric is not None:
            cur_node.local_best_node = cur_node
            cur_node.continue_improve = True
            logger.info(f"[eval] node {cur_node.id}: improvement=N/A, action=continue")
        else:
            should_backpropagate = True
            logger.info(f"[eval] node {cur_node.id}: improvement=N/A, action=backprop")
    elif cur_node.is_buggy is None:
        logger.warning(f"[eval] node {cur_node.id}: improvement=N/A, action=backprop")
        should_backpropagate = True
    else:
        if cur_node.debug_depth >= agent.scfg.back_debug_depth:
            should_backpropagate = True
            if cur_node.debug_depth >= agent.scfg.max_debug_depth:
                cur_node.is_terminal = True

    if should_backpropagate:
        reward = get_node_reward(agent, cur_node)
        backpropagate(cur_node, reward)
    else:
        agent.current_node_list.append(cur_node)
    return should_backpropagate
