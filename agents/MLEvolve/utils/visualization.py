"""Render the solution tree as a Rich tree or plain-text string."""
from engine.search_node import Journal
from engine.search_node import SearchNode
from rich.tree import Tree


def journal_to_rich_tree(journal: Journal):
    best_node = journal.get_best_node()

    def append_rec(node: SearchNode, tree):
        stage_str = ""
        try:
            if hasattr(node, 'stage'):
                stage_val = getattr(node, 'stage', None)
                if stage_val:
                    stage_str = f" ({stage_val})"
        except Exception:
            pass

        if not stage_str:
            try:
                if hasattr(node, '__dict__') and node.__dict__ and 'stage' in node.__dict__:
                    stage_val = node.__dict__['stage']
                    if stage_val:
                        stage_str = f" ({stage_val})"
            except Exception:
                pass

        if not stage_str:
            try:
                if hasattr(node, 'stage_name'):
                    stage_name_val = node.stage_name
                    if stage_name_val:
                        stage_str = f" ({stage_name_val})"
            except Exception:
                pass

        if node.is_buggy:
            s = f"[red]◍ bug{stage_str} (ID: {node.id})"
        else:
            style = "bold " if node is best_node else ""
            metric_str = f"{node.metric.value:.3f}" if node.metric.value is not None else "None"
            if node is best_node:
                s = f"[{style}green]● {metric_str}{stage_str} (best) (ID: {node.id})"
            else:
                s = f"[{style}green]● {metric_str}{stage_str} (ID: {node.id})"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree


def journal_to_string_tree(journal: Journal) -> str:
    best_node = journal.get_best_node()
    tree_str = "Solution tree\n"

    def append_rec(node: SearchNode, level: int):
        nonlocal tree_str
        indent = "  " * level
        if node.is_buggy:
            s = f"{indent}◍ bug (ID: {node.id})\n"
        else:
            markers = []
            if node is best_node:
                markers.append("best")
            marker_str = " & ".join(markers)

            stage_str = ""
            try:
                if hasattr(node, 'stage'):
                    stage_val = getattr(node, 'stage', None)
                    if stage_val:
                        stage_str = f" [{stage_val}]"
            except Exception:
                pass

            if not stage_str:
                try:
                    if hasattr(node, '__dict__') and node.__dict__ and 'stage' in node.__dict__:
                        stage_val = node.__dict__['stage']
                        if stage_val:
                            stage_str = f" ({stage_val})"
                except Exception:
                    pass

            if not stage_str:
                try:
                    if hasattr(node, 'stage_name'):
                        stage_name_val = node.stage_name
                        if stage_name_val:
                            stage_str = f" ({stage_name_val})"
                except Exception:
                    pass

            metric_str = f"{node.metric.value:.3f}" if (node.metric and node.metric.value is not None) else "None"
            if marker_str and node.metric and node.metric.value is not None:
                s = f"{indent}● {metric_str}{stage_str} ({marker_str}) (ID: {node.id})\n"
            else:
                s = f"{indent}● {metric_str}{stage_str} (ID: {node.id})\n"
        tree_str += s
        for child in node.children:
            append_rec(child, level + 1)

    for n in journal.draft_nodes:
        append_rec(n, 0)

    return tree_str
