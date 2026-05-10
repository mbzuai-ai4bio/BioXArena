import copy
import json
from pathlib import Path
from typing import Type, TypeVar

import dataclasses_json


def _get_journal_cls():
    from engine.search_node import Journal
    return Journal


def dumps_json(obj: dataclasses_json.DataClassJsonMixin):
    """Serialize dataclasses (such as Journals) to JSON."""
    Journal = _get_journal_cls()
    if isinstance(obj, Journal):
        obj = copy.deepcopy(obj)
        node2parent = {n.id: n.parent.id for n in obj.nodes if n.parent is not None}
        node2best_local_node = {n.id: n.local_best_node.id for n in obj.nodes if n.local_best_node is not None and n.local_best_node.metric.value is not None}
        for n in obj.nodes:
            n.parent = None
            n.local_best_node = None
            n.child_count_lock = None
            n.children = set()

    obj_dict = obj.to_dict()

    if isinstance(obj, Journal):
        obj_dict["node2parent"] = node2parent  # type: ignore
        obj_dict["node2best_local_node"] = node2best_local_node
        obj_dict["__version"] = "2"

    def _json_default(o):
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, set):
            return list(o)
        to_dict = getattr(o, "to_dict", None)
        if callable(to_dict):
            try:
                return to_dict()
            except Exception:
                pass
        return str(o)

    return json.dumps(obj_dict, separators=(",", ":"), default=_json_default)


def dump_json(obj: dataclasses_json.DataClassJsonMixin, path: Path):
    tmp_path = path.with_suffix(".json.tmp")
    data = dumps_json(obj)
    with open(tmp_path, "w") as f:
        f.write(data)
    tmp_path.replace(path)


G = TypeVar("G", bound=dataclasses_json.DataClassJsonMixin)


def loads_json(s: str, cls: Type[G]) -> G:
    """Deserialize JSON to dataclasses."""
    Journal = _get_journal_cls()
    obj_dict = json.loads(s)
    obj = cls.from_dict(obj_dict)

    if isinstance(obj, Journal):
        id2nodes = {n.id: n for n in obj.nodes}
        for child_id, parent_id in obj_dict["node2parent"].items():
            id2nodes[child_id].parent = id2nodes[parent_id]
            id2nodes[child_id].__post_init__()
    return obj


def load_json(path: Path, cls: Type[G]) -> G:
    with open(path, "r") as f:
        return loads_json(f.read(), cls)
