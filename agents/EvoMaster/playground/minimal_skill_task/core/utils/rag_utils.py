"""RAG-related utilities: parse Plan output, extract agent responses, handle database parameters, and update prompt placeholders."""

import json
import os
import re
from pathlib import Path
from typing import Any

DEFAULT_VEC_DIR = "evomaster/skills/rag/MLE_DATABASE/node_vectorstore/draft"
DEFAULT_NODES_DATA = "evomaster/skills/rag/MLE_DATABASE/node_vectorstore/draft/draft_407_75_db.json"

# Global embedding configuration (set by the playground).
_embedding_config: dict | None = None


def set_embedding_config(config: dict | None) -> None:
    """Set the global embedding configuration (called by the playground)."""
    global _embedding_config
    _embedding_config = config


def get_embedding_config() -> dict | None:
    """Get the global embedding configuration."""
    return _embedding_config


def _project_root() -> Path:
    """Project root directory (contains evomaster/ and playground/; go three levels up from rag_utils to core, then two more levels up)."""
    return Path(__file__).resolve().parent.parent.parent.parent.parent


def _resolve_db_path(path_str: str, root: Path) -> str:
    """Convert a relative path (such as evomaster/...) to an absolute path; return as-is if it's already absolute."""
    if not path_str or not path_str.strip():
        return path_str
    p = Path(path_str.strip().replace("\\", "/"))
    if p.is_absolute():
        return str(p.resolve())
    return str((root / p).resolve())


def resolve_db_to_absolute_paths(db: dict, project_root: Path | None = None) -> dict:
    """Convert vec_dir, nodes_data, and model in db to absolute paths (for convenient use by RAG and all agents)."""
    root = project_root or _project_root()
    result = {
        "vec_dir": _resolve_db_path(db["vec_dir"], root),
        "nodes_data": _resolve_db_path(db["nodes_data"], root),
        "model": _resolve_db_path(db["model"], root),
    }
    
    # Add embedding configuration parameters.
    embedding_config = get_embedding_config()
    if embedding_config:
        emb_type = embedding_config.get("type", "local")
        result["embedding_type"] = emb_type
        
        if emb_type == "openai":
            openai_cfg = embedding_config.get("openai", {})
            result["model"] = openai_cfg.get("model", "text-embedding-3-large")
            result["embedding_dimensions"] = openai_cfg.get("dimensions")
        else:
            # In local mode, do not fall back to a built-in model; local.model must be provided explicitly in the config.
            local_cfg = embedding_config.get("local", {})
            local_model = (local_cfg.get("model") or "").strip()
            if local_model:
                result["model"] = _resolve_db_path(local_model, root)
            result["embedding_type"] = "local"
    
    return result


def parse_plan_output(text: str) -> dict:
    """Parse query, top_k, and threshold from the Plan Agent output."""
    out = {"query": "", "top_k": 5, "threshold": 1.5}
    if not text:
        return out
    q = re.search(r"query\s*[：:]\s*(.+?)(?=\s*(?:top_k|threshold)|$)", text, re.DOTALL | re.IGNORECASE)
    if q:
        out["query"] = q.group(1).strip().strip('"\'')
    k = re.search(r"top_k\s*[：:]\s*(\d+)", text, re.IGNORECASE)
    if k:
        out["top_k"] = int(k.group(1))
    t = re.search(r"threshold\s*[：:]\s*([\d.]+)", text, re.IGNORECASE)
    if t:
        out["threshold"] = float(t.group(1))
    return out


def extract_agent_response(trajectory: Any) -> str:
    """Extract the final answer from an agent trajectory. If the agent finishes via a finish tool, use its message; otherwise use the last assistant content."""
    if not trajectory or not trajectory.dialogs:
        return ""
    last_dialog = trajectory.dialogs[-1]
    for message in reversed(last_dialog.messages):
        if not (hasattr(message, "role") and getattr(message.role, "value", message.role) == "assistant"):
            continue
        # If this assistant message invoked the finish tool, prefer using the finish message as the answer.
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                fn = getattr(tc, "function", tc) if hasattr(tc, "function") else tc
                name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None)
                if name == "finish":
                    args = getattr(fn, "arguments", None) or (fn.get("arguments", "{}") if isinstance(fn, dict) else "{}")
                    try:
                        obj = json.loads(args) if isinstance(args, str) else args
                        if isinstance(obj, dict) and "message" in obj:
                            return obj["message"]
                    except (json.JSONDecodeError, TypeError):
                        pass
        if hasattr(message, "content") and message.content:
            return message.content
    return ""


def get_db_from_description(description: str) -> dict:
    """Parse database parameters from the task description, falling back to default (relative) values if a field is missing."""
    db = {
        "vec_dir": DEFAULT_VEC_DIR,
        "nodes_data": DEFAULT_NODES_DATA,
        # By default do not specify a concrete model to avoid implicit dependence on project-internal paths; this should be given explicitly in higher-level config or in the description.
        "model": "",
        # Default embedding parameters (will be overwritten / completed by resolve_db_to_absolute_paths).
        "embedding_type": "local",
        "embedding_dimensions": "",
    }
    if "vec_dir:" in description:
        m = re.search(r"vec_dir:\s*(\S+)", description)
        if m:
            db["vec_dir"] = m.group(1).strip()
    if "nodes_data:" in description:
        m = re.search(r"nodes_data:\s*(\S+)", description)
        if m:
            db["nodes_data"] = m.group(1).strip()
    if "model:" in description:
        m = re.search(r"model:\s*(\S+)", description)
        if m:
            db["model"] = m.group(1).strip()
    return db


def update_agent_format_kwargs(agent, **kwargs) -> None:
    """Update an agent's prompt_format_kwargs (used for user prompt placeholders)."""
    if hasattr(agent, "_prompt_format_kwargs"):
        agent._prompt_format_kwargs.update(kwargs)
