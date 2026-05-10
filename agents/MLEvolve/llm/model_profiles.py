"""Model-specific parameter profiles for OpenAI-compatible backends.

Usage:
    profile = get_profile(model_name, use_thinking=True)
    # Returns a dict with any subset of:
    #   temperature, top_p, presence_penalty  — standard OpenAI Chat params
    #   top_k, enable_thinking                — go into extra_body (Qwen-specific)

To add a new model family, add an entry to _PROFILES below.
Each entry has two modes: "thinking" and "non_thinking".
Only include params that differ from provider defaults — missing keys are skipped.
Longer prefixes take precedence (e.g. "gpt-4o" wins over "gpt").
"""

from __future__ import annotations

_PROFILES: dict[str, dict] = {
    # ── Qwen series ──────────────────────────────────────────────────────
    "qwen": {
        "thinking": {
            # Precise coding tasks
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "presence_penalty": 0.0, "enable_thinking": True,
        },
        "non_thinking": {
            # General tasks (used for planner / structured output)
            "temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "presence_penalty": 1.5, "enable_thinking": False,
        },
    },

    # ── GPT series ───
    "gpt": {
        "thinking": {
            # Coding / agentic tasks — OpenAI default temp works well
            "temperature": 1.0,
        },
        "non_thinking": {
            # Planner / structured-output calls
            "temperature": 0.7,
            "presence_penalty": 0.1,
        },
    },

    # ── Fallback for any unrecognised model ──────────────────────────────────
    "default": {
        "thinking":     {},
        "non_thinking": {},
    },
}

# Models that only support {"type": "json_object"}, not json_schema + strict.
_NO_JSON_SCHEMA_PREFIXES = ("deepseek",)

# Models where thinking mode and json_schema are mutually exclusive.
# generate() will drop json_schema for these models to keep thinking enabled,
# relying on prompt instructions + post-processing for JSON extraction.
_THINKING_JSON_INCOMPATIBLE = ("qwen",)


def thinking_json_incompatible(model_name: str) -> bool:
    """Return True for models that cannot use thinking + json_schema simultaneously."""
    name = (model_name or "").lower()
    return any(name.startswith(p) for p in _THINKING_JSON_INCOMPATIBLE)


def supports_json_schema(model_name: str) -> bool:
    """Return False for models that require json_object instead of json_schema+strict."""
    name = (model_name or "").lower()
    return not any(name.startswith(p) for p in _NO_JSON_SCHEMA_PREFIXES)


def get_profile(model_name: str, use_thinking: bool = True) -> dict:
    """Return parameter dict for model_name.

    Matches by longest prefix (case-insensitive). Falls back to 'default'.
    """
    name = (model_name or "").lower()
    for key in sorted(_PROFILES, key=len, reverse=True):
        if key == "default":
            continue
        if name.startswith(key):
            mode = "thinking" if use_thinking else "non_thinking"
            return dict(_PROFILES[key][mode])
    mode = "thinking" if use_thinking else "non_thinking"
    return dict(_PROFILES["default"][mode])
