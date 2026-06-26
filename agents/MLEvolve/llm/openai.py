"""OpenAI-compatible API backend (query + generate). Supports Qwen API and any OpenAI-compatible endpoint."""

import json
import logging
import re
import time
from typing import Any

from openai import OpenAI

from config import Config
from .gemini import FunctionSpec, compile_prompt_to_md
from .model_profiles import get_profile, supports_json_schema, thinking_json_incompatible

logger = logging.getLogger("MLEvolve")


def _write_llm_usage(
    cfg: Config,
    model: str,
    created: int,
    input_tokens: int,
    output_tokens: int,
    request_time_sec: float,
) -> None:
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "created": created,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "request_time_sec": request_time_sec,
    }
    with open(cfg.log_dir / "llm_usage.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _strip_markdown_fences(args: str) -> str:
    """Remove markdown code fences that LLMs sometimes append inside JSON string values."""
    cleaned = re.sub(r'\\n```[a-z]*\s*("?\s*\}?\s*)$', r'\1', args.rstrip())
    cleaned = cleaned.rstrip()
    if not cleaned.endswith('}'):
        if not cleaned.endswith('"'):
            cleaned += '"'
        cleaned += '}'
    return cleaned


def _parse_json_args(args: str) -> dict:
    """Parse function call arguments, tolerating Python literals and markdown fences."""
    # 1. Fast path: valid JSON as-is
    try:
        return json.loads(args)
    except json.JSONDecodeError:
        pass

    # 2. Try stripping markdown fences
    try:
        cleaned = _strip_markdown_fences(args)
        if cleaned != args:
            result = json.loads(cleaned)
            logger.warning("Fixed malformed function args by stripping markdown code fences")
            return result
    except json.JSONDecodeError:
        pass

    # 3. Normalize Python literals (None/True/False) outside quoted strings
    parts = re.split(r'("(?:[^"\\]|\\.)*")', args)
    normalized = []
    for part in parts:
        if part.startswith('"'):
            normalized.append(part)
        else:
            part = re.sub(r'\bNone\b', 'null', part)
            part = re.sub(r'\bTrue\b', 'true', part)
            part = re.sub(r'\bFalse\b', 'false', part)
            normalized.append(part)
    normalized_str = ''.join(normalized)

    try:
        return json.loads(normalized_str)
    except json.JSONDecodeError:
        pass

    # 4. Normalized + strip markdown fences
    cleaned = _strip_markdown_fences(normalized_str)
    return json.loads(cleaned)

# Return type aligned with gemini.query
OutputType = str | dict


def _stage_config_for_model(cfg: Config, model: str):
    """Return code or feedback config depending on which model is being used."""
    if cfg.agent.code.model == model:
        return cfg.agent.code
    return cfg.agent.feedback


def _build_messages(system_message: str | None, user_message: str | None) -> list[dict[str, str]]:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    cfg: Config | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """OpenAI-compatible query (chat completions, optional function calling). Same return shape as gemini.query."""
    if cfg is None:
        raise ValueError("cfg is required for OpenAI backend")
    filtered = {k: v for k, v in model_kwargs.items() if v is not None}
    model = filtered.get("model", "")
    stage = _stage_config_for_model(cfg, model)
    client = OpenAI(
        api_key=stage.api_key,
        base_url=stage.base_url or None,
        timeout=180.0,
    )
    messages = _build_messages(system_message, user_message)
    if not messages:
        raise ValueError("Either system_message or user_message must be provided")

    # Function calling requires non_thinking mode, otherwise Qwen API errors:
    # "tool_choice does not support required/object in thinking mode"
    use_thinking = func_spec is None
    profile = get_profile(model, use_thinking=use_thinking)

    extra_body: dict[str, Any] = {}
    if "top_k" in profile:
        extra_body["top_k"] = profile["top_k"]
    if "enable_thinking" in profile:
        extra_body["enable_thinking"] = profile["enable_thinking"]

    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": profile.get("temperature", filtered.get("temperature", 1.0)),
        "max_tokens": filtered.get("max_tokens", 16384),
    }
    if "top_p" in profile:
        params["top_p"] = profile["top_p"]
    if "presence_penalty" in profile:
        params["presence_penalty"] = profile["presence_penalty"]
    if extra_body:
        params["extra_body"] = extra_body
    if func_spec is not None:
        tool_dict = func_spec.as_openai_tool_dict
        if not supports_json_schema(model):
            tool_dict.pop("strict", None)
        params["tools"] = [tool_dict]
        params["tool_choice"] = func_spec.openai_tool_choice_dict

    t0 = time.time()
    logger.info(f"Querying OpenAI-compatible API with model: {model}")
    try:
        completion = client.chat.completions.create(**params)
    except Exception as e:
        # Endpoint-specific fallback: some OpenAI-compatible endpoints (e.g. gpt-5.x
        # proxied through the Responses API) reject system-only requests because the
        # user content maps to the required `input` field, which ends up empty. Only
        # that specific 400 triggers this branch — the standard Chat Completions path
        # (e.g. OpenRouter) succeeds on the first call and never reaches here, so its
        # behaviour is completely unchanged.
        err = str(e)
        only_system = len(messages) == 1 and messages[0].get("role") == "system"
        responses_api_input_error = "must be provided" in err and '"input"' in err
        if only_system and responses_api_input_error:
            logger.warning(
                "Endpoint rejected system-only request (Responses API); "
                "retrying once with system content promoted to a user message"
            )
            sys_content = messages[0]["content"]
            if not isinstance(sys_content, str):
                sys_content = compile_prompt_to_md(sys_content)
            params["messages"] = [{"role": "user", "content": sys_content}]
            completion = client.chat.completions.create(**params)
        else:
            logger.error(f"Error calling OpenAI-compatible API: {e}")
            raise
    req_time = time.time() - t0
    choice = completion.choices[0]
    message = choice.message

    if getattr(choice, "finish_reason", None) == "length":
        logger.warning(f"Response truncated by max_tokens ({params.get('max_tokens')}), consider increasing it")

    if func_spec is None:
        output = message.content or ""
        logger.info(f"OpenAI response: {output}", extra={"verbose": True})
    else:
        if not message.tool_calls:
            raise ValueError("Expected function call, got no tool_calls")
        tc = message.tool_calls[0]
        if tc.function.name != func_spec.name:
            raise ValueError(f"Function name mismatch: expected {func_spec.name}, got {tc.function.name}")
        try:
            output = _parse_json_args(tc.function.arguments or "{}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid function arguments: {tc.function.arguments}")
            raise e
        logger.info(f"OpenAI function call response: {output}", extra={"verbose": True})

    in_tok = getattr(completion.usage, "prompt_tokens", 0) or 0
    out_tok = getattr(completion.usage, "completion_tokens", 0) or 0
    info = {
        "model": getattr(completion, "model", model),
        "created": getattr(completion, "created", int(time.time())),
    }
    return output, req_time, in_tok, out_tok, info


def _prompt_to_messages(prompt: str | dict | list, model: str = "") -> list[dict[str, str]]:
    """Convert prompt to chat messages.

    Legacy prompt dicts may contain an ``assistant`` field used as context/prefill,
    not as a true prior assistant turn. Merge it into the user message so every
    backend receives a normal system+user prompt instead of a dangling assistant
    continuation.
    """
    if isinstance(prompt, dict) and ("system" in prompt or "user" in prompt or "assistant" in prompt):
        messages = []
        if prompt.get("system"):
            messages.append({"role": "system", "content": str(prompt["system"])})

        user_content = str(prompt["user"]) if prompt.get("user") else ""
        assistant_content = str(prompt["assistant"]) if prompt.get("assistant") else ""

        if assistant_content:
            combined = f"{user_content}\n\n{assistant_content}" if user_content else assistant_content
            messages.append({"role": "user", "content": combined})
        else:
            if user_content:
                messages.append({"role": "user", "content": user_content})

        if not messages:
            raise ValueError("Chat dict must have at least one of: system, user, assistant")
        return messages
    content = prompt if isinstance(prompt, str) else compile_prompt_to_md(prompt)
    return [{"role": "user", "content": content}]


def generate(
    prompt: str | dict | list,
    cfg: Config,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stop_tokens: list[str] | None = None,
    json_schema: dict | None = None,
    max_retries: int = 20,
    retry_delay: float = 3,
) -> str:
    """Streaming text generation via OpenAI-compatible Chat API. Supports chat format {system, user, assistant} for Qwen."""
    stage = cfg.agent.code
    model = stage.model
    messages = _prompt_to_messages(prompt, model=model)
    client = OpenAI(
        api_key=stage.api_key,
        base_url=stage.base_url or None,
        timeout=180.0,
    )
    # Qwen: thinking + json_schema are mutually exclusive — drop schema, keep thinking.
    if json_schema is not None and thinking_json_incompatible(model):
        json_schema = None
    use_thinking = json_schema is None
    profile = get_profile(model, use_thinking=use_thinking)

    extra_body: dict[str, Any] = {}
    if "top_k" in profile:
        extra_body["top_k"] = profile["top_k"]
    if "enable_thinking" in profile:
        extra_body["enable_thinking"] = profile["enable_thinking"]

    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": profile.get("temperature", temperature if temperature is not None else 1.0),
        "max_tokens": max_tokens if max_tokens is not None else 16384,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if "top_p" in profile:
        params["top_p"] = profile["top_p"]
    if "presence_penalty" in profile:
        params["presence_penalty"] = profile["presence_penalty"]
    if extra_body:
        params["extra_body"] = extra_body
    if stop_tokens:
        params["stop"] = stop_tokens
    if json_schema is not None:
        if supports_json_schema(model):
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured_output", "strict": False, "schema": json_schema},
            }
        else:
            params["response_format"] = {"type": "json_object"}

    logger.info(f"generate messages: {len(messages)} turns", extra={"verbose": True})
    for attempt in range(max_retries):
        full_text = ""
        input_tokens = 0
        output_tokens = 0
        usage_seen = False
        response_model = model
        created = int(time.time())
        try:
            t0 = time.time()
            stream = client.chat.completions.create(**params)
            for chunk in stream:
                response_model = getattr(chunk, "model", response_model) or response_model
                created = getattr(chunk, "created", created) or created
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    usage_seen = True
                    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(usage, "completion_tokens", 0) or 0
                if chunk.choices and chunk.choices[0].delta.content:
                    full_text += chunk.choices[0].delta.content
            request_time_sec = time.time() - t0
        except Exception as e:
            logger.warning(f"generate failed, retrying {attempt + 1}/{max_retries}: {e}")
            if attempt >= max_retries - 1:
                logger.error("generate retry limit reached")
                raise
            time.sleep(retry_delay)
            continue

        if "</think>" in full_text:
            full_text = full_text[full_text.find("</think>") + 8:]
        logger.info(f"generate response: {full_text}", extra={"verbose": True})
        if not usage_seen:
            raise RuntimeError("OpenAI-compatible streaming response did not include usage")
        _write_llm_usage(cfg, response_model, created, input_tokens, output_tokens, request_time_sec)
        return full_text
    return ""
