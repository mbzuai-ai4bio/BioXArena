import json
import logging
import time
from pathlib import Path
from threading import Lock
from . import gemini as _gemini
from . import openai as _openai
from .gemini import FunctionSpec, OutputType, PromptType, compile_prompt_to_md
from config import Config
logger = logging.getLogger("MLEvolve")

_USAGE_LOG_LOCK = Lock()


def _append_usage_log(
    cfg: Config | None,
    input_tokens: int,
    output_tokens: int,
    req_time: float,
    info: dict | None,
) -> None:
    """Persist per-call token usage so outer runners can aggregate exact totals."""
    if cfg is None:
        return

    log_dir = getattr(cfg, "log_dir", None)
    if not log_dir:
        return

    record = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": (info or {}).get("model"),
        "created": (info or {}).get("created"),
        "input_tokens": max(0, int(input_tokens or 0)),
        "output_tokens": max(0, int(output_tokens or 0)),
        "request_time_sec": round(float(req_time or 0.0), 6),
    }
    usage_path = Path(log_dir) / "llm_usage.jsonl"

    try:
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _USAGE_LOG_LOCK:
            with open(usage_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:
        logger.warning(f"Failed to append token usage log to {usage_path}: {exc}")


def _provider(model: str) -> str:
    """Use Gemini backend for model names starting with 'gemini', else OpenAI-compatible (e.g. Qwen)."""
    return "gemini" if (model or "").lower().startswith("gemini") else "openai"


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    cfg:Config=None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gemini-3-pro-preview")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.info("---Querying model---", extra={"verbose": True})
    system_message = compile_prompt_to_md(system_message) if system_message else None
    if system_message:
        if len(system_message) > 1000:
            logger.info(f"system: {system_message[-1000:]}", extra={"verbose": True})
        else:
            logger.info(f"system: {system_message}", extra={"verbose": True})
    user_message = compile_prompt_to_md(user_message) if user_message else None
    if user_message:
        if len(user_message) > 1000:
            logger.info(f"user: {user_message[-1000:]}", extra={"verbose": True})
        else:
            logger.info(f"user: {user_message}", extra={"verbose": True})
    if func_spec:
        logger.info(f"function spec: {func_spec.to_dict()}", extra={"verbose": True})

    provider = _provider(model)
    if provider == "openai":
        output, req_time, in_tok_count, out_tok_count, info = _openai.query(
            system_message=system_message,
            user_message=user_message,
            func_spec=func_spec,
            cfg=cfg,
            **model_kwargs,
        )
    else:
        output, req_time, in_tok_count, out_tok_count, info = _gemini.query(
            system_message=system_message,
            user_message=user_message,
            func_spec=func_spec,
            cfg=cfg,
            **model_kwargs,
        )
    _append_usage_log(
        cfg=cfg,
        input_tokens=in_tok_count,
        output_tokens=out_tok_count,
        req_time=req_time,
        info=info,
    )
    logger.info("---Query complete---", extra={"verbose": True})

    return output


def generate(
    prompt,
    cfg,
    temperature=None,
    max_tokens=None,
    stop_tokens=None,
    json_schema=None,
    max_retries=20,
    retry_delay=3,
):
    """Streaming text generation. Dispatches to Gemini or OpenAI-compatible backend by cfg.agent.code.model."""
    model = getattr(cfg.agent.code, "model", "") or ""
    if _provider(model) == "openai":
        return _openai.generate(
            prompt=prompt,
            cfg=cfg,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_tokens=stop_tokens,
            json_schema=json_schema,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
    return _gemini.generate(
        prompt=prompt,
        cfg=cfg,
        temperature=temperature,
        max_tokens=max_tokens,
        stop_tokens=stop_tokens,
        json_schema=json_schema,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
