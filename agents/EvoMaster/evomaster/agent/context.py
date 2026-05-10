"""EvoMaster Agent Context Management

Provides context management functionality, including conversation history management,
context window control, and history compaction.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from evomaster.utils.llm import BaseLLM
    from evomaster.utils.types import AssistantMessage, Dialog, Message, ToolMessage
else:
    from evomaster.utils.types import AssistantMessage, Dialog, Message, ToolMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compaction prompts
# ---------------------------------------------------------------------------

COMPACTION_SYSTEM_PROMPT = (
    "You are a conversation summarizer. "
    "Produce a detailed but concise summary of the conversation. "
    "Focus on information that would be helpful for continuing the conversation, including: "
    "what was done, what is currently being worked on, key user requests and constraints, "
    "important decisions and their reasons, and key facts or data. "
    "Do not respond to any questions in the conversation, only output the summary."
)

COMPACTION_USER_PROMPT = """\
Summarize the conversation above for handoff to a continuing agent.

Use this template:
---
## Goal
[What the user is trying to accomplish]

## Key Decisions & Discoveries
[Important findings, user preferences, constraints]

## Accomplished
[What was completed, what is in progress, what remains]

## Context
[Key facts, data, or state needed to continue naturally]
---"""

# Prune thresholds
_PRUNE_PROTECT_TOKENS = 40_000  # Protect recent tool outputs from being cleared
_PRUNE_MINIMUM_TOKENS = 10_000  # Minimum tokens to clear for a prune to be worthwhile
_RESERVED_OUTPUT_TOKENS = 20_000  # Token space reserved for LLM response


class TruncationStrategy(str, Enum):
    """History truncation strategy"""
    NONE = "none"  # No truncation
    LATEST_HALF = "latest_half"  # Keep the latest half
    SLIDING_WINDOW = "sliding_window"  # Sliding window
    SUMMARY = "summary"  # Summary compaction


class ContextConfig(BaseModel):
    """Context management configuration"""
    max_tokens: int = Field(default=128000, description="Maximum token count")
    truncation_strategy: TruncationStrategy = Field(
        default=TruncationStrategy.LATEST_HALF,
        description="Truncation strategy"
    )
    preserve_system_messages: bool = Field(
        default=True,
        description="Whether to preserve system messages"
    )
    preserve_recent_turns: int = Field(
        default=5,
        description="Number of recent conversation turns to preserve"
    )


class ContextManager:
    """Context Manager

    Responsible for managing conversation context, including:
    - Context window size control
    - History message truncation and compaction
    - Token counting (extensible)
    """

    def __init__(self, config: ContextConfig | None = None):
        self.config = config or ContextConfig()
        self._token_counter: TokenCounter | None = None
        self._summary_llm: BaseLLM | None = None
        self._last_prompt_tokens: int = 0
        self._last_prompt_msg_count: int = 0
        self.on_before_compaction: Callable[[list[Message]], None] | None = None

    def set_token_counter(self, counter: TokenCounter) -> None:
        """Set the token counter"""
        self._token_counter = counter

    def set_summary_llm(self, llm: BaseLLM) -> None:
        """Set the LLM used for auto-compact summarization.

        When the truncation strategy is SUMMARY, this LLM is used to summarize
        and compress old messages.
        """
        self._summary_llm = llm

    def update_usage(self, usage: dict[str, int], msg_count: int = 0) -> None:
        """Record real token usage returned by the LLM API.

        Only cares about prompt_tokens (i.e., the actual input token count consumed
        by the dialog). completion_tokens includes thinking tokens, but thinking is
        not stored in the dialog, so total_tokens cannot be used to judge dialog size.

        Args:
            usage: The usage dictionary returned by the API.
            msg_count: Number of messages sent to the API (used for incremental estimation).
        """
        self._last_prompt_tokens = usage.get("prompt_tokens", 0)
        if msg_count > 0:
            self._last_prompt_msg_count = msg_count

    def estimate_tokens(self, dialog: Dialog) -> int:
        """Estimate the token count of a dialog.

        If a token counter is set, use it; otherwise use a simple estimation.
        """
        if self._token_counter:
            return self._token_counter.count_dialog(dialog)

        total_chars = self._count_messages_chars(dialog.messages)

        # Tool specs are sent with each API request and also consume context
        if dialog.tools:
            import json as _json
            for spec in dialog.tools:
                try:
                    total_chars += len(_json.dumps(spec.model_dump()))
                except Exception:
                    total_chars += 500  # fallback per tool

        return total_chars // 4

    def _count_messages_chars(self, messages: list[Message]) -> int:
        """Count the total characters of a set of messages (used for token estimation)."""
        total_chars = 0
        for msg in messages:
            content = msg.content
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # Multimodal content: only count text parts; estimate images at a fixed token count
                for block in content:
                    if block.get("type") == "text":
                        total_chars += len(block.get("text", ""))
                    elif block.get("type") in ("image_url", "image"):
                        total_chars += 3000  # Images ~750 tokens, estimated as 3000 characters
            # tool_calls arguments also consume tokens
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    total_chars += len(tc.function.name) + len(tc.function.arguments or "")
        return total_chars

    def _estimate_messages_tokens(self, messages: list[Message]) -> int:
        """Estimate the token count of a set of messages (excluding tool specs), used for incremental calculation."""
        return self._count_messages_chars(messages) // 4

    def should_truncate(self, dialog: Dialog) -> bool:
        """Determine whether truncation is needed"""
        return self.estimate_tokens(dialog) > self.config.max_tokens

    def truncate(self, dialog: Dialog) -> Dialog:
        """Truncate conversation history according to the configured strategy.

        Returns:
            A new Dialog object after truncation.
        """
        if self.config.truncation_strategy == TruncationStrategy.NONE:
            return dialog
        elif self.config.truncation_strategy == TruncationStrategy.LATEST_HALF:
            return self._truncate_latest_half(dialog)
        elif self.config.truncation_strategy == TruncationStrategy.SLIDING_WINDOW:
            return self._truncate_sliding_window(dialog)
        elif self.config.truncation_strategy == TruncationStrategy.SUMMARY:
            return self._truncate_with_summary(dialog)
        else:
            return dialog

    def _truncate_latest_half(self, dialog: Dialog) -> Dialog:
        """Keep the latest half of the history.

        Preserves system messages and the initial user message, then keeps the most recent half of the conversation.
        """
        messages = dialog.messages
        
        # Find the position of the first assistant message
        assistant_start = 0
        for i, msg in enumerate(messages):
            if msg.role.value == "assistant":
                assistant_start = i
                break

        # Calculate the number of messages to preserve
        num_messages = len(messages)
        num_to_truncate = num_messages - assistant_start
        num_to_preserve = num_to_truncate // 2
        preserve_start = num_messages - num_to_preserve

        # Ensure we start from an assistant message
        while preserve_start < num_messages and messages[preserve_start].role.value != "assistant":
            preserve_start += 1

        if preserve_start >= num_messages:
            # Cannot truncate, return original dialog
            return dialog

        # Build new dialog
        new_messages = messages[:assistant_start] + messages[preserve_start:]
        
        return Dialog(
            messages=new_messages,
            tools=dialog.tools,
            meta={**dialog.meta, "truncated": True, "strategy": "latest_half"}
        )

    def _truncate_sliding_window(self, dialog: Dialog) -> Dialog:
        """Sliding window truncation.

        Preserves system messages and the most recent N turns of conversation.
        One turn = one assistant message and its associated tool messages.
        """
        messages = dialog.messages
        preserve_turns = self.config.preserve_recent_turns

        # Separate system messages from other messages
        system_messages: list[Message] = []
        other_messages: list[Message] = []

        for msg in messages:
            if msg.role.value == "system":
                system_messages.append(msg)
            else:
                other_messages.append(msg)

        # Count preserve_turns assistant messages from the end to determine the keep-from index
        assistant_count = 0
        keep_from = len(other_messages)
        for i in range(len(other_messages) - 1, -1, -1):
            if other_messages[i].role.value == "assistant":
                assistant_count += 1
                if assistant_count >= preserve_turns:
                    keep_from = i
                    break

        if keep_from == 0:
            return dialog

        # Keep the most recent messages
        new_messages = system_messages + other_messages[keep_from:]

        return Dialog(
            messages=new_messages,
            tools=dialog.tools,
            meta={**dialog.meta, "truncated": True, "strategy": "sliding_window"}
        )

    def _truncate_with_summary(self, dialog: Dialog) -> Dialog:
        """Auto-compact: summarize old messages with an LLM, replacing them with a compact context summary.

        Splits the conversation into three parts:
        1. system_msgs: System messages + initial user message (kept unchanged)
        2. old_msgs: Old messages to be summarized
        3. recent_msgs: Recent messages to preserve (kept unchanged)

        Sends old_msgs as a complete conversation (including tool_calls structure) to the
        summary LLM. The resulting dialog = system_msgs + [UserMessage(summary)] + recent_msgs.
        Falls back to the latest_half strategy if the LLM call fails.
        """
        if self._summary_llm is None:
            logger.warning("Summary LLM not set, falling back to latest_half")
            return self._truncate_latest_half(dialog)

        from evomaster.utils.types import (
            AssistantMessage as AMsg,
            Dialog as DialogCls,
            SystemMessage,
            UserMessage,
        )

        messages = dialog.messages

        # Find the position of the first assistant message (after system + initial user)
        assistant_start = 0
        for i, msg in enumerate(messages):
            if msg.role.value == "assistant":
                assistant_start = i
                break

        if assistant_start == 0:
            return dialog

        # Count preserve_recent_turns assistant messages from the end to determine recent_start
        assistant_count = 0
        recent_start = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role.value == "assistant":
                assistant_count += 1
                if assistant_count >= self.config.preserve_recent_turns:
                    recent_start = i
                    break

        if recent_start >= len(messages) or recent_start <= assistant_start:
            return self._truncate_latest_half(dialog)

        # system_msgs only keeps messages with role "system"; the initial user message will be covered by the summary
        system_msgs = [m for m in messages[:assistant_start] if m.role.value == "system"]
        # old_msgs includes the initial user message and everything up to recent_start
        old_msgs = messages[len(system_msgs):recent_start]
        recent_msgs = messages[recent_start:]

        if not old_msgs:
            return dialog

        # Trigger pre-compaction hook (e.g., for memory extraction)
        if self.on_before_compaction:
            try:
                self.on_before_compaction(list(old_msgs))
            except Exception:
                logger.exception("on_before_compaction hook failed")

        # Build summary dialog: system prompt + old_msgs complete structure + summary instruction
        # old_msgs are kept intact without truncation so the summary LLM sees the full context
        try:
            summary_dialog = DialogCls(
                messages=[
                    SystemMessage(content=COMPACTION_SYSTEM_PROMPT),
                    *old_msgs,
                    UserMessage(content=COMPACTION_USER_PROMPT),
                ],
                tools=[],
            )

            response = self._summary_llm.query(summary_dialog)
            summary_text = response.content or ""

            if not summary_text.strip():
                logger.warning("Empty summary from LLM, falling back to latest_half")
                return self._truncate_latest_half(dialog)

            logger.info(
                "Auto-compact: summarized %d messages -> %d chars summary",
                len(old_msgs),
                len(summary_text),
            )

            # Use a user ask + assistant answer pair to represent the summary
            compaction_request = UserMessage(
                content="What did we do so far?",
            )
            compaction_response = AMsg(
                content=summary_text,
                meta={"summary": True, "strategy": "compaction"},
            )
            new_messages = (
                list(system_msgs)
                + [compaction_request, compaction_response]
                + list(recent_msgs)
            )

            return DialogCls(
                messages=new_messages,
                tools=dialog.tools,
                meta={**dialog.meta, "truncated": True, "strategy": "summary"},
            )

        except Exception:
            logger.exception("Auto-compact failed, falling back to latest_half")
            return self._truncate_latest_half(dialog)

    def reset_prompt_tokens(self) -> None:
        """Reset prompt_tokens record, used to force re-estimation after compact write-back."""
        self._last_prompt_tokens = 0
        self._last_prompt_msg_count = 0

    def prepare_for_query(self, dialog: Dialog) -> tuple[Dialog, bool]:
        """Prepare dialog for LLM query.

        References OpenCode's isOverflow logic:
        - Uses real prompt_tokens (from the last LLM API response) + incremental estimation
          to determine if overflow occurs.
        - Falls back to estimate_tokens on the first call when no usage data is available.
        - usable = max_tokens - reserved_output_tokens

        Two-tier strategy:
        1. tokens >= usable -> full summary (truncate) -- permanent compaction
        2. tokens >= 80% usable -> lightweight prune (clear old tool outputs) -- temporary view

        Returns:
            (dialog_for_query, compacted):
            - compacted=True: permanent compaction was performed (truncate/summary);
              the caller should write back to current_dialog.
            - compacted=False: no change or only a temporary prune; should not be written back
              (preserve full tool outputs for future summaries).
        """
        usable = self.config.max_tokens - _RESERVED_OUTPUT_TOKENS
        if usable <= 0:
            # max_tokens is less than reserved, cannot compute a valid threshold; skip truncation
            logger.warning(
                "max_tokens (%d) <= _RESERVED_OUTPUT_TOKENS (%d), skipping truncation",
                self.config.max_tokens,
                _RESERVED_OUTPUT_TOKENS,
            )
            return dialog, False

        if self._last_prompt_tokens > 0 and self._last_prompt_msg_count > 0:
            # Incremental estimation: last real tokens + estimated new message tokens
            current_msg_count = len(dialog.messages)
            if current_msg_count > self._last_prompt_msg_count:
                new_msgs = dialog.messages[self._last_prompt_msg_count:]
                delta = self._estimate_messages_tokens(new_msgs)
                tokens = self._last_prompt_tokens + delta
            else:
                tokens = self._last_prompt_tokens
        else:
            # First call with no usage data; use full estimation
            tokens = self.estimate_tokens(dialog)

        # 5% safety margin
        tokens = int(tokens * 1.05)

        if tokens >= usable:
            return self.truncate(dialog), True
        if tokens >= int(usable * 0.8):
            return self._prune_old_tool_outputs(dialog), False
        return dialog, False

    def is_overflow(self, usage: dict[str, int]) -> bool:
        """Use real token counts returned by the API to determine whether compaction is needed.

        References OpenCode's isOverflow: called after each successful LLM response,
        using real total_tokens to determine if the context limit is being approached.
        If so, the caller should immediately perform compaction to avoid overflow on the next call.
        """
        total = usage.get("total_tokens") or (
            usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        )
        usable = self.config.max_tokens - _RESERVED_OUTPUT_TOKENS
        return total >= usable

    def _prune_old_tool_outputs(self, dialog: Dialog) -> Dialog:
        """Lightweight prune: clear old tool outputs while protecting recent ones.

        References OpenCode's prune strategy:
        - Scan from the most recent message backwards
        - Protect tool outputs from the 2 most recent user turns
        - Replace content of ToolMessages beyond the protection range with "[Old tool output cleared]"
        - Only execute if the clearable amount exceeds the threshold
        """
        from evomaster.utils.types import Dialog as DialogCls, ToolMessage as TMsg

        messages = dialog.messages
        tool_token_total = 0
        prunable_indices: list[int] = []
        prunable_tokens: list[int] = []
        user_turns = 0

        # Scan from the end backwards
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.role.value == "user":
                user_turns += 1
            if user_turns < 2:
                continue  # Protect the 2 most recent user turns

            if isinstance(msg, TMsg):
                content = msg.content or ""
                if isinstance(content, str) and len(content) > 200:
                    tokens = len(content) // 4
                    tool_token_total += tokens
                    if tool_token_total > _PRUNE_PROTECT_TOKENS:
                        prunable_indices.append(i)
                        prunable_tokens.append(tokens)

        total_prunable = sum(prunable_tokens)
        if total_prunable < _PRUNE_MINIMUM_TOKENS:
            return dialog

        logger.info(
            "Prune: clearing %d old tool outputs (~%d tokens)",
            len(prunable_indices),
            total_prunable,
        )

        new_messages = list(messages)
        for idx in prunable_indices:
            old_msg = new_messages[idx]
            assert isinstance(old_msg, TMsg)
            new_messages[idx] = TMsg(
                content="[Old tool output cleared]",
                tool_call_id=old_msg.tool_call_id,
                name=old_msg.name,
                meta=old_msg.meta,
            )

        return DialogCls(
            messages=new_messages,
            tools=dialog.tools,
            meta={**dialog.meta, "pruned": True},
        )


class TokenCounter(ABC):
    """Abstract base class for token counters"""

    @abstractmethod
    def count_text(self, text: str) -> int:
        """Count the tokens in a text string"""
        pass

    @abstractmethod
    def count_message(self, message: Message) -> int:
        """Count the tokens in a single message"""
        pass

    def count_dialog(self, dialog: Dialog) -> int:
        """Count the total tokens of a dialog"""
        return sum(self.count_message(msg) for msg in dialog.messages)


class SimpleTokenCounter(TokenCounter):
    """Simple Token Counter

    A simple estimation based on character count.
    """
    
    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token

    def count_text(self, text: str) -> int:
        return int(len(text) / self.chars_per_token)

    def count_message(self, message: Message) -> int:
        content = message.content
        if isinstance(content, str):
            content_tokens = self.count_text(content)
        elif isinstance(content, list):
            content_tokens = 0
            for block in content:
                if block.get("type") == "text":
                    content_tokens += self.count_text(block.get("text", ""))
                elif block.get("type") in ("image_url", "image"):
                    content_tokens += 750  # Fixed estimate for images
        else:
            content_tokens = 0
        # Extra token overhead (role, formatting, etc.)
        overhead = 4
        return content_tokens + overhead

