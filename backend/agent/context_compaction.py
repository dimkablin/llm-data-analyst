from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from backend.agent.context_window import (
    MessageTokenCounter,
    build_context_usage_snapshot,
    reserved_response_tokens_for_settings,
)
from backend.agent.runtime_llm import build_runtime_llm
from backend.agent.services.events import emit_context_usage_event
from backend.core.config import Settings
from backend.sessions.session_memory import SessionMemory

logger = logging.getLogger(__name__)

_KEEP_RECENT_HISTORY_MESSAGES = 4
_SUMMARY_MAX_TOKENS = 900

ContextCompactionResultStatus = Literal["idle", "done", "failed"]


@dataclass(frozen=True)
class ContextCompactionResult:
    status: ContextCompactionResultStatus
    compacted_message_count: int = 0
    error: str = ""


def compact_context_if_needed(
    *,
    messages: Sequence[BaseMessage],
    history: list[dict[str, Any]],
    settings: Settings,
    session_memory: SessionMemory,
    callbacks: list[Any],
    include_reasoning: bool = False,
    count_message_tokens: MessageTokenCounter | None = None,
) -> ContextCompactionResult:
    max_context_tokens = int(settings.llm_num_ctx or 0)
    if max_context_tokens <= 0:
        return ContextCompactionResult(status="idle")

    snapshot = build_context_usage_snapshot(
        messages,
        max_context_tokens=max_context_tokens,
        reserved_response_tokens=0,
        context_window_source="settings",
        count_message_tokens=count_message_tokens,
    )
    if snapshot.usage_ratio is None or snapshot.usage_ratio < _max_context_per(settings):
        return ContextCompactionResult(status="idle")

    already_compacted_count = max(
        0,
        int(getattr(session_memory, "compacted_message_count", 0) or 0),
    )
    target_compacted_count = max(0, len(history) - _KEEP_RECENT_HISTORY_MESSAGES)
    if target_compacted_count <= already_compacted_count:
        return ContextCompactionResult(
            status="idle",
            compacted_message_count=already_compacted_count,
        )

    history_to_compact = [
        dict(item)
        for item in history[:target_compacted_count]
        if isinstance(item, dict)
    ]
    if not history_to_compact:
        return ContextCompactionResult(
            status="idle",
            compacted_message_count=already_compacted_count,
        )

    _emit_compaction_usage(
        callbacks,
        messages=messages,
        settings=settings,
        include_reasoning=include_reasoning,
        count_message_tokens=count_message_tokens,
        status="running",
    )
    try:
        llm = build_runtime_llm(
            settings,
            role="chat",
            include_reasoning=False,
            max_tokens_override=_SUMMARY_MAX_TOKENS,
        )
        response = llm.invoke(
            _summary_messages(
                existing_summary=str(getattr(session_memory, "context_summary", "") or ""),
                history=history_to_compact,
            )
        )
        summary = _response_text(response)
        if not summary:
            raise RuntimeError("empty context summary")
    except Exception as exc:
        logger.warning("context compaction failed: %s", exc)
        _emit_compaction_usage(
            callbacks,
            messages=messages,
            settings=settings,
            include_reasoning=include_reasoning,
            count_message_tokens=count_message_tokens,
            status="failed",
        )
        return ContextCompactionResult(status="failed", error=str(exc))

    session_memory.context_summary = summary
    session_memory.compacted_message_count = target_compacted_count
    _emit_compaction_usage(
        callbacks,
        messages=messages,
        settings=settings,
        include_reasoning=include_reasoning,
        count_message_tokens=count_message_tokens,
        status="done",
    )
    return ContextCompactionResult(
        status="done",
        compacted_message_count=target_compacted_count,
    )


def _max_context_per(settings: Settings) -> float:
    try:
        value = float(settings.max_context_per)
    except (TypeError, ValueError):
        value = 0.8
    return min(1.0, max(0.0, value))


def _emit_compaction_usage(
    callbacks: list[Any],
    *,
    messages: Sequence[BaseMessage],
    settings: Settings,
    include_reasoning: bool,
    count_message_tokens: MessageTokenCounter | None,
    status: Literal["running", "done", "failed"],
) -> None:
    max_context_tokens = settings.llm_num_ctx if settings.llm_num_ctx > 0 else None
    emit_context_usage_event(
        callbacks,
        build_context_usage_snapshot(
            messages,
            max_context_tokens=max_context_tokens,
            reserved_response_tokens=reserved_response_tokens_for_settings(
                settings,
                include_reasoning=include_reasoning,
            ),
            context_window_source="settings" if max_context_tokens else "unavailable",
            compaction_status=status,
            count_message_tokens=count_message_tokens,
        ),
    )


def _summary_messages(
    *,
    existing_summary: str,
    history: list[dict[str, Any]],
) -> list[BaseMessage]:
    sections: list[str] = []
    if existing_summary.strip():
        sections.append(f"Existing compressed context:\n{existing_summary.strip()}")
    sections.append("Conversation messages to compress:\n" + _history_lines(history))
    return [
        SystemMessage(
            content=(
                "Summarize prior chat context for a data-analysis agent. "
                "Keep goals, decisions, important facts, artifacts, constraints, and open questions. "
                "Do not include greetings, filler, or chain-of-thought. Return concise Markdown."
            )
        ),
        HumanMessage(content="\n\n".join(sections)),
    ]


def _history_lines(history: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in history:
        role = "user" if str(item.get("role", "")).lower() == "user" else "assistant"
        content = str(item.get("content", "") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content or "").strip()
