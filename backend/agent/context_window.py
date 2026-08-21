from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from pydantic import BaseModel, Field

ContextUsageStatus = Literal["unavailable", "normal", "warning", "critical", "overflow"]
ContextCompactionStatus = Literal["idle", "running", "done", "failed"]
ContextWindowSource = Literal["settings", "unavailable"]

_WARNING_USAGE_RATIO = 0.75
_CRITICAL_USAGE_RATIO = 0.90
_COMPACT_TOOL_RESULT_TEXT = "[tool result compacted because context limit was reached]"
MessageTokenCounter = Callable[[Sequence[BaseMessage]], int]


class ContextUsageSnapshot(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    reserved_response_tokens: int = Field(default=0, ge=0)
    used_tokens: int = Field(default=0, ge=0)
    max_context_tokens: int | None = Field(default=None, ge=1)
    remaining_tokens: int | None = Field(default=None, ge=0)
    usage_ratio: float | None = Field(default=None, ge=0.0)
    usage_percent: int | None = Field(default=None, ge=0, le=100)
    overflow: bool = False
    status: ContextUsageStatus = "unavailable"
    context_window_source: ContextWindowSource = "unavailable"
    compaction_status: ContextCompactionStatus = "idle"


def estimate_message_tokens(
    messages: Sequence[BaseMessage],
    *,
    count_message_tokens: MessageTokenCounter | None = None,
) -> int:
    if count_message_tokens is not None:
        try:
            return max(0, int(count_message_tokens(messages)))
        except Exception:
            pass
    return max(0, int(count_tokens_approximately(messages)))


def trim_context_messages(
    messages: Sequence[BaseMessage],
    *,
    max_input_tokens: int,
    count_message_tokens: MessageTokenCounter | None = None,
) -> list[BaseMessage]:
    if max_input_tokens <= 0:
        return list(messages)

    def token_counter(seen_messages: list[BaseMessage]) -> int:
        return estimate_message_tokens(
            seen_messages,
            count_message_tokens=count_message_tokens,
        )

    try:
        trimmed_messages = trim_messages(
            list(messages),
            max_tokens=max_input_tokens,
            token_counter=token_counter,
            strategy="last",
            include_system=True,
            start_on="human",
        )
    except Exception:
        return list(messages)

    trimmed_messages = _drop_invalid_tool_call_sequences(trimmed_messages)

    preserved_messages = _preserve_current_tool_exchange(
        messages,
        trimmed_messages,
        max_input_tokens=max_input_tokens,
        count_message_tokens=count_message_tokens,
    )

    if _latest_human_message(messages) is not None and not _has_human_message(preserved_messages):
        return _minimal_valid_messages(
            messages,
            max_input_tokens=max_input_tokens,
            count_message_tokens=count_message_tokens,
        )

    return preserved_messages


def _latest_human_message(messages: Sequence[BaseMessage]) -> HumanMessage | None:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    return None


def _has_human_message(messages: Sequence[BaseMessage]) -> bool:
    return any(isinstance(message, HumanMessage) for message in messages)


def _first_system_message(messages: Sequence[BaseMessage]) -> SystemMessage | None:
    if messages and isinstance(messages[0], SystemMessage):
        return messages[0]
    return None


def _minimal_valid_messages(
    messages: Sequence[BaseMessage],
    *,
    max_input_tokens: int,
    count_message_tokens: MessageTokenCounter | None,
) -> list[BaseMessage]:
    latest_human_message = _latest_human_message(messages)
    if latest_human_message is None:
        return list(messages)

    system_message = _first_system_message(messages)
    if system_message is None:
        return [latest_human_message]

    candidate = [system_message, latest_human_message]
    if (
        estimate_message_tokens(candidate, count_message_tokens=count_message_tokens)
        <= max_input_tokens
    ):
        return candidate
    return [latest_human_message]


def _tool_call_ids(message: BaseMessage) -> list[str]:
    if not isinstance(message, AIMessage):
        return []
    ids: list[str] = []
    for tool_call in getattr(message, "tool_calls", None) or []:
        if isinstance(tool_call, dict):
            tool_call_id = str(tool_call.get("id", "")).strip()
        else:
            tool_call_id = str(getattr(tool_call, "id", "") or "").strip()
        if tool_call_id:
            ids.append(tool_call_id)
    return ids


def _tool_message_ids(messages: Sequence[BaseMessage]) -> set[str]:
    return {
        str(message.tool_call_id).strip()
        for message in messages
        if isinstance(message, ToolMessage) and str(message.tool_call_id).strip()
    }


def _ai_tool_call_ids(messages: Sequence[BaseMessage]) -> set[str]:
    return {tool_call_id for message in messages for tool_call_id in _tool_call_ids(message)}


def _complete_tool_groups(messages: Sequence[BaseMessage]) -> list[list[BaseMessage]]:
    groups: list[list[BaseMessage]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        tool_call_ids = _tool_call_ids(message)
        if not tool_call_ids:
            index += 1
            continue

        group: list[BaseMessage] = [message]
        seen_ids: set[str] = set()
        index += 1
        while index < len(messages) and isinstance(messages[index], ToolMessage):
            tool_message = messages[index]
            tool_call_id = str(tool_message.tool_call_id).strip()
            if tool_call_id in tool_call_ids and tool_call_id not in seen_ids:
                group.append(tool_message)
                seen_ids.add(tool_call_id)
            index += 1

        if seen_ids == set(tool_call_ids):
            groups.append(group)

    return groups


def _drop_invalid_tool_call_sequences(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    valid_group_messages = {
        id(message)
        for group in _complete_tool_groups(messages)
        for message in group
    }
    cleaned: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            if id(message) in valid_group_messages:
                cleaned.append(message)
            continue
        if _tool_call_ids(message) and id(message) not in valid_group_messages:
            continue
        cleaned.append(message)
    return cleaned


def _latest_human_index(messages: Sequence[BaseMessage]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return index
    return None


def _current_turn_tool_exchange(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    human_index = _latest_human_index(messages)
    if human_index is None:
        return []

    exchange: list[BaseMessage] = []
    for group in _complete_tool_groups(messages[human_index + 1:]):
        exchange.extend(group)
    return exchange


def _preserve_current_tool_exchange(
    messages: Sequence[BaseMessage],
    trimmed_messages: Sequence[BaseMessage],
    *,
    max_input_tokens: int,
    count_message_tokens: MessageTokenCounter | None,
) -> list[BaseMessage]:
    current_exchange = _current_turn_tool_exchange(messages)
    if not current_exchange:
        return list(trimmed_messages)

    required_ids = _ai_tool_call_ids(current_exchange)
    if (
        required_ids
        and required_ids.issubset(_ai_tool_call_ids(trimmed_messages))
        and required_ids.issubset(_tool_message_ids(trimmed_messages))
    ):
        return list(trimmed_messages)

    latest_human_message = _latest_human_message(messages)
    if latest_human_message is None:
        return list(trimmed_messages)

    base: list[BaseMessage] = []
    system_message = _first_system_message(messages)
    if system_message is not None:
        base.append(system_message)
    base.append(latest_human_message)
    base.extend(current_exchange)
    return _fit_tool_exchange_messages(
        base,
        fallback_messages=messages,
        max_input_tokens=max_input_tokens,
        count_message_tokens=count_message_tokens,
    )


def _fit_tool_exchange_messages(
    messages: Sequence[BaseMessage],
    *,
    fallback_messages: Sequence[BaseMessage],
    max_input_tokens: int,
    count_message_tokens: MessageTokenCounter | None,
) -> list[BaseMessage]:
    candidates = [
        list(messages),
        _compact_tool_exchange_messages(messages),
    ]
    if candidates[-1] and isinstance(candidates[-1][0], SystemMessage):
        candidates.append(candidates[-1][1:])

    for candidate in candidates:
        candidate = _drop_invalid_tool_call_sequences(candidate)
        if (
            estimate_message_tokens(candidate, count_message_tokens=count_message_tokens)
            <= max_input_tokens
        ):
            return candidate

    return _minimal_valid_messages(
        fallback_messages,
        max_input_tokens=max_input_tokens,
        count_message_tokens=count_message_tokens,
    )


def _compact_tool_exchange_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    latest_plan_call_id = next(
        (
            str(message.tool_call_id).strip()
            for message in reversed(messages)
            if _is_plan_result(message)
        ),
        "",
    )
    return [
        message
        if isinstance(message, ToolMessage)
        and str(message.tool_call_id).strip() == latest_plan_call_id
        else _compact_tool_message(message)
        for message in messages
    ]


def _is_plan_result(message: BaseMessage) -> bool:
    if not isinstance(message, ToolMessage) or getattr(message, "name", None) != "update_plan":
        return False
    try:
        payload = json.loads(str(message.content))
    except (TypeError, ValueError):
        return False
    plan = payload.get("plan") if isinstance(payload, dict) else None
    return bool(plan) and all(
        isinstance(item, dict)
        and str(item.get("step") or "").strip()
        and item.get("status") in {"pending", "in_progress", "completed"}
        for item in plan
    )


def _compact_tool_message(message: BaseMessage) -> BaseMessage:
    if not isinstance(message, ToolMessage) or getattr(message, "name", None) in {
        "planner_tool",
        "get_tool_instructions",
    }:
        return message
    return message.model_copy(update={"content": _COMPACT_TOOL_RESULT_TEXT})


def build_context_usage_snapshot(
    messages: Sequence[BaseMessage],
    *,
    max_context_tokens: int | None,
    reserved_response_tokens: int,
    context_window_source: ContextWindowSource,
    compaction_status: ContextCompactionStatus = "idle",
    count_message_tokens: MessageTokenCounter | None = None,
) -> ContextUsageSnapshot:
    input_tokens = estimate_message_tokens(
        list(messages),
        count_message_tokens=count_message_tokens,
    )
    reserved_tokens = max(0, int(reserved_response_tokens))
    used_tokens = input_tokens + reserved_tokens

    if max_context_tokens is None:
        return ContextUsageSnapshot(
            input_tokens=input_tokens,
            reserved_response_tokens=reserved_tokens,
            used_tokens=used_tokens,
            max_context_tokens=None,
            context_window_source=context_window_source,
            compaction_status=compaction_status,
        )

    limit = max(1, int(max_context_tokens))
    overflow = used_tokens > limit
    remaining_tokens = max(0, limit - used_tokens)
    usage_ratio = used_tokens / limit
    return ContextUsageSnapshot(
        input_tokens=input_tokens,
        reserved_response_tokens=reserved_tokens,
        used_tokens=used_tokens,
        max_context_tokens=limit,
        remaining_tokens=remaining_tokens,
        usage_ratio=usage_ratio,
        usage_percent=min(100, round(usage_ratio * 100)),
        overflow=overflow,
        status=_status_from_ratio(usage_ratio, overflow=overflow),
        context_window_source=context_window_source,
        compaction_status=compaction_status,
    )


def reserved_response_tokens_for_settings(settings: Any, *, include_reasoning: bool) -> int:
    field_name = "llm_max_tokens_reasoning" if include_reasoning else "llm_max_tokens_default"
    try:
        return max(0, int(getattr(settings, field_name)))
    except (TypeError, ValueError):
        return 0


def _status_from_ratio(usage_ratio: float, *, overflow: bool) -> ContextUsageStatus:
    if overflow:
        return "overflow"
    if usage_ratio >= _CRITICAL_USAGE_RATIO:
        return "critical"
    if usage_ratio >= _WARNING_USAGE_RATIO:
        return "warning"
    return "normal"
