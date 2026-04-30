from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from backend.agent_graph.llm import MessageBuilder
from backend.agent_graph.state import MessageState


def message_to_state(message: BaseMessage) -> MessageState:
    role = _message_role(message)
    state: MessageState = {
        "role": role,
        "content": MessageBuilder.content_to_text(getattr(message, "content", "")),
    }

    name = getattr(message, "name", None)
    if isinstance(name, str) and name:
        state["name"] = name

    tool_call_id = getattr(message, "tool_call_id", None)
    if isinstance(tool_call_id, str) and tool_call_id:
        state["tool_call_id"] = tool_call_id

    tool_calls = getattr(message, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        state["tool_calls"] = [dict(item) for item in tool_calls if isinstance(item, dict)]

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict) and additional_kwargs:
        state["additional_kwargs"] = dict(additional_kwargs)

    return state


def state_to_message(state: MessageState) -> BaseMessage:
    role = state.get("role")
    content = str(state.get("content") or "")
    additional_kwargs = dict(state.get("additional_kwargs") or {})

    if role == "system":
        return SystemMessage(content=content, additional_kwargs=additional_kwargs)
    if role == "assistant":
        return AIMessage(
            content=content,
            tool_calls=list(state.get("tool_calls") or []),
            additional_kwargs=additional_kwargs,
        )
    if role == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=str(state.get("tool_call_id") or ""),
            additional_kwargs=additional_kwargs,
        )
    return HumanMessage(content=content, additional_kwargs=additional_kwargs)


def states_to_messages(states: list[MessageState]) -> list[BaseMessage]:
    return [state_to_message(state) for state in states]


def messages_to_states(messages: list[BaseMessage]) -> list[MessageState]:
    return [message_to_state(message) for message in messages]


def _message_role(message: BaseMessage) -> str:
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, ToolMessage):
        return "tool"
    return "user"
