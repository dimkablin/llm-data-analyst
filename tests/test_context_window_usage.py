from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from backend.agent.callbacks import ContextUsageCollector
from backend.agent.context_manager import ContextBudget
from backend.agent.context_window import (
    ContextUsageSnapshot,
    build_context_usage_snapshot,
    trim_context_messages,
)
from backend.agent.tool_loop import ToolLoopRequest, direct_tool_loop
from backend.core.config import Settings


def _message_count_tokens(messages) -> int:
    return sum(len(str(message.content).split()) for message in messages)


def _loop_messages(
    system_prompt: str,
    prompt: str,
    history: list[dict[str, str]] | None = None,
) -> list:
    messages: list = [SystemMessage(content=system_prompt)]
    for item in history or []:
        content = str(item.get("content", ""))
        if item.get("role") == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    messages.append(HumanMessage(content=prompt))
    return messages


def _tool_exchange(
    tool_name: str,
    call_id: str,
    content: str,
    *,
    status: str | None = None,
) -> list:
    tool_message = ToolMessage(content=content, name=tool_name, tool_call_id=call_id)
    if status is not None:
        tool_message = tool_message.model_copy(update={"status": status})
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": tool_name,
                    "args": {},
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        ),
        tool_message,
    ]


def test_context_usage_snapshot_accepts_message_token_counter() -> None:
    messages = [
        SystemMessage(content="system prompt"),
        HumanMessage(content="user question"),
    ]

    snapshot = build_context_usage_snapshot(
        messages,
        max_context_tokens=20,
        reserved_response_tokens=4,
        context_window_source="settings",
        compaction_status="running",
        count_message_tokens=lambda seen_messages: len(seen_messages) * 3,
    )

    assert snapshot.input_tokens == 6
    assert snapshot.used_tokens == 10
    assert snapshot.compaction_status == "running"


def test_context_usage_snapshot_counts_prompt_and_reserved_response_tokens() -> None:
    messages = [
        SystemMessage(content="system prompt"),
        HumanMessage(content="user question"),
    ]

    snapshot = build_context_usage_snapshot(
        messages,
        max_context_tokens=20,
        reserved_response_tokens=4,
        context_window_source="settings",
        count_message_tokens=_message_count_tokens,
    )

    assert isinstance(snapshot, ContextUsageSnapshot)
    assert snapshot.input_tokens == _message_count_tokens(messages)
    assert snapshot.reserved_response_tokens == 4
    assert snapshot.used_tokens == snapshot.input_tokens + 4
    assert snapshot.remaining_tokens == 20 - snapshot.used_tokens
    assert snapshot.usage_ratio == snapshot.used_tokens / 20
    assert snapshot.usage_percent == round(snapshot.usage_ratio * 100)
    assert snapshot.status == "normal"
    assert snapshot.overflow is False
    assert snapshot.context_window_source == "settings"


def test_context_usage_snapshot_uses_plan_thresholds_and_clamps_percent() -> None:
    assert (
        build_context_usage_snapshot(
            [],
            max_context_tokens=100,
            reserved_response_tokens=0,
            context_window_source="settings",
            count_message_tokens=lambda _messages: 75,
        ).status
        == "warning"
    )
    assert (
        build_context_usage_snapshot(
            [],
            max_context_tokens=100,
            reserved_response_tokens=0,
            context_window_source="settings",
            count_message_tokens=lambda _messages: 90,
        ).status
        == "critical"
    )

    overflow = build_context_usage_snapshot(
        [],
        max_context_tokens=100,
        reserved_response_tokens=0,
        context_window_source="settings",
        count_message_tokens=lambda _messages: 121,
    )

    assert overflow.usage_percent == 100
    assert overflow.status == "overflow"


def test_context_usage_snapshot_marks_overflow_when_reserved_tokens_exceed_window() -> None:
    snapshot = build_context_usage_snapshot(
        [HumanMessage(content="one two three")],
        max_context_tokens=4,
        reserved_response_tokens=3,
        context_window_source="settings",
        count_message_tokens=_message_count_tokens,
    )

    assert snapshot.overflow is True
    assert snapshot.remaining_tokens == 0
    assert snapshot.status == "overflow"


def test_context_usage_snapshot_is_unavailable_without_context_limit() -> None:
    snapshot = build_context_usage_snapshot(
        [HumanMessage(content="one two three")],
        max_context_tokens=None,
        reserved_response_tokens=3,
        context_window_source="unavailable",
        count_message_tokens=_message_count_tokens,
    )

    assert snapshot.max_context_tokens is None
    assert snapshot.usage_ratio is None
    assert snapshot.usage_percent is None
    assert snapshot.remaining_tokens is None
    assert snapshot.status == "unavailable"


def test_context_budget_accepts_usage_metadata() -> None:
    budget = ContextBudget(
        strategy="token_limit",
        status="planned",
        max_context_tokens=100,
        reserved_response_tokens=20,
        estimated_context_tokens=60,
        usage_ratio=0.8,
        usage_percent=80,
        remaining_context_tokens=20,
        overflow=False,
    )

    assert budget.usage_percent == 80
    assert budget.remaining_context_tokens == 20


def test_trim_context_messages_preserves_latest_human_prompt_after_large_tool_output() -> None:
    messages = [
        SystemMessage(content="system " * 100),
        HumanMessage(content="current user question"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "rag_tool",
                    "args": {},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="rag result " * 2000, tool_call_id="call-1"),
    ]

    trimmed = trim_context_messages(
        messages,
        max_input_tokens=50,
        count_message_tokens=_message_count_tokens,
    )

    assert any(
        isinstance(message, HumanMessage) and message.content == "current user question"
        for message in trimmed
    )


def test_trim_context_messages_preserves_latest_tool_exchange_after_large_tool_output() -> None:
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="current user question"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "rag_tool",
                    "args": {},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="rag result " * 2000, tool_call_id="call-1"),
    ]

    trimmed = trim_context_messages(
        messages,
        max_input_tokens=30,
        count_message_tokens=_message_count_tokens,
    )

    tool_call_ids = {
        tool_call["id"]
        for message in trimmed
        for tool_call in (getattr(message, "tool_calls", None) or [])
    }
    tool_message_ids = {
        message.tool_call_id for message in trimmed if isinstance(message, ToolMessage)
    }

    assert "call-1" in tool_call_ids
    assert tool_message_ids == {"call-1"}
    assert _message_count_tokens(trimmed) <= 30
    assert any(
        isinstance(message, ToolMessage)
        and "tool result compacted" in str(message.content)
        for message in trimmed
    )


def test_trim_context_messages_preserves_only_latest_valid_analysis_plan() -> None:
    first_plan = json.dumps(
        {"plan": [{"step": "Inspect the original source", "status": "in_progress"}]}
    )
    latest_plan = json.dumps(
        {
            "plan": [
                {"step": "Inspect the discovered relationship", "status": "completed"},
                {"step": "Calculate the requested comparison", "status": "in_progress"},
            ]
        }
    )
    messages = [SystemMessage(content="system"), HumanMessage(content="analyze")]
    for call_id, tool_name, content in [
        ("plan-1", "update_plan", first_plan),
        ("sql-1", "sql_tool", "large result " * 200),
        ("plan-2", "update_plan", latest_plan),
        ("pandas-1", "pandas_tool", "another result " * 200),
    ]:
        messages.extend(_tool_exchange(tool_name, call_id, content))

    trimmed = trim_context_messages(
        messages,
        max_input_tokens=45,
        count_message_tokens=_message_count_tokens,
    )
    results = {
        message.tool_call_id: str(message.content)
        for message in trimmed
        if isinstance(message, ToolMessage)
    }

    assert results["plan-1"] == "[tool result compacted because context limit was reached]"
    assert results["plan-2"] == latest_plan
    assert results["sql-1"] == "[tool result compacted because context limit was reached]"
    assert results["pandas-1"] == "[tool result compacted because context limit was reached]"


def test_invalid_plan_result_does_not_replace_latest_valid_plan_during_compaction() -> None:
    valid_plan = json.dumps(
        {"plan": [{"step": "Inspect the source", "status": "in_progress"}]}
    )
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="analyze"),
        *_tool_exchange("update_plan", "plan-1", valid_plan),
        *_tool_exchange("update_plan", "plan-2", "plan validation failed", status="error"),
        *_tool_exchange("sql_tool", "sql-1", "large result " * 200),
    ]

    trimmed = trim_context_messages(
        messages,
        max_input_tokens=35,
        count_message_tokens=_message_count_tokens,
    )
    results = {
        message.tool_call_id: str(message.content)
        for message in trimmed
        if isinstance(message, ToolMessage)
    }

    assert results["plan-1"] == valid_plan
    assert results["plan-2"] == "[tool result compacted because context limit was reached]"


def test_direct_tool_loop_emits_context_usage_before_model_response() -> None:
    class FakeLLM:
        def bind_tools(self, _tools):
            return self

        def get_num_tokens_from_messages(self, messages):
            return _message_count_tokens(messages)

        def invoke(self, _messages, config=None):
            del config
            return AIMessage(content="Готово")

    settings = replace(
        Settings(),
        llm_provider="vllm",
        llm_num_ctx=1000,
        llm_max_tokens_default=77,
        llm_warmup_enabled=False,
    )
    collector = ContextUsageCollector()

    with patch("backend.agent.tool_loop.build_runtime_llm", return_value=FakeLLM()):
        response = direct_tool_loop(
            ToolLoopRequest(
                settings=settings,
                include_reasoning=False,
                tools=[],
                callbacks=[collector],
                max_iterations=1,
                trace_context={"session_id": "stream-context-usage"},
                messages=_loop_messages("system", "prompt"),
            )
        )

    assert response.final_text == "Готово"
    assert collector.snapshots
    assert collector.snapshots[0]["max_context_tokens"] == 1000
    assert collector.snapshots[0]["reserved_response_tokens"] == 77


def test_direct_tool_loop_reserves_bound_tool_schema_tokens() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.kwargs = {}
            self.invoked_messages = []

        def bind_tools(self, _tools):
            self.kwargs = {"tools": [{"type": "function", "function": {"name": "fake_tool"}}]}
            return self

        def get_num_tokens(self, _text):
            return 200

        def get_num_tokens_from_messages(self, messages):
            return _message_count_tokens(messages)

        def invoke(self, messages, config=None):
            del config
            self.invoked_messages = list(messages)
            assert _message_count_tokens(messages) <= 188
            return AIMessage(content="Done")

    @tool
    def fake_tool() -> str:
        """Return a fake result."""
        return "ok"

    settings = replace(
        Settings(),
        llm_provider="vllm",
        llm_num_ctx=1000,
        llm_max_tokens_default=100,
        llm_warmup_enabled=False,
    )
    collector = ContextUsageCollector()
    fake_llm = FakeLLM()
    messages = _loop_messages(
        "system",
        "current prompt",
        [
            {"role": "user", "content": "old1 " * 100},
            {"role": "assistant", "content": "old2 " * 100},
            {"role": "user", "content": "old3 " * 100},
            {"role": "assistant", "content": "old4 " * 100},
        ],
    )

    with patch("backend.agent.tool_loop.build_runtime_llm", return_value=fake_llm):
        response = direct_tool_loop(
            ToolLoopRequest(
                settings=settings,
                include_reasoning=False,
                tools=[fake_tool],
                callbacks=[collector],
                max_iterations=1,
                trace_context={"session_id": "context-tool-schema-budget"},
                messages=messages,
            )
        )

    assert response.final_text == "Done"
    assert collector.snapshots[0]["input_tokens"] == (
        _message_count_tokens(fake_llm.invoked_messages) + 200
    )
    assert collector.snapshots[0]["used_tokens"] <= 1000 - 512


def test_direct_tool_loop_preserves_tool_result_when_trimming_before_next_model_call() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.calls = 0

        def bind_tools(self, _tools):
            return self

        def get_num_tokens_from_messages(self, messages):
            return _message_count_tokens(messages)

        def invoke(self, messages, config=None):
            del config
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "fake_tool",
                            "args": {},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                )

            tool_call_ids = {
                tool_call["id"]
                for message in messages
                for tool_call in (getattr(message, "tool_calls", None) or [])
            }
            tool_message_ids = {
                message.tool_call_id for message in messages if isinstance(message, ToolMessage)
            }
            assert tool_call_ids == {"call-1"}
            assert tool_message_ids == {"call-1"}
            return AIMessage(content="Done")

    @tool
    def fake_tool() -> str:
        """Return a large fake tool result."""
        return "large tool result " * 2000

    settings = replace(
        Settings(),
        llm_provider="vllm",
        llm_num_ctx=600,
        llm_max_tokens_default=20,
        llm_warmup_enabled=False,
    )
    collector = ContextUsageCollector()
    fake_llm = FakeLLM()

    with patch("backend.agent.tool_loop.build_runtime_llm", return_value=fake_llm):
        response = direct_tool_loop(
            ToolLoopRequest(
                settings=settings,
                include_reasoning=False,
                tools=[fake_tool],
                callbacks=[collector],
                max_iterations=2,
                trace_context={"session_id": "context-tool-trim"},
                messages=_loop_messages("system", "current prompt"),
            )
        )

    assert response.final_text == "Done"
    assert len(collector.snapshots) == 2
    assert collector.snapshots[-1]["input_tokens"] > collector.snapshots[0]["input_tokens"]


def test_direct_tool_loop_still_trims_history_when_reserved_tokens_exceed_window() -> None:
    class FakeLLM:
        def bind_tools(self, _tools):
            return self

        def get_num_tokens_from_messages(self, messages):
            return _message_count_tokens(messages)

        def invoke(self, messages, config=None):
            del config
            text = "\n".join(str(message.content) for message in messages)
            assert "old_history_marker" not in text
            assert "current_prompt_marker" in text
            return AIMessage(content="Done")

    settings = replace(
        Settings(),
        llm_provider="vllm",
        llm_num_ctx=100,
        llm_max_tokens_default=200,
        llm_warmup_enabled=False,
        agent_history_max_messages=8,
    )

    with patch("backend.agent.tool_loop.build_runtime_llm", return_value=FakeLLM()):
        response = direct_tool_loop(
            ToolLoopRequest(
                settings=settings,
                include_reasoning=False,
                tools=[],
                callbacks=[],
                max_iterations=1,
                trace_context={"session_id": "context-negative-budget"},
                messages=_loop_messages(
                    "system",
                    "current_prompt_marker",
                    [{"role": "user", "content": "old_history_marker " + "old " * 100}],
                ),
            )
        )

    assert response.final_text == "Done"


def test_direct_tool_loop_trims_old_history_before_model_response() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.invoked_messages = []

        def bind_tools(self, _tools):
            return self

        def get_num_tokens_from_messages(self, messages):
            return _message_count_tokens(messages)

        def invoke(self, messages, config=None):
            del config
            self.invoked_messages = list(messages)
            return AIMessage(content="Готово")

    old_history = "old_history_marker " + " ".join(f"old{i}" for i in range(80))
    latest_history = "latest_history_marker short"
    latest_prompt = "current_prompt_marker"
    settings = replace(
        Settings(),
        llm_provider="vllm",
        llm_num_ctx=600,
        llm_max_tokens_default=20,
        llm_warmup_enabled=False,
        agent_history_max_messages=4,
    )
    fake_llm = FakeLLM()

    with patch("backend.agent.tool_loop.build_runtime_llm", return_value=fake_llm):
        response = direct_tool_loop(
            ToolLoopRequest(
                settings=settings,
                include_reasoning=False,
                tools=[],
                callbacks=[],
                max_iterations=1,
                trace_context={"session_id": "stream-context-trim"},
                messages=_loop_messages(
                    "system_anchor",
                    latest_prompt,
                    [
                        {"role": "user", "content": old_history},
                        {"role": "assistant", "content": "old assistant reply"},
                        {"role": "user", "content": latest_history},
                    ],
                ),
            )
        )

    invoked_text = "\n".join(str(message.content) for message in fake_llm.invoked_messages)
    assert response.final_text == "Готово"
    assert "system_anchor" in invoked_text
    assert latest_prompt in invoked_text
    assert "old_history_marker" not in invoked_text
