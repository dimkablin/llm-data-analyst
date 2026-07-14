from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.agent.callbacks import ContextUsageCollector
from backend.agent.context_compaction import compact_context_if_needed
from backend.core.config import Settings
from backend.sessions.session_memory import StructuredSessionMemory


def _history(count: int) -> list[dict[str, str]]:
    return [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"history-marker-{index}",
        }
        for index in range(count)
    ]


def test_context_compaction_skips_below_threshold() -> None:
    memory = StructuredSessionMemory()
    settings = replace(Settings(), llm_num_ctx=100, max_context_per=0.8)

    result = compact_context_if_needed(
        messages=[SystemMessage(content="system"), HumanMessage(content="prompt")],
        history=_history(8),
        settings=settings,
        session_memory=memory,
        callbacks=[],
        count_message_tokens=lambda _messages: 79,
    )

    assert result.status == "idle"
    assert memory.context_summary == ""
    assert memory.compacted_message_count == 0


def test_context_compaction_summarizes_history_above_threshold() -> None:
    class FakeLLM:
        def invoke(self, messages):
            text = "\n".join(str(message.content) for message in messages)
            assert "history-marker-0" in text
            assert "history-marker-1" in text
            assert "history-marker-5" not in text
            return AIMessage(content="Compressed session facts")

    memory = StructuredSessionMemory()
    collector = ContextUsageCollector()
    settings = replace(Settings(), llm_num_ctx=100, max_context_per=0.8)

    with patch("backend.agent.context_compaction.build_runtime_llm", return_value=FakeLLM()):
        result = compact_context_if_needed(
            messages=[SystemMessage(content="system"), HumanMessage(content="prompt")],
            history=_history(6),
            settings=settings,
            session_memory=memory,
            callbacks=[collector],
            count_message_tokens=lambda _messages: 81,
        )

    assert result.status == "done"
    assert result.compacted_message_count == 2
    assert memory.context_summary == "Compressed session facts"
    assert memory.compacted_message_count == 2
    assert [snap["compaction_status"] for snap in collector.snapshots] == ["running", "done"]


def test_context_compaction_keeps_history_when_summary_llm_fails() -> None:
    class FailingLLM:
        def invoke(self, _messages):
            raise RuntimeError("summary unavailable")

    memory = StructuredSessionMemory()
    settings = replace(Settings(), llm_num_ctx=100, max_context_per=0.8)

    with patch("backend.agent.context_compaction.build_runtime_llm", return_value=FailingLLM()):
        result = compact_context_if_needed(
            messages=[SystemMessage(content="system"), HumanMessage(content="prompt")],
            history=_history(6),
            settings=settings,
            session_memory=memory,
            callbacks=[],
            count_message_tokens=lambda _messages: 81,
        )

    assert result.status == "failed"
    assert memory.context_summary == ""
    assert memory.compacted_message_count == 0
