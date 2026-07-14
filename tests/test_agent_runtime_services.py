from __future__ import annotations

from pydantic import BaseModel

from backend.agent.services.chat_title import ChatTitleRequest, ChatTitleService
from backend.agent.services.runtime_effects import (
    RuntimeEffectsBuilder,
    RuntimeEffectsRequest,
)
from backend.core.config import Settings
from backend.sessions.session_memory import SessionMemory


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeTitleLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[object] = []

    def invoke(self, messages, config=None):
        self.calls.append((messages, config))
        return _FakeMessage(self.content)


def test_chat_title_service_is_pydantic_contract_and_normalizes_llm_output(
    monkeypatch,
) -> None:
    fake_llm = _FakeTitleLLM('"Quarterly Sales Variance!"')

    monkeypatch.setattr(
        "backend.agent.services.chat_title.build_runtime_llm",
        lambda *_args, **_kwargs: fake_llm,
    )
    monkeypatch.setattr(
        "backend.agent.services.chat_title.record_llm_usage_on_active_span",
        lambda *_args, **_kwargs: None,
    )

    service = ChatTitleService(settings=Settings(backend_query_timeout_sec=30))
    title = service.generate(
        ChatTitleRequest(
            dataset_name="sales.csv",
            user_queries=["show variance by region", "explain the biggest gap"],
            trace_context={"session_id": "s1", "request_kind": "title_generate"},
        )
    )

    assert isinstance(service, BaseModel)
    assert title == "Quarterly Sales Variance"
    assert fake_llm.calls


def test_chat_title_service_returns_none_without_user_queries() -> None:
    service = ChatTitleService(settings=Settings())

    title = service.generate(
        ChatTitleRequest(dataset_name="sales.csv", user_queries=[])
    )

    assert title is None


def test_runtime_effects_builder_merges_notes_without_mutating_source_memory() -> None:
    session_memory = SessionMemory(notes="existing")
    builder = RuntimeEffectsBuilder()

    effects = builder.build(
        RuntimeEffectsRequest(
            session_memory=session_memory,
            user_memory_notes=[" remember user ", "", "remember too"],
            session_memory_notes=[" session note ", ""],
        )
    )

    assert isinstance(builder, BaseModel)
    assert effects.user_memory_notes == ("remember user", "remember too")
    assert effects.session_memory_notes == ("session note",)
    assert effects.session_memory is not session_memory
    assert effects.session_memory.notes == "existing\nsession note"
    assert session_memory.notes == "existing"
