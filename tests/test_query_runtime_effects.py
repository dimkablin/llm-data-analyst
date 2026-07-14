from __future__ import annotations

from types import SimpleNamespace

from backend.agent.models import AgentResponse, AgentRuntimeEffects
from backend.api.services.query_execution import (
    QueryExecutionDependencies,
    QueryExecutionService,
)
from backend.sessions.session_memory import StructuredSessionMemory


class _FakeMemoryService:
    def __init__(self) -> None:
        self.scheduled: list[tuple[int, list[str], object]] = []

    def schedule_consolidation(self, user_id: int, notes: list[str], invoke) -> None:
        self.scheduled.append((user_id, notes, invoke))


class _FakeStore:
    def __init__(self) -> None:
        self.persisted: list[tuple[str, StructuredSessionMemory]] = []

    def set_structured_memory(
        self,
        session_id: str,
        memory: StructuredSessionMemory,
    ) -> None:
        self.persisted.append((session_id, memory))

def test_persist_runtime_effects_schedules_and_persists_public_contract(monkeypatch) -> None:
    invoke = object()
    monkeypatch.setattr(
        QueryExecutionService,
        "_build_memory_consolidation_invoker",
        staticmethod(lambda _settings: invoke),
    )

    memory = StructuredSessionMemory(notes="existing")
    response = AgentResponse(
        final_text="ok",
        reasoning=None,
        artifacts=[],
        runtime_effects=AgentRuntimeEffects(
            user_memory_notes=("remember this",),
            session_memory_notes=("session note",),
            session_memory=memory,
        ),
    )
    service = _FakeMemoryService()
    store = _FakeStore()
    execution_service = QueryExecutionService(
        dependencies=QueryExecutionDependencies.model_construct(
            user_memory_service=service,
            store=store,
        )
    )

    execution_service._persist_runtime_effects(
        response,
        user_id=42,
        session_id="session-1",
        runtime_settings=SimpleNamespace(),
    )

    assert service.scheduled == [(42, ["remember this"], invoke)]
    assert store.persisted == [("session-1", memory)]
