from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import ClassVar

from pydantic import BaseModel

from backend.agent.models import AgentResponse
from backend.agent.runtime_contracts import AgentRunRequest, AgentRunResult
from backend.api.models import QueryMetrics, QueryRequest, QueryResponse
from backend.api.routes import query as query_route
from backend.api.services.query_execution import (
    QueryExecutionDependencies,
    QueryExecutionRequest,
    QueryExecutionService,
    QueryStreamExecutionContext,
    QueryStreamExecutionRequest,
)
from backend.auth.auth_db import AuthDB, AuthUser, UserSettings
from backend.core.config import Settings
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.notebook.manifest_store import ManifestStore
from backend.sessions.session_store import SessionState, SessionStore
from backend.skills.models import Skill
from backend.skills.registry import SkillRegistry


def test_query_execution_service_contracts_are_pydantic_models() -> None:
    assert issubclass(QueryExecutionDependencies, BaseModel)
    assert issubclass(QueryExecutionRequest, BaseModel)
    assert issubclass(QueryStreamExecutionRequest, BaseModel)
    assert issubclass(QueryStreamExecutionContext, BaseModel)
    assert issubclass(QueryExecutionService, BaseModel)

    assert {
        "session_id",
        "payload",
        "current_user",
        "persist",
        "callbacks",
    }.issubset(QueryExecutionRequest.model_fields)
    assert {
        "session_id",
        "payload",
        "current_user",
    }.issubset(QueryStreamExecutionRequest.model_fields)
    assert {
        "auth_db",
        "store",
        "skill_registry",
        "agent_runner_cls",
        "llm_text_collector_cls",
        "tool_collector_cls",
        "csv_runtime",
        "manifest_store",
    }.issubset(QueryExecutionDependencies.model_fields)


def test_query_route_delegates_non_stream_execution_to_service() -> None:
    fake_service = _FakeRouteQueryService()
    old_service = query_route._query_execution_service
    query_route._query_execution_service = fake_service
    user = _auth_user()

    try:
        response = anyio_run(
            query_route.query(
                "session-1",
                QueryRequest(query="run analysis"),
                user,
            )
        )
        evaluate_response = anyio_run(
            query_route.evaluate(
                "session-2",
                QueryRequest(query="dry run"),
                user,
            )
        )
    finally:
        query_route._query_execution_service = old_service

    assert response.text == "route answer"
    assert evaluate_response.text == "route answer"
    assert [(req.session_id, req.payload.query, req.persist) for req in fake_service.executed] == [
        ("session-1", "run analysis", True),
        ("session-2", "dry run", False),
    ]


def test_query_route_delegates_stream_execution_to_service() -> None:
    fake_service = _FakeRouteQueryService()
    old_service = query_route._query_execution_service
    query_route._query_execution_service = fake_service

    try:
        response = anyio_run(
            query_route.query_stream(
                "session-1",
                QueryRequest(query="stream analysis"),
                _auth_user(),
            )
        )
        body = anyio_run(_drain_streaming_response(response))
    finally:
        query_route._query_execution_service = old_service

    assert response.media_type == "text/event-stream"
    assert fake_service.prepared[0].session_id == "session-1"
    assert fake_service.streamed_contexts == ["stream-context"]
    assert "event: final" in body
    assert "stream answer" in body


def test_query_execution_service_prepares_runner_for_execute_and_stream(tmp_path) -> None:
    service, _calls, runner_cls = _build_service_for_query_tests(tmp_path)

    response = anyio_run(
        service.execute(
            QueryExecutionRequest(
                session_id="session-1",
                payload=QueryRequest(query="run analysis", selected_skill_ids=["allowed"]),
                current_user=_auth_user(),
                persist=True,
            )
        )
    )
    context = service.prepare_stream(
        QueryStreamExecutionRequest(
            session_id="session-1",
            payload=QueryRequest(query="stream analysis", selected_skill_ids=["allowed"]),
            current_user=_auth_user(),
        )
    )

    execute_runner, stream_runner = runner_cls.instances[-2:]
    assert response.text == "service answer"
    assert execute_runner.run_request is not None
    assert execute_runner.run_request.prompt == "run analysis"
    assert context.query_runtime is stream_runner
    assert stream_runner.run_request is None
    assert execute_runner.kwargs["enabled_analytical_skill_ids"] == {"allowed"}
    assert stream_runner.kwargs["enabled_analytical_skill_ids"] == {"allowed"}


def test_query_execution_service_filters_skills_and_persists_response(tmp_path) -> None:
    session_id = "session-1"
    service, calls, runner_cls = _build_service_for_query_tests(
        tmp_path,
        state_selected_skill_ids=["allowed", "disabled", "default_disabled"],
        user_skill_settings={"allowed": True, "disabled": False},
        skills=[
            _skill(tmp_path, "allowed"),
            _skill(tmp_path, "disabled"),
            _skill(tmp_path, "default_disabled", enabled_by_default=False),
        ],
    )

    response = anyio_run(
        service.execute(
            QueryExecutionRequest(
                session_id=session_id,
                payload=QueryRequest(
                    query="run analysis",
                    selected_skill_ids=["allowed", "disabled", "default_disabled"],
                ),
                current_user=AuthUser(
                    id=7,
                    username="user",
                    is_admin=False,
                    created_at="2026-01-01T00:00:00+00:00",
                ),
                persist=True,
            )
        )
    )

    runner = runner_cls.instances[0]
    assert response.text == "service answer"
    assert runner.run_request is not None
    assert runner.run_request.selected_skill_ids == ["allowed"]
    assert runner.run_request.prompt == "run analysis"
    assert calls["selected_skill_ids"] == [(session_id, ["allowed"])]
    assert calls["messages"][0][:3] == (session_id, "user", "run analysis")
    assert calls["messages"][1][:3] == (session_id, "ai", "service answer")


def test_query_stream_prepare_raises_http_errors_before_generator(tmp_path) -> None:
    service, _calls, _runner_cls = _build_service_for_query_tests(tmp_path)
    service.dependencies.auth_db.is_owner = False

    try:
        service.prepare_stream(
            QueryStreamExecutionRequest(
                session_id="missing-session",
                payload=QueryRequest(query="stream analysis"),
                current_user=_auth_user(),
            )
        )
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 404
    else:
        raise AssertionError("prepare_stream must raise before StreamingResponse is created")


def test_query_stream_events_use_agent_run_contract_and_emit_final(tmp_path) -> None:
    service, calls, runner_cls = _build_service_for_query_tests(tmp_path)

    context = service.prepare_stream(
        QueryStreamExecutionRequest(
            session_id="session-1",
            payload=QueryRequest(query="stream analysis", selected_skill_ids=["allowed"]),
            current_user=_auth_user(),
        )
    )
    events = anyio_run(_collect_stream_events(service.stream_events(context)))

    assert events[0] == ("start", {"session_id": "session-1"})
    assert events[-1][0] == "final"
    assert events[-1][1]["text"] == "service answer"
    runner = runner_cls.instances[-1]
    assert isinstance(runner.run_request, AgentRunRequest)
    assert runner.run_request.prompt == "stream analysis"
    assert runner.run_request.selected_skill_ids == ["allowed"]
    assert calls["messages"][0][:3] == ("session-1", "user", "stream analysis")
    assert calls["messages"][1][:3] == ("session-1", "ai", "service answer")
    assert calls["context_usage"] == ("session-1", {"usage_percent": 60})


def test_query_stream_interrupted_turn_persists_partial_tool_history(tmp_path) -> None:
    service, calls, _runner_cls = _build_service_for_query_tests(tmp_path)
    token_collector = SimpleNamespace(
        collected_visible=lambda: None,
        collected_reasoning=lambda: None,
        all_reasoning_steps=lambda: [],
    )
    tool_collector = SimpleNamespace(
        tool_calls=1,
        events=[{"phase": "start", "tool_name": "sql_tool"}],
        artifacts=[],
        to_persisted_activities=lambda **_kwargs: [
            {
                "tool_name": "sql_tool",
                "status": "error",
                "input_summary": "SELECT 1",
                "output_preview": "Остановлено пользователем.",
            }
        ],
    )

    persisted = service._persist_interrupted_stream_turn(
        session_id="session-1",
        query_text="stream analysis",
        token_collector=token_collector,
        tool_collector=tool_collector,
        callbacks=[SimpleNamespace(snapshots=[{"usage_percent": 44}])],
        selected_skill_ids=["allowed"],
        initial_history_len=1,
    )

    assert persisted is True
    assert calls["messages"][0][:3] == ("session-1", "user", "stream analysis")
    assert calls["messages"][1][0:2] == ("session-1", "ai")
    assert "Остановлено пользователем" in calls["messages"][1][2]
    assert calls["messages"][1][3]["tools"][0]["tool_name"] == "sql_tool"
    assert calls["context_usage"] == ("session-1", {"usage_percent": 44})


def test_query_stream_cancellation_signals_running_agent(tmp_path) -> None:
    service, _calls, _runner_cls = _build_service_for_query_tests(tmp_path)
    service.dependencies.agent_runner_cls = _CancellableFakeRunner
    _CancellableFakeRunner.instances.clear()

    context = service.prepare_stream(
        QueryStreamExecutionRequest(
            session_id="session-1",
            payload=QueryRequest(query="stream analysis"),
            current_user=_auth_user(),
        )
    )

    runner = anyio_run(_cancel_stream_after_agent_starts(service.stream_events(context)))

    assert runner.cancel_seen.is_set()


class _FakeRouteQueryService:
    def __init__(self) -> None:
        self.executed: list[QueryExecutionRequest] = []
        self.prepared: list[QueryStreamExecutionRequest] = []
        self.streamed_contexts: list[str] = []

    async def execute(self, request: QueryExecutionRequest) -> QueryResponse:
        self.executed.append(request)
        return QueryResponse(
            session_id=request.session_id,
            text="route answer",
            artifacts=[],
            metrics=QueryMetrics(
                duration_ms=1,
                artifact_count=0,
                table_count=0,
                plot_count=0,
                value_count=0,
                model="test",
            ),
        )

    def prepare_stream(self, request: QueryStreamExecutionRequest) -> str:
        self.prepared.append(request)
        return "stream-context"

    async def stream_events(self, context: str):
        self.streamed_contexts.append(context)
        yield "final", {"text": "stream answer"}


async def _drain_streaming_response(response) -> str:
    chunks = [
        chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        async for chunk in response.body_iterator
    ]
    return "".join(chunks)


class _FakeAuthDB(AuthDB):
    def __init__(self, calls: dict[str, object], skill_settings: dict[str, bool]) -> None:
        self.calls = calls
        self.skill_settings = skill_settings
        self.is_owner = True

    def is_session_owner(self, _session_id: str, _user_id: int) -> bool:
        return self.is_owner

    def touch_session(self, session_id: str) -> None:
        self.calls.setdefault("touched", session_id)

    def update_session_after_reply(
        self,
        session_id: str,
        text: str,
        auto_title: str | None = None,
    ) -> None:
        self.calls.setdefault("session_reply", (session_id, text, auto_title))

    def list_user_tool_settings(self, _user_id: int) -> dict[str, bool]:
        return {}

    def list_user_skill_settings(self, _user_id: int) -> dict[str, bool]:
        return self.skill_settings

    def get_user_settings(self, _user_id: int) -> UserSettings:
        return _user_settings()


class _FakeSessionStore(SessionStore):
    def __init__(self, state: SessionState, calls: dict[str, object]) -> None:
        self.state = state
        self.calls = calls

    def load_session(self, _session_id: str) -> SessionState:
        return self.state

    def get_dataframe(self, _session_id: str):
        return None

    def load_data_catalog(self, _session_id: str):
        return None

    def get_structured_memory(self, _session_id: str):
        return None

    def set_structured_memory(self, session_id: str, memory) -> None:
        self.calls.setdefault("structured_memory", (session_id, memory))

    def add_artifacts(self, session_id: str, artifacts) -> None:
        self.calls.setdefault("artifacts", (session_id, artifacts))

    def set_selected_skill_ids(self, session_id: str, skill_ids: list[str]) -> None:
        self.calls["selected_skill_ids"].append((session_id, skill_ids))

    def add_chat_message(self, session_id: str, role: str, text: str, **kwargs) -> None:
        self.calls["messages"].append((session_id, role, text, kwargs))

    def set_context_usage(self, session_id: str, snapshot: dict[str, int]) -> None:
        self.calls.setdefault("context_usage", (session_id, snapshot))


class _FakeSkillRegistry(SkillRegistry):
    def __init__(self, skills_dir, skills: list[Skill]) -> None:
        self.skills_dir = skills_dir
        self._skills = skills

    def list_skills(self) -> list[Skill]:
        return self._skills

    def resolve_selection(self, skill_ids) -> list[Skill]:
        requested = set(skill_ids)
        return [skill for skill in self._skills if skill.skill_id in requested]


def _skill(tmp_path, skill_id: str, *, enabled_by_default: bool = True) -> Skill:
    return Skill(
        skill_id=skill_id,
        name=f"{skill_id.title()} Skill",
        description="",
        core_markdown="",
        details_markdown=None,
        source_path=str(tmp_path / skill_id / "SKILL.md"),
        kind="analytical",
        enabled_by_default=enabled_by_default,
    )


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> None:
        return None


def anyio_run(awaitable):
    import anyio

    return anyio.run(lambda: awaitable)


async def _collect_stream_events(stream):
    return [event async for event in stream]


async def _cancel_stream_after_agent_starts(stream):
    first = await stream.__anext__()
    assert first == ("start", {"session_id": "session-1"})

    pending = asyncio.create_task(stream.__anext__())
    while not _CancellableFakeRunner.instances:
        await asyncio.sleep(0.01)
    runner = _CancellableFakeRunner.instances[-1]
    while not runner.started.is_set():
        await asyncio.sleep(0.01)

    pending.cancel()
    try:
        await pending
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0.05)
    return runner


def _auth_user() -> AuthUser:
    return AuthUser(
        id=7,
        username="user",
        is_admin=False,
        created_at="2026-01-01T00:00:00+00:00",
    )


def _build_service_for_query_tests(
    tmp_path,
    *,
    state_selected_skill_ids: list[str] | None = None,
    user_skill_settings: dict[str, bool] | None = None,
    skills: list[Skill] | None = None,
):
    session_id = "session-1"
    state = SessionState(
        session_id=session_id,
        created_at="2026-01-01T00:00:00+00:00",
        last_access="2026-01-01T00:00:00+00:00",
        chat_history=[{"role": "user", "content": "previous"}],
        artifacts=[],
        source_type=None,
        selected_skill_ids=state_selected_skill_ids or ["allowed"],
    )
    calls: dict[str, object] = {"selected_skill_ids": [], "messages": []}
    auth_db = _FakeAuthDB(calls, user_skill_settings or {"allowed": True})
    store = _FakeSessionStore(state, calls)
    registry = _FakeSkillRegistry(tmp_path, skills or [_skill(tmp_path, "allowed")])

    class _FakeRunner:
        instances: ClassVar[list[_FakeRunner]] = []

        def __init__(self, _settings, **kwargs) -> None:
            self.kwargs = kwargs
            self.run_request: AgentRunRequest | None = None
            self.instances.append(self)

        def run(self, request: AgentRunRequest) -> AgentRunResult:
            self.run_request = request
            return AgentRunResult(
                response=AgentResponse(
                    final_text="service answer",
                    reasoning="service reasoning",
                    artifacts=[],
                    route="analysis",
                )
            )

    service = QueryExecutionService(
        dependencies=QueryExecutionDependencies(
            auth_db=auth_db,
            store=store,
            skill_registry=registry,
            db_runtime_service=None,
            search_integration_service=SimpleNamespace(source_descriptor=lambda: {"key": "search"}),
            forecast_integration_service=SimpleNamespace(source_descriptor=lambda: {"key": "forecast"}),
            anomaly_planfact_integration_service=SimpleNamespace(
                source_descriptor=lambda: {"key": "anomaly"}
            ),
            rag_service=SimpleNamespace(source_descriptor=lambda: {"key": "rag"}),
            user_memory_service=SimpleNamespace(
                load=lambda _user_id: None,
                schedule_consolidation=lambda *_args, **_kwargs: None,
            ),
            build_trace_context_fn=lambda **kwargs: kwargs,
            query_trace_context_fn=lambda **_kwargs: _NullContext(),
            settings=Settings(backend_query_timeout_sec=30),
            csv_runtime=CSVSessionRuntime(base_dir=tmp_path / "csv"),
            manifest_store=ManifestStore(tmp_path),
            storage_dir=tmp_path,
            llm_text_collector_cls=lambda: object(),
            tool_collector_cls=_ToolCollector,
            build_stream_callbacks_fn=_build_stream_callbacks,
            agent_runner_cls=_FakeRunner,
            effective_enabled_tool_keys_fn=lambda _catalog: {"python"},
            build_tool_catalog_fn=lambda **kwargs: list(kwargs["source_descriptors"]),
        )
    )
    return service, calls, _FakeRunner


class _CancellableFakeRunner:
    instances: ClassVar[list[_CancellableFakeRunner]] = []

    def __init__(self, _settings, **kwargs) -> None:
        self.kwargs = kwargs
        self.run_request: AgentRunRequest | None = None
        self.started = threading.Event()
        self.cancel_seen = threading.Event()
        self.instances.append(self)

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        self.run_request = request
        self.started.set()
        cancel_event = getattr(request, "cancel_event", None)
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                self.cancel_seen.set()
                break
            time.sleep(0.01)
        return AgentRunResult(
            response=AgentResponse(
                final_text="service answer",
                reasoning="service reasoning",
                artifacts=[],
                route="analysis",
            )
        )


def _user_settings() -> UserSettings:
    return UserSettings(
        theme="light",
        default_include_reasoning=False,
        default_answer_style="concise",
        analysis_mode="fast",
        analysis_depth="medium",
        llm_temperature_chat=0.1,
        llm_temperature_tool=0.1,
        llm_max_tokens_default=2400,
        llm_max_tokens_reasoning=3200,
        backend_query_timeout_sec=30,
        agent_max_steps=9,
        agent_step_timeout_sec=30,
        agent_inner_recursion_limit=10,
        llm_streaming=False,
        show_thinking=False,
    )


class _ToolCollector:
    def __init__(self, **_kwargs) -> None:
        self.tool_names: list[str] = []
        self.tool_calls = 0
        self.events: list[dict[str, object]] = []

    def to_persisted_activities(self, **_kwargs) -> list[dict[str, object]]:
        return []


class _TokenCollector:
    def collected_reasoning(self) -> str:
        return ""

    def all_reasoning_steps(self) -> list[str]:
        return []


class _PhaseCollector:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.graph_tracker = None
        self._graph_version = 0


class _ContextUsageCollector:
    snapshots: ClassVar[list[dict[str, int]]] = [{"usage_percent": 60}]


def _build_stream_callbacks(**_kwargs):
    token_collector = _TokenCollector()
    tool_collector = _ToolCollector()
    phase_collector = _PhaseCollector()
    context_usage_collector = _ContextUsageCollector()
    return (
        [token_collector, tool_collector, phase_collector, context_usage_collector],
        token_collector,
        tool_collector,
        phase_collector,
        None,
    )
