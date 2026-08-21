from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import anyio
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from backend.agent.anomaly_guard import check_numeric_consistency
from backend.agent.models import AgentOutcome, AgentResponse, ErrorCategory
from backend.agent.reasoning import MAX_REASONING_STEPS, ReasoningStep
from backend.agent.runtime_contracts import AgentRunRequest
from backend.agent.runtime_llm import build_runtime_llm
from backend.api.models import (
    QueryMetrics,
    QueryRequest,
    QueryResponse,
)
from backend.auth.auth_db import AuthDB, AuthUser
from backend.auth.user_memory import UserMemoryService
from backend.core.config import Settings
from backend.core.config import settings as default_settings
from backend.core.json_utils import make_json_safe
from backend.core.public_identity import display_model_name
from backend.data_access.catalog_refresh import attach_catalog_to_session_source
from backend.data_access.csv_runtime_state_service import (
    CSVRuntimeStateError,
    CSVRuntimeStateService,
)
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.data_access.db_runtime_service import DBRuntimeService
from backend.integrations.anomaly_planfact import AnomalyPlanfactIntegrationService
from backend.integrations.forecast import ForecastIntegrationService
from backend.integrations.rag import RAGService
from backend.mcp.models import MCPServerConfig, MCPToolDescriptor
from backend.mcp.service import MCPServerService, MCPToolProvider
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.session_source import is_duckdb_source_type
from backend.observability.phoenix import record_agent_outcome_on_active_span
from backend.sessions.session_store import SessionState, SessionStore
from backend.skills import SkillSelectionError
from backend.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

INTERRUPTED_STREAM_TEXT = "Остановлено пользователем."


class QueryExecutionDependencies(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    auth_db: AuthDB
    store: SessionStore
    skill_registry: SkillRegistry
    db_runtime_service: SkipValidation[DBRuntimeService | None]
    forecast_integration_service: SkipValidation[ForecastIntegrationService]
    anomaly_planfact_integration_service: SkipValidation[AnomalyPlanfactIntegrationService]
    rag_service: SkipValidation[RAGService]
    user_memory_service: SkipValidation[UserMemoryService]
    build_trace_context_fn: Callable[..., dict[str, Any]]
    query_trace_context_fn: Callable[..., Any]
    settings: Settings
    csv_runtime: CSVSessionRuntime
    manifest_store: ManifestStore
    storage_dir: Path
    blob_store: SkipValidation[Any | None] = None
    llm_text_collector_cls: Callable[..., Any]
    tool_collector_cls: Callable[..., Any]
    build_stream_callbacks_fn: Callable[..., tuple[list[Any], Any, Any, Any, Any]] | None = None
    agent_runner_cls: Callable[..., Any]
    effective_enabled_tool_keys_fn: Callable[[list[dict[str, Any]]], set[str]]
    build_tool_catalog_fn: Callable[..., list[dict[str, Any]]]
    mcp_service: SkipValidation[MCPServerService | None] = None
    semantic_context_builder: SkipValidation[Any | None] = None
    semantic_catalog_service: SkipValidation[Any | None] = None
    semantic_generation_service: SkipValidation[Any | None] = None


class MCPRuntimeContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: SkipValidation[MCPToolProvider | None] = None
    configs_by_id: dict[str, MCPServerConfig] = Field(default_factory=dict)
    tool_descriptors: list[MCPToolDescriptor] = Field(default_factory=list)
    tool_keys: set[str] = Field(default_factory=set)


class QueryExecutionRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    payload: QueryRequest
    current_user: AuthUser
    persist: bool
    callbacks: list[Any] = Field(default_factory=list)


class QueryStreamExecutionRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    payload: QueryRequest
    current_user: AuthUser


class QueryStreamExecutionContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: QueryStreamExecutionRequest
    state: SessionState
    selected_skill_ids: list[str]
    requested_tool_key: str | None = None
    df: pd.DataFrame | None = None
    session_source: dict[str, Any]
    session_db_connection_id: str | None = None
    has_active_source: bool
    selected_skill_names: str | None = None
    trace_context: dict[str, Any]
    runtime_settings: Settings
    query_runtime: Any
    show_thinking: bool
    started_at: float
    query_lease: SkipValidation[Any | None] = None


class PreparedAgentRuntime(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: SessionState
    selected_skill_ids: list[str]
    requested_tool_key: str | None = None
    df: pd.DataFrame | None = None
    session_source: dict[str, Any]
    session_db_connection_id: str | None = None
    has_active_source: bool
    selected_skill_names: str | None = None
    trace_context: dict[str, Any]
    runtime_settings: Settings
    query_runtime: Any
    started_at: float


class QueryExecutionService(BaseModel):
    """Run non-stream query requests behind the FastAPI route boundary."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dependencies: QueryExecutionDependencies

    async def execute(self, request: QueryExecutionRequest) -> QueryResponse:
        lease = self.dependencies.store.acquire_query_lease(request.session_id)
        try:
            return await self._execute_locked(request)
        finally:
            lease.release()

    async def _execute_locked(self, request: QueryExecutionRequest) -> QueryResponse:
        deps = self.dependencies
        request_kind = "query" if request.persist else "evaluate"
        prepared = self._prepare_agent_runtime(
            session_id=request.session_id,
            payload=request.payload,
            current_user=request.current_user,
            request_kind=request_kind,
        )
        from backend.artifacts.execution import ExecutionStore

        exec_store = ExecutionStore(session_id=request.session_id)
        text_collector = deps.llm_text_collector_cls()
        tool_collector = deps.tool_collector_cls(
            source_context=prepared.session_source,
            execution_store=exec_store,
            artifact_sink=(
                lambda artifacts: (
                    deps.store.add_artifacts(
                        request.session_id,
                        artifacts,
                        user_id=request.current_user.id,
                    )
                    if request.persist
                    else None
                )
            ),
        )
        active_callbacks = list(request.callbacks)
        active_callbacks.extend([text_collector, tool_collector])
        cancel_event = threading.Event()

        try:
            with deps.query_trace_context_fn(
                session_id=request.session_id,
                user_id=request.current_user.id,
                username=request.current_user.username,
                request_kind=request_kind,
                use_history=request.payload.use_history,
                include_reasoning=request.payload.include_reasoning,
                query=request.payload.query,
                db_connection_id=prepared.session_db_connection_id,
                csv_session_id=prepared.session_source.get("csv_session_id"),
                csv_duckdb_loaded=bool(prepared.session_source.get("csv_loaded")),
                selected_skill_names=prepared.selected_skill_names,
            ):
                with anyio.fail_after(prepared.runtime_settings.backend_query_timeout_sec):
                    run_result = await anyio.to_thread.run_sync(
                        prepared.query_runtime.run,
                        AgentRunRequest(
                            df=prepared.df,
                            prompt=request.payload.query,
                            history=list(prepared.state.chat_history or []),
                            use_history=request.payload.use_history,
                            include_reasoning=request.payload.include_reasoning,
                            callbacks=active_callbacks,
                            trace_context=prepared.trace_context,
                            session_source=prepared.session_source,
                            selected_skill_ids=prepared.selected_skill_ids,
                            requested_tool_key=prepared.requested_tool_key,
                            cancel_event=cancel_event,
                        ),
                        abandon_on_cancel=True,
                    )
                    response = run_result.response
                    record_agent_outcome_on_active_span(response)
        except SkillSelectionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except TimeoutError:
            cancel_event.set()
            return self._fallback(
                request=request,
                reason="timeout",
                started_at=prepared.started_at,
                model_name=prepared.runtime_settings.llm_model,
            )
        except Exception:
            logger.exception(
                "query execution failed; returning fallback response session_id=%s user_id=%s",
                request.session_id,
                request.current_user.id,
            )
            return self._fallback(
                request=request,
                reason="runtime_error",
                started_at=prepared.started_at,
                model_name=prepared.runtime_settings.llm_model,
            )

        self._persist_runtime_effects(
            response,
            user_id=request.current_user.id,
            session_id=request.session_id,
            runtime_settings=prepared.runtime_settings,
        )
        artifacts = self._serialize_execution_artifacts(response.artifacts)
        anomaly_check = self._anomaly_check(
            request.current_user.id,
            response.final_text,
            artifacts,
            request.payload.query,
        )
        duration_ms = int((time.perf_counter() - prepared.started_at) * 1000)
        effective_reasoning = self._build_reasoning_trace(
            response_text=response.final_text,
            response_reasoning=response.reasoning,
            route=response.route,
            tool_collector=tool_collector,
            use_history=request.payload.use_history,
            duration_ms=duration_ms,
            has_dataset=prepared.has_active_source,
        )
        if request.persist:
            deps.store.set_selected_skill_ids(request.session_id, prepared.selected_skill_ids)
            deps.store.add_chat_message(request.session_id, "user", request.payload.query)
            deps.store.add_chat_message(
                request.session_id,
                "ai",
                response.final_text,
                artifacts=artifacts,
                reasoning=effective_reasoning,
                anomaly_check=anomaly_check,
            )
            self._persist_uncollected_artifacts(
                store=deps.store,
                session_id=request.session_id,
                user_id=request.current_user.id,
                artifacts=response.artifacts,
                tool_collector=tool_collector,
            )
            self._persist_context_usage_snapshot(
                deps.store,
                request.session_id,
                active_callbacks,
            )
            deps.auth_db.update_session_after_reply(
                request.session_id,
                response.final_text,
                auto_title=None,
            )

        return self._build_response(
            request.session_id,
            response.final_text,
            effective_reasoning,
            artifacts,
            duration_ms,
            prepared.runtime_settings.llm_model,
            llm_duration_ms=int(getattr(text_collector, "llm_duration_ms", 0)),
            llm_calls=int(getattr(text_collector, "llm_calls", 0)),
            is_admin=request.current_user.is_admin,
            include_reasoning=request.payload.include_reasoning,
            force_reasoning=request.persist,
            agent_response=response,
            anomaly_check=anomaly_check,
        )

    async def stream_events(
        self,
        context: QueryStreamExecutionContext,
    ):
        try:
            async for event in self._stream_events_locked(context):
                yield event
        finally:
            if context.query_lease is not None:
                context.query_lease.release()

    async def _stream_events_locked(
        self,
        context: QueryStreamExecutionContext,
    ):
        deps = self.dependencies
        request = context.request
        state = context.state
        selected_skill_ids = context.selected_skill_ids
        requested_tool_key = context.requested_tool_key
        df = context.df
        session_source = context.session_source
        session_db_connection_id = context.session_db_connection_id
        has_active_source = context.has_active_source
        selected_skill_names = context.selected_skill_names
        trace_context = context.trace_context
        runtime_settings = context.runtime_settings
        query_runtime = context.query_runtime
        started_at = context.started_at

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        agent_finished = asyncio.Event()
        client_cancelled = asyncio.Event()
        cancel_event = threading.Event()
        initial_history_len = len(state.chat_history or [])
        partial_persisted = False

        from backend.artifacts.execution import ExecutionStore

        exec_store = ExecutionStore(session_id=request.session_id)
        if deps.build_stream_callbacks_fn is None:
            raise RuntimeError("Query stream callback builder is not configured")
        callbacks, token_collector, tool_collector, phase_collector, _graph_tracker = (
            deps.build_stream_callbacks_fn(
                queue=queue,
                loop=loop,
                session_source=session_source,
                exec_store=exec_store,
                include_reasoning=request.payload.include_reasoning,
                show_thinking=context.show_thinking,
            )
        )
        tool_collector.artifact_sink = lambda artifacts: deps.store.add_artifacts(
            request.session_id,
            artifacts,
            user_id=request.current_user.id,
        )

        def persist_interrupted_once() -> None:
            nonlocal partial_persisted
            if partial_persisted:
                return
            try:
                partial_persisted = self._persist_interrupted_stream_turn(
                    session_id=request.session_id,
                    user_id=request.current_user.id,
                    query_text=request.payload.query,
                    token_collector=token_collector,
                    tool_collector=tool_collector,
                    callbacks=callbacks,
                    selected_skill_ids=selected_skill_ids,
                    initial_history_len=initial_history_len,
                )
            except Exception:
                partial_persisted = True
                logger.exception(
                    "query_stream interrupted-turn persistence failed session_id=%s user_id=%s",
                    request.session_id,
                    request.current_user.id,
                )

        async def run_agent() -> None:
            try:
                with deps.query_trace_context_fn(
                    session_id=request.session_id,
                    user_id=request.current_user.id,
                    username=request.current_user.username,
                    request_kind="stream",
                    use_history=request.payload.use_history,
                    include_reasoning=request.payload.include_reasoning,
                    query=request.payload.query,
                    db_connection_id=session_db_connection_id,
                    csv_session_id=session_source.get("csv_session_id"),
                    csv_duckdb_loaded=bool(session_source.get("csv_loaded")),
                    selected_skill_names=selected_skill_names,
                ):
                    with anyio.fail_after(runtime_settings.backend_query_timeout_sec):
                        run_result = await anyio.to_thread.run_sync(
                            query_runtime.run,
                            AgentRunRequest(
                                df=df,
                                prompt=request.payload.query,
                                history=list(state.chat_history or []),
                                use_history=request.payload.use_history,
                                include_reasoning=request.payload.include_reasoning,
                                callbacks=callbacks,
                                trace_context=trace_context,
                                session_source=session_source,
                                selected_skill_ids=selected_skill_ids,
                                requested_tool_key=requested_tool_key,
                                cancel_event=cancel_event,
                            ),
                            abandon_on_cancel=True,
                        )
                        response = run_result.response
                        record_agent_outcome_on_active_span(response)
                if client_cancelled.is_set():
                    return
                self._persist_runtime_effects(
                    response,
                    user_id=request.current_user.id,
                    session_id=request.session_id,
                    runtime_settings=runtime_settings,
                )

                duration_ms = int((time.perf_counter() - started_at) * 1000)
                streamed_reasoning = token_collector.collected_reasoning()
                merged_reasoning = self._merge_reasoning_text(
                    response.reasoning,
                    streamed_reasoning,
                )
                raw_steps = token_collector.all_reasoning_steps() or response.reasoning_steps
                ordered_start_names = [
                    event["tool_name"] for event in tool_collector.events if event.get("phase") == "start"
                ]
                reasoning_steps = self._build_reasoning_steps(
                    raw_steps,
                    tool_collector.tool_calls,
                    ordered_start_names,
                )
                artifacts = self._serialize_execution_artifacts(response.artifacts)
                anomaly_check = self._anomaly_check(
                    request.current_user.id,
                    response.final_text,
                    artifacts,
                    request.payload.query,
                )
                try:
                    effective_reasoning = self._build_reasoning_trace(
                        response_text=response.final_text,
                        response_reasoning=merged_reasoning,
                        route=response.route,
                        tool_collector=tool_collector,
                        use_history=request.payload.use_history,
                        duration_ms=duration_ms,
                        has_dataset=has_active_source,
                    )
                    deps.store.set_selected_skill_ids(request.session_id, selected_skill_ids)
                    deps.store.add_chat_message(
                        request.session_id,
                        "user",
                        request.payload.query,
                    )
                    deps.store.add_chat_message(
                        request.session_id,
                        "ai",
                        response.final_text,
                        artifacts=artifacts,
                        reasoning=effective_reasoning,
                        reasoning_steps=[step.to_dict() for step in reasoning_steps] or None,
                        tools=tool_collector.to_persisted_activities() or None,
                        anomaly_check=anomaly_check,
                    )
                    self._persist_uncollected_artifacts(
                        store=deps.store,
                        session_id=request.session_id,
                        user_id=request.current_user.id,
                        artifacts=response.artifacts,
                        tool_collector=tool_collector,
                    )
                    self._persist_context_usage_snapshot(
                        deps.store,
                        request.session_id,
                        callbacks,
                    )
                    deps.auth_db.update_session_after_reply(
                        request.session_id,
                        response.final_text,
                        auto_title=None,
                    )
                except Exception:
                    logger.exception(
                        "query_stream post-processing failed; returning agent response without persistence "
                        "session_id=%s user_id=%s",
                        request.session_id,
                        request.current_user.id,
                    )
                    effective_reasoning = merged_reasoning or response.reasoning
                    persistence_failed = True
                else:
                    persistence_failed = False

                final_payload = self._build_response(
                    request.session_id,
                    response.final_text,
                    effective_reasoning,
                    artifacts,
                    duration_ms,
                    runtime_settings.llm_model,
                    llm_duration_ms=int(
                        getattr(
                            next(
                                (callback for callback in callbacks if hasattr(callback, "llm_duration_ms")),
                                None,
                            ),
                            "llm_duration_ms",
                            0,
                        )
                    ),
                    llm_calls=int(
                        getattr(
                            next(
                                (callback for callback in callbacks if hasattr(callback, "llm_calls")),
                                None,
                            ),
                            "llm_calls",
                            0,
                        )
                    ),
                    is_admin=request.current_user.is_admin,
                    include_reasoning=request.payload.include_reasoning,
                    force_reasoning=True,
                    agent_response=response,
                    anomaly_check=anomaly_check,
                ).model_dump()
                if persistence_failed:
                    final_payload["persistence_failed"] = True
                graph_tracker = phase_collector.graph_tracker
                if graph_tracker is not None and graph_tracker:
                    final_payload["execution_graph"] = graph_tracker.snapshot()
                await queue.put(("final", final_payload))
            except SkillSelectionError as exc:
                if client_cancelled.is_set():
                    return
                await queue.put(("error", {"detail": str(exc)}))
                return
            except TimeoutError:
                cancel_event.set()
                if client_cancelled.is_set():
                    return
                duration_ms = int((time.perf_counter() - started_at) * 1000)
                fallback_payload = self._build_fallback_response(
                    session_id=request.session_id,
                    query=request.payload.query,
                    reason="timeout",
                    duration_ms=duration_ms,
                    model_name=runtime_settings.llm_model,
                    is_admin=request.current_user.is_admin,
                    include_reasoning=request.payload.include_reasoning,
                    force_reasoning=True,
                )
                record_agent_outcome_on_active_span(
                    AgentResponse(
                        final_text=fallback_payload.text,
                        reasoning=fallback_payload.reasoning,
                        artifacts=[],
                        route="analysis",
                        outcome=AgentOutcome.unavailable(ErrorCategory.TRANSPORT),
                    )
                )
                self._persist_fallback_response(
                    request.session_id,
                    request.current_user.id,
                    request.payload.query,
                    fallback_payload.text,
                    reasoning=fallback_payload.reasoning,
                )
                await queue.put(("final", fallback_payload.model_dump()))
            except Exception:
                if client_cancelled.is_set():
                    return
                logger.exception(
                    "query_stream failed; returning fallback response session_id=%s user_id=%s",
                    request.session_id,
                    request.current_user.id,
                )
                duration_ms = int((time.perf_counter() - started_at) * 1000)
                fallback_payload = self._build_fallback_response(
                    session_id=request.session_id,
                    query=request.payload.query,
                    reason="runtime_error",
                    duration_ms=duration_ms,
                    model_name=runtime_settings.llm_model,
                    is_admin=request.current_user.is_admin,
                    include_reasoning=request.payload.include_reasoning,
                    force_reasoning=True,
                )
                record_agent_outcome_on_active_span(
                    AgentResponse(
                        final_text=fallback_payload.text,
                        reasoning=fallback_payload.reasoning,
                        artifacts=[],
                        route="analysis",
                        outcome=AgentOutcome.failed(ErrorCategory.INTERNAL),
                    )
                )
                self._persist_fallback_response(
                    request.session_id,
                    request.current_user.id,
                    request.payload.query,
                    fallback_payload.text,
                    reasoning=fallback_payload.reasoning,
                )
                await queue.put(("final", fallback_payload.model_dump()))
            finally:
                agent_finished.set()
                await queue.put(("done", None))

        async def emit_live_reasoning() -> None:
            emitted_phase_events = 0
            emitted_graph_version = 0
            while True:
                emitted_any = False

                while emitted_phase_events < len(phase_collector.events):
                    current = phase_collector.events[emitted_phase_events]
                    emitted_phase_events += 1
                    await queue.put(("phase", current))
                    emitted_any = True

                graph_tracker = phase_collector.graph_tracker
                if graph_tracker is not None:
                    graph_version = phase_collector._graph_version  # noqa: SLF001
                    if graph_version > emitted_graph_version:
                        emitted_graph_version = graph_version
                        await queue.put(("execution_graph", graph_tracker.snapshot()))
                        emitted_any = True

                if agent_finished.is_set() and emitted_phase_events >= len(phase_collector.events):
                    break

                if not emitted_any:
                    await asyncio.sleep(0.02)

        yield "start", {"session_id": request.session_id}
        agent_task = asyncio.create_task(run_agent())
        reasoning_task = asyncio.create_task(emit_live_reasoning())
        deferred_final: list[tuple[str, Any]] = []
        try:
            while True:
                event, data = await queue.get()
                if event == "done":
                    await reasoning_task
                    while not queue.empty():
                        extra_event, extra_data = await queue.get()
                        if extra_event == "done":
                            continue
                        if extra_event == "final":
                            deferred_final.append((extra_event, extra_data))
                            continue
                        yield extra_event, extra_data
                    for final_event, final_data in deferred_final:
                        yield final_event, final_data
                    break
                if event == "final":
                    deferred_final.append((event, data))
                    continue
                yield event, data
            await agent_task
            await reasoning_task
        except (asyncio.CancelledError, GeneratorExit):
            client_cancelled.set()
            cancel_event.set()
            persist_interrupted_once()

            def consume_task_result(task: asyncio.Task) -> None:
                try:
                    task.result()
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("background stream task failed after client cancellation")

            agent_task.add_done_callback(consume_task_result)
            if not reasoning_task.done():
                reasoning_task.cancel()
            reasoning_task.add_done_callback(consume_task_result)
            raise

    def prepare_stream(
        self,
        request: QueryStreamExecutionRequest,
    ) -> QueryStreamExecutionContext:
        deps = self.dependencies
        lease = deps.store.acquire_query_lease(request.session_id)
        try:
            prepared = self._prepare_agent_runtime(
                session_id=request.session_id,
                payload=request.payload,
                current_user=request.current_user,
                request_kind="stream",
            )
            user_stream_settings = deps.auth_db.get_user_settings(request.current_user.id)
            if deps.build_stream_callbacks_fn is None:
                raise RuntimeError("Query stream callback builder is not configured")

            return QueryStreamExecutionContext(
                request=request,
                state=prepared.state,
                selected_skill_ids=prepared.selected_skill_ids,
                requested_tool_key=prepared.requested_tool_key,
                df=prepared.df,
                session_source=prepared.session_source,
                session_db_connection_id=prepared.session_db_connection_id,
                has_active_source=prepared.has_active_source,
                selected_skill_names=prepared.selected_skill_names,
                trace_context=prepared.trace_context,
                runtime_settings=prepared.runtime_settings,
                query_runtime=prepared.query_runtime,
                show_thinking=bool(user_stream_settings.show_thinking),
                started_at=prepared.started_at,
                query_lease=lease,
            )
        except Exception:
            lease.release()
            raise

    def _prepare_agent_runtime(
        self,
        *,
        session_id: str,
        payload: QueryRequest,
        current_user: AuthUser,
        request_kind: str,
    ) -> PreparedAgentRuntime:
        deps = self.dependencies
        state = self._load_owned_session(session_id, current_user)
        if is_duckdb_source_type(self._session_source_type(state)):
            state = self._ensure_csv_runtime_state(session_id, state)

        selected_skill_ids = self._validated_selected_skill_ids(
            current_user.id,
            state,
            payload,
        )
        df = self._active_session_dataframe(state, session_id)
        session_source = self._session_runtime_source_payload(state)
        session_source = self._attach_semantic_context(
            session_id=session_id,
            user_id=current_user.id,
            query=payload.query,
            session_source=session_source,
        )
        session_db_connection_id = self._session_db_connection_id(state)
        has_active_source = (
            df is not None
            or session_db_connection_id is not None
            or bool(session_source.get("csv_loaded"))
            or self._session_source_type(state) == "rag"
        )
        requested_tool_key = str(payload.requested_tool_key or "").strip() or None
        started_at = time.perf_counter()
        deps.auth_db.touch_session(session_id)
        selected_skill_names = self._resolve_skill_names(selected_skill_ids)
        trace_context = deps.build_trace_context_fn(
            session_id=session_id,
            user_id=current_user.id,
            username=current_user.username,
            request_kind=request_kind,
            use_history=payload.use_history,
            include_reasoning=payload.include_reasoning,
            db_connection_id=session_db_connection_id,
            csv_session_id=session_source.get("csv_session_id"),
            csv_duckdb_loaded=bool(session_source.get("csv_loaded")),
            selected_skill_names=selected_skill_names,
        )
        runtime_settings = self._effective_runtime_settings(
            current_user.id,
            analysis_depth_override=payload.analysis_depth,
        )
        mcp_runtime = self._mcp_runtime_context_for_user(current_user.id)
        allowed_tool_keys = self._enabled_tool_keys_for_user(current_user.id)
        allowed_tool_keys.update(mcp_runtime.tool_keys)
        query_runtime = deps.agent_runner_cls(
            runtime_settings,
            db_runtime_service=deps.db_runtime_service,
            forecast_service=deps.forecast_integration_service,
            anomaly_planfact_service=deps.anomaly_planfact_integration_service,
            rag_service=deps.rag_service,
            allowed_tool_keys=allowed_tool_keys,
            user_memory=deps.user_memory_service.load(current_user.id),
            session_memory=deps.store.get_structured_memory(session_id),
            skill_registry=deps.skill_registry,
            enabled_analytical_skill_ids=self._enabled_analytical_skill_ids_for_user(
                current_user.id,
            ),
            mcp_tool_provider=mcp_runtime.provider,
            mcp_server_configs=mcp_runtime.configs_by_id,
            mcp_tool_descriptors=mcp_runtime.tool_descriptors,
            semantic_catalog_service=deps.semantic_catalog_service,
            semantic_generation_service=deps.semantic_generation_service,
            session_store=deps.store,
            blob_store=deps.blob_store,
        )
        return PreparedAgentRuntime(
            state=state,
            selected_skill_ids=selected_skill_ids,
            requested_tool_key=requested_tool_key,
            df=df,
            session_source=session_source,
            session_db_connection_id=session_db_connection_id,
            has_active_source=has_active_source,
            selected_skill_names=selected_skill_names,
            trace_context=trace_context,
            runtime_settings=runtime_settings,
            query_runtime=query_runtime,
            started_at=started_at,
        )

    def _fallback(
        self,
        *,
        request: QueryExecutionRequest,
        reason: str,
        started_at: float,
        model_name: str,
    ) -> QueryResponse:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        fallback = self._build_fallback_response(
            session_id=request.session_id,
            query=request.payload.query,
            reason=reason,
            duration_ms=duration_ms,
            model_name=model_name,
            is_admin=request.current_user.is_admin,
            include_reasoning=request.payload.include_reasoning,
            force_reasoning=request.persist,
        )
        fallback_agent_response = AgentResponse(
            final_text=fallback.text,
            reasoning=fallback.reasoning,
            artifacts=[],
            route="analysis",
            outcome=(
                AgentOutcome.unavailable(ErrorCategory.TRANSPORT)
                if reason == "timeout"
                else AgentOutcome.failed(ErrorCategory.INTERNAL)
            ),
        )
        record_agent_outcome_on_active_span(fallback_agent_response)
        if request.persist:
            self._persist_fallback_response(
                request.session_id,
                request.current_user.id,
                request.payload.query,
                fallback.text,
                reasoning=fallback.reasoning,
            )
        return fallback

    def _load_owned_session(
        self,
        session_id: str,
        current_user: AuthUser,
    ) -> SessionState:
        deps = self.dependencies
        if not deps.auth_db.is_session_owner(session_id, current_user.id):
            raise HTTPException(status_code=404, detail="Session not found")
        state = deps.store.load_session(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return state

    def _session_runtime_source_payload(self, state: SessionState) -> dict[str, Any]:
        payload = self._session_source_payload(state)
        source_type = self._session_source_type(state)
        if is_duckdb_source_type(source_type):
            payload["csv_loaded"] = bool(state.csv_loaded)
            payload["csv_session_id"] = state.csv_session_id
            payload["csv_table_names"] = list(state.csv_table_names or [])
            payload["csv_expires_at"] = state.csv_expires_at
        else:
            payload["csv_loaded"] = False
            payload["csv_session_id"] = None
            payload["csv_table_names"] = []
            payload["csv_expires_at"] = None
        return attach_catalog_to_session_source(
            self.dependencies.store,
            state.session_id,
            payload,
        )

    def _attach_semantic_context(
        self,
        *,
        session_id: str,
        user_id: int,
        query: str,
        session_source: dict[str, Any],
    ) -> dict[str, Any]:
        builder = self.dependencies.semantic_context_builder
        if builder is None or not getattr(self.dependencies.settings, "semantic_layer_enabled", False):
            return session_source
        payload = dict(session_source)
        try:
            context = builder.build(
                session_id=session_id,
                user_id=user_id,
                query=query,
            )
        except Exception as exc:
            logger.warning("Failed to build semantic context: %s", exc)
            payload["semantic_context_status"] = "failed"
            payload["semantic_context_error"] = str(exc)
            return payload

        payload["semantic_context_status"] = context.status
        payload["semantic_context_prompt"] = context.prompt
        payload["semantic_context_hints"] = context.hints
        return payload

    def _ensure_csv_runtime_state(
        self,
        session_id: str,
        state: SessionState,
    ) -> SessionState:
        deps = self.dependencies
        if not is_duckdb_source_type(self._session_source_type(state)):
            return state
        try:
            refreshed = CSVRuntimeStateService(
                store=deps.store,
                csv_runtime=deps.csv_runtime,
                manifest_store=deps.manifest_store,
                storage_dir=deps.storage_dir,
                blob_store=deps.blob_store,
            ).ensure_csv_runtime(
                session_id=session_id,
                ttl_seconds=getattr(
                    deps.settings,
                    "csv_session_ttl_sec",
                    default_settings.csv_session_ttl_sec,
                ),
            )
        except CSVRuntimeStateError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize CSV runtime: {exc}",
            ) from exc
        return refreshed

    def _active_session_dataframe(
        self,
        state: SessionState,
        session_id: str,
    ) -> pd.DataFrame | None:
        if not is_duckdb_source_type(self._session_source_type(state)):
            return None
        return self.dependencies.store.get_dataframe(session_id)

    def _session_db_connection_id(self, state: SessionState) -> str | None:
        if self._session_source_type(state) not in {"db_connection", "openproject"}:
            return None
        ref_id = str(state.source_ref_id or "").strip()
        return ref_id or None

    @staticmethod
    def _session_source_payload(state: SessionState) -> dict[str, Any]:
        return {
            "source_type": state.source_type,
            "source_ref_id": state.source_ref_id,
            "source_label": state.source_label,
            "source_mode": state.source_mode,
        }

    @staticmethod
    def _session_source_type(state: SessionState) -> str:
        return str(state.source_type or "").strip().lower()

    def _integration_source_descriptors(self) -> list[dict[str, Any]]:
        deps = self.dependencies
        return [
            deps.rag_service.source_descriptor(),
            deps.forecast_integration_service.source_descriptor(),
            deps.anomaly_planfact_integration_service.source_descriptor(),
        ]

    def _tool_catalog_payload(self, user_id: int) -> list[dict[str, Any]]:
        deps = self.dependencies
        return deps.build_tool_catalog_fn(
            source_descriptors=self._integration_source_descriptors(),
            user_settings=deps.auth_db.list_user_tool_settings(user_id),
        )

    def _enabled_tool_keys_for_user(self, user_id: int) -> set[str]:
        return self.dependencies.effective_enabled_tool_keys_fn(
            self._tool_catalog_payload(user_id),
        )

    def _mcp_runtime_context_for_user(self, user_id: int) -> MCPRuntimeContext:
        mcp_service = self.dependencies.mcp_service
        if mcp_service is None:
            return MCPRuntimeContext()
        user_settings = self.dependencies.auth_db.list_user_mcp_server_settings(user_id)
        descriptors = mcp_service.enabled_tool_descriptors(user_settings=user_settings)
        configs_by_id = mcp_service.configs_by_id()
        return MCPRuntimeContext(
            provider=mcp_service.provider,
            configs_by_id=configs_by_id,
            tool_descriptors=descriptors,
            tool_keys={descriptor.tool_key for descriptor in descriptors},
        )

    @staticmethod
    def _effective_selected_skill_ids(
        state: SessionState,
        payload: QueryRequest,
    ) -> list[str]:
        if payload.selected_skill_ids is None:
            selected_skill_ids = list(state.selected_skill_ids or [])
        else:
            selected_skill_ids = [
                str(skill_id).strip() for skill_id in payload.selected_skill_ids if str(skill_id).strip()
            ]
        return list(dict.fromkeys(selected_skill_ids))

    def _enabled_analytical_skill_ids_for_user(self, user_id: int) -> set[str]:
        user_skill_settings = self.dependencies.auth_db.list_user_skill_settings(user_id)
        return {
            skill.skill_id
            for skill in self.dependencies.skill_registry.list_skills()
            if skill.kind == "analytical"
            and user_skill_settings.get(skill.skill_id, skill.enabled_by_default)
        }

    def _filter_selected_skill_ids_for_user(
        self,
        user_id: int,
        selected_skill_ids: list[str],
    ) -> list[str]:
        allowed_skill_ids = self._enabled_analytical_skill_ids_for_user(user_id)
        return [skill_id for skill_id in selected_skill_ids if skill_id in allowed_skill_ids]

    def _validated_selected_skill_ids(
        self,
        user_id: int,
        state: SessionState,
        payload: QueryRequest,
    ) -> list[str]:
        selected = self._effective_selected_skill_ids(state, payload)
        allowed = self._enabled_analytical_skill_ids_for_user(user_id)
        unavailable = [skill_id for skill_id in selected if skill_id not in allowed]
        if payload.selected_skill_ids is not None and unavailable:
            raise HTTPException(
                status_code=422,
                detail="Selected skill is unavailable: " + ", ".join(unavailable),
            )
        return [skill_id for skill_id in selected if skill_id in allowed]

    def _resolve_skill_names(self, skill_ids: list[str]) -> str | None:
        if not skill_ids:
            return None
        try:
            skills = self.dependencies.skill_registry.resolve_selection(skill_ids)
            return ", ".join(skill.name for skill in skills)
        except Exception:
            return ", ".join(skill_ids)

    def _effective_runtime_settings(
        self,
        user_id: int,
        *,
        analysis_depth_override: str | None = None,
    ) -> Settings:
        user_runtime = self.dependencies.auth_db.get_user_settings(user_id)
        depth = analysis_depth_override or user_runtime.analysis_depth
        runtime = replace(
            self.dependencies.settings,
            llm_temperature_chat=user_runtime.llm_temperature_chat,
            llm_temperature_tool=user_runtime.llm_temperature_tool,
            llm_max_tokens_default=user_runtime.llm_max_tokens_default,
            llm_max_tokens_reasoning=user_runtime.llm_max_tokens_reasoning,
            backend_query_timeout_sec=user_runtime.backend_query_timeout_sec,
            agent_max_steps=user_runtime.agent_max_steps,
            agent_step_timeout_sec=user_runtime.agent_step_timeout_sec,
            agent_inner_recursion_limit=user_runtime.agent_inner_recursion_limit,
            agent_analysis_depth=depth,
            llm_streaming=user_runtime.llm_streaming,
            anomaly_check_enabled=user_runtime.anomaly_check_enabled,
            always_use_analysis_plan=user_runtime.always_use_analysis_plan,
        )
        return runtime

    def _persist_runtime_effects(
        self,
        response: AgentResponse,
        *,
        user_id: int,
        session_id: str,
        runtime_settings: Settings,
    ) -> None:
        deps = self.dependencies
        effects = response.runtime_effects
        if effects.user_memory_notes:
            try:
                deps.user_memory_service.schedule_consolidation(
                    user_id,
                    list(effects.user_memory_notes),
                    self._build_memory_consolidation_invoker(runtime_settings),
                )
            except Exception:
                logger.exception("Failed to schedule user memory consolidation")

        if effects.session_memory is not None:
            try:
                deps.store.set_structured_memory(session_id, effects.session_memory)
            except Exception:
                logger.exception("Failed to persist structured session memory")

    @staticmethod
    def _build_memory_consolidation_invoker(runtime_settings: Settings) -> Callable[..., Any]:
        llm = build_runtime_llm(
            runtime_settings,
            role="chat",
            include_reasoning=False,
            max_tokens_override=800,
        )
        return llm.invoke

    def _persist_fallback_response(
        self,
        session_id: str,
        user_id: int,
        query_text: str,
        error_text: str,
        *,
        reasoning: str | None = None,
        auto_title: str | None = None,
    ) -> None:
        _ = user_id
        deps = self.dependencies
        deps.store.add_chat_message(session_id, "user", query_text)
        deps.store.add_chat_message(session_id, "ai", error_text, reasoning=reasoning)
        deps.auth_db.update_session_after_reply(
            session_id,
            error_text,
            auto_title=auto_title,
        )

    def _persist_interrupted_stream_turn(
        self,
        *,
        session_id: str,
        user_id: int,
        query_text: str,
        token_collector: Any,
        tool_collector: Any,
        callbacks: list[Any],
        selected_skill_ids: list[str],
        initial_history_len: int | None = None,
    ) -> bool:
        deps = self.dependencies
        if initial_history_len is not None:
            state = deps.store.load_session(session_id)
            if state is not None and len(state.chat_history or []) > initial_history_len:
                return False

        visible_text = ""
        collected_visible = getattr(token_collector, "collected_visible", None)
        if callable(collected_visible):
            visible_text = str(collected_visible() or "").strip()

        collected_reasoning = getattr(token_collector, "collected_reasoning", None)
        reasoning = (
            self._merge_reasoning_text(str(collected_reasoning() or ""))
            if callable(collected_reasoning)
            else None
        )

        to_persisted = getattr(tool_collector, "to_persisted_activities", None)
        tools = (
            to_persisted(
                unfinished_status="error",
                unfinished_error=INTERRUPTED_STREAM_TEXT,
            )
            if callable(to_persisted)
            else []
        )
        artifacts = self._serialize_execution_artifacts(list(getattr(tool_collector, "artifacts", []) or []))
        if not (visible_text or reasoning or tools or artifacts):
            return False

        all_reasoning_steps = getattr(token_collector, "all_reasoning_steps", None)
        raw_steps = all_reasoning_steps() if callable(all_reasoning_steps) else []
        ordered_start_names = [
            event["tool_name"]
            for event in getattr(tool_collector, "events", []) or []
            if event.get("phase") == "start"
        ]
        reasoning_steps = self._build_reasoning_steps(
            raw_steps,
            int(getattr(tool_collector, "tool_calls", 0) or 0),
            ordered_start_names,
        )

        text = visible_text or INTERRUPTED_STREAM_TEXT
        deps.store.set_selected_skill_ids(session_id, selected_skill_ids)
        deps.store.add_chat_message(session_id, "user", query_text)
        deps.store.add_chat_message(
            session_id,
            "ai",
            text,
            artifacts=artifacts or None,
            reasoning=reasoning,
            reasoning_steps=[step.to_dict() for step in reasoning_steps] or None,
            tools=tools or None,
        )
        if getattr(tool_collector, "artifacts", None):
            self._persist_uncollected_artifacts(
                store=deps.store,
                session_id=session_id,
                user_id=user_id,
                artifacts=tool_collector.artifacts,
                tool_collector=tool_collector,
            )
        self._persist_context_usage_snapshot(deps.store, session_id, callbacks)
        deps.auth_db.update_session_after_reply(session_id, text, auto_title=None)
        return True

    @staticmethod
    def _persist_uncollected_artifacts(
        *,
        store: Any,
        session_id: str,
        user_id: int,
        artifacts: list[Any],
        tool_collector: Any,
    ) -> None:
        persisted_ids = set(getattr(tool_collector, "persisted_artifact_ids", set()) or set())
        remaining = [
            artifact for artifact in artifacts if str(getattr(artifact, "id", "")) not in persisted_ids
        ]
        if remaining:
            store.add_artifacts(session_id, remaining, user_id=user_id)

    @staticmethod
    def _persist_context_usage_snapshot(
        store: Any,
        session_id: str,
        callbacks: list[Any],
    ) -> None:
        setter = getattr(store, "set_context_usage", None)
        if not callable(setter):
            return
        for callback in callbacks:
            snapshots = getattr(callback, "snapshots", None)
            if isinstance(snapshots, list) and snapshots and isinstance(snapshots[-1], dict):
                setter(session_id, make_json_safe(snapshots[-1]))
                return

    @staticmethod
    def _serialize_execution_artifacts(artifacts: list[Any]) -> list[dict[str, Any]]:
        from backend.artifacts.bridge import execution_to_api_payload

        serialized: list[dict[str, Any]] = []
        for artifact in artifacts:
            try:
                serialized.append(execution_to_api_payload(artifact))
            except Exception:
                logger.exception(
                    "artifact serialization failed name=%s type=%s",
                    getattr(artifact, "name", ""),
                    getattr(artifact, "artifact_type", ""),
                )
        return serialized

    def _anomaly_check(
        self,
        user_id: int,
        text: str,
        artifacts: list[dict[str, Any]],
        source_text: str,
    ) -> dict[str, Any] | None:
        if not self.dependencies.auth_db.get_user_settings(user_id).anomaly_check_enabled:
            return None
        return check_numeric_consistency(text, artifacts, source_text)

    @staticmethod
    def _build_response(
        session_id: str,
        text: str,
        reasoning: str | None,
        artifacts: list[dict[str, Any]],
        duration_ms: int,
        model_name: str,
        *,
        llm_duration_ms: int = 0,
        llm_calls: int = 0,
        is_admin: bool = False,
        include_reasoning: bool = False,
        force_reasoning: bool = False,
        agent_response: AgentResponse | None = None,
        contract_valid: bool = False,
        terminal_status: str = "failed",
        error_category: str = "internal",
        anomaly_check: dict[str, Any] | None = None,
    ) -> QueryResponse:
        table_count = 0
        plot_count = 0
        value_count = 0
        json_count = 0
        values: dict[str, Any] = {}
        for artifact in artifacts:
            artifact_type = artifact.get("type")
            if artifact_type == "table":
                table_count += 1
            elif artifact_type == "plot":
                plot_count += 1
            elif artifact_type == "value":
                value_count += 1
                raw_values = artifact.get("data", {}).get("data", {})
                if isinstance(raw_values, dict):
                    values.update(make_json_safe(raw_values))
            elif artifact_type == "json":
                json_count += 1
        return QueryResponse(
            session_id=session_id,
            text=text,
            reasoning=reasoning if (include_reasoning or force_reasoning) else None,
            artifacts=artifacts,
            values=values or None,
            anomaly_check=anomaly_check,
            metrics=QueryMetrics(
                duration_ms=duration_ms,
                llm_duration_ms=llm_duration_ms,
                non_llm_duration_ms=max(0, duration_ms - llm_duration_ms),
                llm_calls=llm_calls,
                artifact_count=len(artifacts),
                table_count=table_count,
                plot_count=plot_count,
                value_count=value_count,
                json_count=json_count,
                model=display_model_name(is_admin=is_admin, model=model_name),
            ),
            response_envelope_valid=(
                agent_response.response_envelope_valid if agent_response is not None else True
            ),
            task_contract_satisfied=(
                agent_response.task_contract_satisfied if agent_response is not None else contract_valid
            ),
            contract_valid=(
                agent_response.task_contract_satisfied if agent_response is not None else contract_valid
            ),
            terminal_status=(
                agent_response.terminal_status if agent_response is not None else terminal_status
            ),
            error_category=(agent_response.error_category if agent_response is not None else error_category),
            partial_result=(agent_response.partial_result if agent_response is not None else False),
            capability_outcomes=(
                [outcome.model_dump(mode="json") for outcome in agent_response.capability_outcomes]
                if agent_response is not None
                else []
            ),
            error_fingerprints=(
                list(agent_response.error_fingerprints) if agent_response is not None else []
            ),
            retry_count=(agent_response.retry_count if agent_response is not None else 0),
            tool_error_count=(agent_response.tool_error_count if agent_response is not None else 0),
        )

    def _build_fallback_response(
        self,
        session_id: str,
        query: str,
        reason: str,
        duration_ms: int,
        model_name: str,
        *,
        is_admin: bool = False,
        include_reasoning: bool = False,
        force_reasoning: bool = False,
    ) -> QueryResponse:
        fallback_reasoning = (
            "Fallback response generated due to timeout."
            if reason == "timeout"
            else "Fallback response generated due to runtime error."
        )
        return self._build_response(
            session_id=session_id,
            text=self._fallback_text(query, reason),
            reasoning=fallback_reasoning,
            artifacts=[],
            duration_ms=duration_ms,
            model_name=model_name,
            is_admin=is_admin,
            include_reasoning=include_reasoning,
            force_reasoning=force_reasoning,
            contract_valid=False,
            terminal_status="unavailable" if reason == "timeout" else "failed",
            error_category="transport" if reason == "timeout" else "internal",
        )

    @staticmethod
    def _fallback_text(query: str, reason: str) -> str:
        if query.strip():
            short_query = query.strip()
            if len(short_query) > 220:
                short_query = short_query[:220] + "..."
            reason_text = "таймаут выполнения" if reason == "timeout" else "внутренняя техническая ошибка"
            return (
                "Я получил ваш запрос, но не смог завершить полноценный анализ "
                f"({reason_text}). Попробуйте повторить запрос.\n\n"
                f"Запрос: {short_query}"
            )
        return "Я получил запрос, но не смог сформировать содержательный ответ."

    @staticmethod
    def _build_reasoning_trace(
        *,
        response_text: str,
        response_reasoning: str | None,
        route: str | None,
        tool_collector: Any,
        use_history: bool,
        duration_ms: int,
        has_dataset: bool,
    ) -> str | None:
        normalized_route = (route or "").strip().lower()
        if normalized_route not in {"analysis", "summary", "report"}:
            normalized_route = "analysis"

        unique_tools: list[str] = []
        seen_tools: set[str] = set()
        for name in tool_collector.tool_names:
            normalized = str(name).strip()
            if not normalized or normalized in seen_tools:
                continue
            seen_tools.add(normalized)
            unique_tools.append(normalized)

        lines: list[str] = []

        cleaned_model_reasoning = (response_reasoning or "").strip()
        if cleaned_model_reasoning:
            if len(cleaned_model_reasoning) > 12000:
                cleaned_model_reasoning = f"{cleaned_model_reasoning[:12000]}..."
            lines.append("### Chain of Thought")
            lines.append(cleaned_model_reasoning)
            lines.append("")

        lines.extend(
            [
                "### Reason-Action Trace",
                f"- Route: `{normalized_route}`",
                f"- Dataset attached: `{'yes' if has_dataset else 'no'}`",
                f"- Use history: `{'yes' if use_history else 'no'}`",
                f"- Tool calls: `{tool_collector.tool_calls}`",
                f"- Duration: `{duration_ms / 1000:.1f}с`",
            ]
        )
        if unique_tools:
            lines.append(f"- Tools: `{', '.join(unique_tools)}`")

        end_events = [event for event in tool_collector.events if str(event.get("phase")) == "end"]
        if end_events:
            lines.append("")
            lines.append("### Tool events")
            for idx, event in enumerate(end_events[:8], start=1):
                tool_name = str(event.get("tool_name", "unknown"))
                status = str(event.get("status", "ok"))
                artifact_keys = event.get("artifact_keys")
                if isinstance(artifact_keys, list) and artifact_keys:
                    payload = ", ".join(str(item) for item in artifact_keys[:4])
                else:
                    payload = "none"
                event_line = f"{idx}. `{tool_name}` -> status: `{status}`, artifacts: `{payload}`"
                if status == "error":
                    error_raw = str(event.get("error", "")).strip()
                    if error_raw:
                        compact_error = error_raw.splitlines()[-1][:220]
                        event_line += f", error: `{compact_error}`"
                lines.append(event_line)

        if not cleaned_model_reasoning and not response_text.strip():
            lines.append("")
            lines.append("### Notes")
            lines.append("Fallback path used: model returned empty text.")

        reasoning = "\n".join(lines).strip()
        return reasoning or None

    @staticmethod
    def _merge_reasoning_text(*parts: str | None) -> str | None:
        normalized: list[str] = []
        seen: set[str] = set()
        for part in parts:
            clean = str(part or "").strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            normalized.append(clean)
        if not normalized:
            return None
        return "\n\n".join(normalized)

    @staticmethod
    def _build_reasoning_steps(
        raw_steps: list[str],
        tool_call_count: int,
        ordered_tool_names: list[str] | None = None,
    ) -> list[ReasoningStep]:
        from backend.agent.callbacks import _INFRA_TOOL_NAMES

        steps = raw_steps[:MAX_REASONING_STEPS]
        result: list[ReasoningStep] = []
        for index, content in enumerate(steps):
            if not content.strip():
                continue
            has_tool = index < tool_call_count
            tool_name_for_step: str | None = None
            if has_tool:
                tool_name_for_step = (
                    ordered_tool_names[index]
                    if ordered_tool_names and index < len(ordered_tool_names)
                    else None
                )
                is_infra = tool_name_for_step in _INFRA_TOOL_NAMES if tool_name_for_step else False
                if not is_infra:
                    continue
            is_last = index == len(steps) - 1
            if index == 0 and len(steps) > 1:
                kind: str = "planning"
            elif is_last:
                kind = "final_synthesis"
            else:
                kind = "tool_synthesis"
            step = ReasoningStep(
                step_index=index,
                kind=kind,  # type: ignore[arg-type]
                content=content,
                tool_name=tool_name_for_step,
            ).truncated()
            result.append(step)
        return result
