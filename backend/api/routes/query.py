from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.agent.graph_tracker import ExecutionGraphTracker
from backend.api.deps import get_current_user
from backend.api.models import (
    QueryRequest,
    QueryResponse,
)
from backend.api.services.query_execution import (
    QueryExecutionDependencies,
    QueryExecutionRequest,
    QueryExecutionService,
    QueryStreamExecutionRequest,
)
from backend.auth.auth_db import AuthDB, AuthUser
from backend.core.config import settings
from backend.core.json_utils import NumpyEncoder as _NumpyEncoder
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.notebook.manifest_store import ManifestStore
from backend.sessions.session_store import SessionBusyError, SessionStore
from backend.skills.registry import SkillRegistry

router = APIRouter(tags=["Запросы и агент"])

# Singletons set during app startup
_auth_db: AuthDB = None  # type: ignore
_store: SessionStore = None  # type: ignore
_skill_registry: SkillRegistry = None  # type: ignore
_db_runtime_service = None  # type: ignore
_forecast_integration_service = None  # type: ignore
_anomaly_planfact_integration_service = None  # type: ignore
_rag_service = None  # type: ignore
_user_memory_service = None  # type: ignore
_build_trace_context_fn = None  # type: ignore
_query_trace_context_fn = None  # type: ignore
_settings = None  # type: ignore
_csv_runtime: CSVSessionRuntime = None  # type: ignore
_manifest_store: ManifestStore = None  # type: ignore
_blob_store = None  # type: ignore
_storage_dir: Path | None = None
_query_execution_service: QueryExecutionService = None  # type: ignore

# Callback classes set during startup
_LLMTextCollector = None  # type: ignore
_ToolCollector = None  # type: ignore
_PhaseCollector = None  # type: ignore
_TokenStreamCallbackHandler = None  # type: ignore
_ContextUsageCollector = None  # type: ignore
_AgentRunner = None  # type: ignore

_effective_enabled_tool_keys_fn = None  # type: ignore
_build_tool_catalog_fn = None  # type: ignore
_known_tool_keys = None  # type: ignore
_mcp_service = None  # type: ignore
_semantic_context_builder = None  # type: ignore
_semantic_catalog_service = None  # type: ignore
_semantic_generation_service = None  # type: ignore


def setup(
    auth_db: AuthDB,
    store: SessionStore,
    skill_registry: SkillRegistry,
    db_runtime_service,
    forecast_integration_service,
    anomaly_planfact_integration_service,
    rag_service,
    user_memory_service,
    build_trace_context_fn,
    query_trace_context_fn,
    app_settings,
    LLMTextCollector,
    ToolCollector,
    PhaseCollector,
    TokenStreamCallbackHandler,
    ContextUsageCollector,
    AgentRunner,
    effective_enabled_tool_keys_fn,
    build_tool_catalog_fn,
    known_tool_keys,
    csv_runtime: CSVSessionRuntime,
    manifest_store: ManifestStore,
    blob_store=None,
    storage_dir: str | Path | None = None,
    mcp_service=None,
    semantic_context_builder=None,
    semantic_catalog_service=None,
    semantic_generation_service=None,
) -> None:
    global _auth_db, _store, _skill_registry, _db_runtime_service
    global _forecast_integration_service
    global _anomaly_planfact_integration_service, _rag_service
    global _user_memory_service
    global _build_trace_context_fn, _query_trace_context_fn, _settings
    global _LLMTextCollector, _ToolCollector
    global _PhaseCollector, _TokenStreamCallbackHandler
    global _ContextUsageCollector, _AgentRunner, _effective_enabled_tool_keys_fn
    global _build_tool_catalog_fn, _known_tool_keys, _csv_runtime
    global _manifest_store, _blob_store, _storage_dir, _query_execution_service, _mcp_service
    global _semantic_context_builder, _semantic_catalog_service, _semantic_generation_service

    _auth_db = auth_db
    _store = store
    _skill_registry = skill_registry
    _db_runtime_service = db_runtime_service
    _forecast_integration_service = forecast_integration_service
    _anomaly_planfact_integration_service = anomaly_planfact_integration_service
    _rag_service = rag_service
    _user_memory_service = user_memory_service
    _build_trace_context_fn = build_trace_context_fn
    _query_trace_context_fn = query_trace_context_fn
    _settings = app_settings
    _LLMTextCollector = LLMTextCollector
    _ToolCollector = ToolCollector
    _PhaseCollector = PhaseCollector
    _TokenStreamCallbackHandler = TokenStreamCallbackHandler
    _ContextUsageCollector = ContextUsageCollector
    _AgentRunner = AgentRunner
    _effective_enabled_tool_keys_fn = effective_enabled_tool_keys_fn
    _build_tool_catalog_fn = build_tool_catalog_fn
    _known_tool_keys = known_tool_keys
    _mcp_service = mcp_service
    _semantic_context_builder = semantic_context_builder
    _semantic_catalog_service = semantic_catalog_service
    _semantic_generation_service = semantic_generation_service
    _csv_runtime = csv_runtime
    _manifest_store = manifest_store
    _blob_store = blob_store
    _storage_dir = Path(storage_dir) if storage_dir is not None else Path(settings.storage_dir)
    _query_execution_service = QueryExecutionService(
        dependencies=QueryExecutionDependencies(
            auth_db=_auth_db,
            store=_store,
            skill_registry=_skill_registry,
            db_runtime_service=_db_runtime_service,
            forecast_integration_service=_forecast_integration_service,
            anomaly_planfact_integration_service=_anomaly_planfact_integration_service,
            rag_service=_rag_service,
            user_memory_service=_user_memory_service,
            build_trace_context_fn=_build_trace_context_fn,
            query_trace_context_fn=_query_trace_context_fn,
            settings=_settings,
            csv_runtime=_csv_runtime,
            manifest_store=_manifest_store,
            storage_dir=_storage_dir,
            blob_store=_blob_store,
            llm_text_collector_cls=_LLMTextCollector,
            tool_collector_cls=_ToolCollector,
            build_stream_callbacks_fn=_build_stream_callbacks,
            agent_runner_cls=_AgentRunner,
            effective_enabled_tool_keys_fn=_effective_enabled_tool_keys_fn,
            build_tool_catalog_fn=_build_tool_catalog_fn,
            mcp_service=_mcp_service,
            semantic_context_builder=_semantic_context_builder,
            semantic_catalog_service=_semantic_catalog_service,
            semantic_generation_service=_semantic_generation_service,
        )
    )


def _build_stream_callbacks(
    *,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    session_source: dict[str, Any],
    exec_store: Any = None,
    include_reasoning: bool = True,
    show_thinking: bool = True,
) -> tuple[list[Any], Any, Any, Any, Any]:
    """Build callback stack for a streaming request."""
    text_collector = _LLMTextCollector()
    tool_collector = _ToolCollector(
        source_context=session_source,
        queue=queue,
        loop=loop,
        execution_store=exec_store,
    )
    phase_collector = _PhaseCollector()
    graph_tracker = ExecutionGraphTracker()
    phase_collector.graph_tracker = graph_tracker
    tool_collector.graph_tracker = graph_tracker
    tool_collector._phase_collector_ref = phase_collector  # noqa: SLF001
    token_collector = _TokenStreamCallbackHandler(
        queue,
        loop,
        show_think=settings.llm_show_think and show_thinking,
        enable_thinking=settings.llm_enable_thinking and include_reasoning,
    )
    context_usage_collector = _ContextUsageCollector(queue, loop)
    tool_collector.token_callback = token_collector
    callbacks: list[Any] = [
        token_collector,
        text_collector,
        tool_collector,
        phase_collector,
        context_usage_collector,
    ]
    return callbacks, token_collector, tool_collector, phase_collector, graph_tracker


def _sse_event(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, cls=_NumpyEncoder)
    return f"event: {event}\ndata: {payload}\n\n"


# Route handlers


@router.post("/sessions/{session_id}/query", response_model=QueryResponse)
async def query(
    session_id: str,
    payload: QueryRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> QueryResponse:
    try:
        return await _query_execution_service.execute(
            QueryExecutionRequest(
                session_id=session_id,
                payload=payload,
                current_user=current_user,
                persist=True,
            )
        )
    except SessionBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/sessions/{session_id}/evaluate", response_model=QueryResponse)
async def evaluate(
    session_id: str,
    payload: QueryRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> QueryResponse:
    try:
        return await _query_execution_service.execute(
            QueryExecutionRequest(
                session_id=session_id,
                payload=payload,
                current_user=current_user,
                persist=False,
            )
        )
    except SessionBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/sessions/{session_id}/query/stream")
async def query_stream(
    session_id: str,
    payload: QueryRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> StreamingResponse:
    try:
        stream_context = _query_execution_service.prepare_stream(
            QueryStreamExecutionRequest(
                session_id=session_id,
                payload=payload,
                current_user=current_user,
            )
        )
    except SessionBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    async def event_generator():
        async for event, data in _query_execution_service.stream_events(stream_context):
            yield _sse_event(event, data)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
