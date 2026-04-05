from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import replace
from typing import Annotated, Any

import anyio
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.agent.graph_tracker import ExecutionGraphTracker
from backend.api.deps import get_current_user
from backend.api.models import (
    QueryMetrics,
    QueryRequest,
    QueryResponse,
)
from backend.auth.auth_db import AuthDB, AuthUser
from backend.core.config import settings
from backend.core.json_utils import NumpyEncoder as _NumpyEncoder
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.sessions.session_store import SessionState, SessionStore
from backend.skills import SkillSelectionError

router = APIRouter(tags=["Запросы и агент"])

CHAT_FALLBACK_RE = re.compile(
    r"^(привет|здравствуй|здравствуйте|добрый|как дела|что нового|кто ты|помоги|hello|hi)\b",
    re.IGNORECASE,
)

# Singletons set during app startup
_auth_db: AuthDB = None  # type: ignore
_store: SessionStore = None  # type: ignore
_runner = None  # type: ignore
_db_runtime_service = None  # type: ignore
_search_integration_service = None  # type: ignore
_forecast_integration_service = None  # type: ignore
_anomaly_planfact_integration_service = None  # type: ignore
_rag_service = None  # type: ignore
_user_memory_service = None  # type: ignore
_build_trace_context_fn = None  # type: ignore
_query_trace_context_fn = None  # type: ignore
_settings = None  # type: ignore
_csv_runtime: CSVSessionRuntime = None  # type: ignore

# Callback classes set during startup
_LLMTextCollector = None  # type: ignore
_ToolCollector = None  # type: ignore
_AgentProgressCollector = None  # type: ignore
_PhaseCollector = None  # type: ignore
_TokenStreamCallbackHandler = None  # type: ignore
_AgentRunner = None  # type: ignore

_effective_enabled_tool_keys_fn = None  # type: ignore
_build_tool_catalog_fn = None  # type: ignore
_known_tool_keys = None  # type: ignore

logger = logging.getLogger(__name__)


def setup(
    auth_db: AuthDB,
    store: SessionStore,
    runner,
    db_runtime_service,
    search_integration_service,
    forecast_integration_service,
    anomaly_planfact_integration_service,
    rag_service,
    user_memory_service,
    build_trace_context_fn,
    query_trace_context_fn,
    app_settings,
    LLMTextCollector,
    ToolCollector,
    AgentProgressCollector,
    PhaseCollector,
    TokenStreamCallbackHandler,
    AgentRunner,
    effective_enabled_tool_keys_fn,
    build_tool_catalog_fn,
    known_tool_keys,
    csv_runtime: CSVSessionRuntime,
) -> None:
    global _auth_db, _store, _runner, _db_runtime_service
    global _search_integration_service, _forecast_integration_service
    global _anomaly_planfact_integration_service, _rag_service
    global _user_memory_service
    global _build_trace_context_fn, _query_trace_context_fn, _settings
    global _LLMTextCollector, _ToolCollector, _AgentProgressCollector
    global _PhaseCollector, _TokenStreamCallbackHandler
    global _AgentRunner, _effective_enabled_tool_keys_fn
    global _build_tool_catalog_fn, _known_tool_keys, _csv_runtime

    _auth_db = auth_db
    _store = store
    _runner = runner
    _db_runtime_service = db_runtime_service
    _search_integration_service = search_integration_service
    _forecast_integration_service = forecast_integration_service
    _anomaly_planfact_integration_service = anomaly_planfact_integration_service
    _rag_service = rag_service
    _user_memory_service = user_memory_service
    _build_trace_context_fn = build_trace_context_fn
    _query_trace_context_fn = query_trace_context_fn
    _settings = app_settings
    _LLMTextCollector = LLMTextCollector
    _ToolCollector = ToolCollector
    _AgentProgressCollector = AgentProgressCollector
    _PhaseCollector = PhaseCollector
    _TokenStreamCallbackHandler = TokenStreamCallbackHandler
    _AgentRunner = AgentRunner
    _effective_enabled_tool_keys_fn = effective_enabled_tool_keys_fn
    _build_tool_catalog_fn = build_tool_catalog_fn
    _known_tool_keys = known_tool_keys
    _csv_runtime = csv_runtime


# ── Helper functions ──────────────────────────────────────────────────────────

def _load_owned_session(session_id: str, current_user: AuthUser) -> SessionState:
    if not _auth_db.is_session_owner(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    state = _store.load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


def _session_source_payload(state: SessionState) -> dict[str, Any]:
    return {
        "source_type": state.source_type,
        "source_ref_id": state.source_ref_id,
        "source_label": state.source_label,
        "source_mode": state.source_mode,
    }


def _session_runtime_source_payload(state: SessionState) -> dict[str, Any]:
    payload = _session_source_payload(state)
    source_type = _session_source_type(state)
    if source_type == "csv":
        payload["csv_loaded"] = bool(state.csv_loaded)
        payload["csv_session_id"] = state.csv_session_id
        payload["csv_table_names"] = list(state.csv_table_names or [])
        payload["csv_expires_at"] = state.csv_expires_at
    else:
        payload["csv_loaded"] = False
        payload["csv_session_id"] = None
        payload["csv_table_names"] = []
        payload["csv_expires_at"] = None
    return payload


def _ensure_csv_runtime_state(session_id: str, state: SessionState) -> SessionState:
    if _session_source_type(state) != "csv":
        return state
    # Consider the session valid only if it exists AND has not expired (with 60s buffer).
    session_still_valid = (
        state.csv_loaded
        and bool(state.csv_session_id)
        and (state.csv_expires_at is None or state.csv_expires_at > int(time.time()) + 60)
    )
    if session_still_valid:
        return state
    if not state.df_path:
        raise HTTPException(status_code=400, detail="CSV dataset is not attached to this session")
    df = _store.get_dataframe(session_id)
    if df is None:
        raise HTTPException(status_code=400, detail="Failed to load CSV dataframe for this session")
    try:
        csv_info = _csv_runtime.register_dataframe(
            session_id=session_id,
            table_name=state.dataset_name or "uploaded.csv",
            df=df,
            ttl_seconds=settings.csv_session_ttl_sec,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to initialize CSV runtime: {exc}") from exc
    _store.set_csv_runtime_state(
        session_id,
        csv_loaded=True,
        csv_session_id=csv_info.session_id,
        csv_table_names=list(csv_info.table_names),
        csv_expires_at=csv_info.expires_at,
    )
    refreshed = _store.load_session(session_id)
    if refreshed is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return refreshed


def _session_db_connection_id(state: SessionState) -> str | None:
    if str(state.source_type or "").strip().lower() != "db_connection":
        return None
    ref_id = str(state.source_ref_id or "").strip()
    return ref_id or None


def _session_source_type(state: SessionState) -> str:
    return str(state.source_type or "").strip().lower()


def _active_session_dataframe(
    state: SessionState, session_id: str
) -> pd.DataFrame | None:
    if _session_source_type(state) != "csv":
        return None
    return _store.get_dataframe(session_id)


def _integration_source_descriptors() -> list[dict[str, Any]]:
    return [
        _search_integration_service.source_descriptor(),
        _rag_service.source_descriptor(),
        _forecast_integration_service.source_descriptor(),
        _anomaly_planfact_integration_service.source_descriptor(),
    ]


def _tool_catalog_payload(user_id: int) -> list[dict[str, Any]]:
    return _build_tool_catalog_fn(
        source_descriptors=_integration_source_descriptors(),
        user_settings=_auth_db.list_user_tool_settings(user_id),
    )


def _enabled_tool_keys_for_user(user_id: int) -> set[str]:
    return _effective_enabled_tool_keys_fn(_tool_catalog_payload(user_id))


def _effective_selected_skill_ids(
    state: SessionState,
    payload: QueryRequest,
) -> list[str]:
    if payload.selected_skill_ids is None:
        selected_skill_ids = list(state.selected_skill_ids or [])
    else:
        selected_skill_ids = [
            str(skill_id).strip()
            for skill_id in payload.selected_skill_ids
            if str(skill_id).strip()
        ]
    return list(dict.fromkeys(selected_skill_ids))


def _effective_runtime_settings(user_id: int, *, analysis_depth_override: str | None = None):
    user_runtime = _auth_db.get_user_settings(user_id)
    depth = analysis_depth_override or user_runtime.analysis_depth
    return replace(
        _settings,
        llm_temperature_chat=user_runtime.llm_temperature_chat,
        llm_temperature_tool=user_runtime.llm_temperature_tool,
        llm_max_tokens_default=user_runtime.llm_max_tokens_default,
        llm_max_tokens_reasoning=user_runtime.llm_max_tokens_reasoning,
        backend_query_timeout_sec=user_runtime.backend_query_timeout_sec,
        agent_max_steps=user_runtime.agent_max_steps,
        agent_step_timeout_sec=user_runtime.agent_step_timeout_sec,
        agent_inner_recursion_limit=user_runtime.agent_inner_recursion_limit,
        agent_analysis_depth=depth,
    )


def _build_stream_callbacks(
    *,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    session_source: dict[str, Any],
    exec_store: Any = None,
    include_reasoning: bool = True,
) -> tuple[list[Any], Any, Any, Any, Any, Any]:
    """Build the full callback stack for a streaming request.

    Returns: (callbacks, token_collector, tool_collector, progress_collector,
               phase_collector, graph_tracker)
    """
    text_collector = _LLMTextCollector()
    tool_collector = _ToolCollector(
        source_context=session_source, queue=queue, loop=loop, execution_store=exec_store
    )
    progress_collector = _AgentProgressCollector()
    phase_collector = _PhaseCollector()
    graph_tracker = ExecutionGraphTracker()
    phase_collector.graph_tracker = graph_tracker
    tool_collector.graph_tracker = graph_tracker
    tool_collector._phase_collector_ref = phase_collector  # noqa: SLF001
    token_collector = _TokenStreamCallbackHandler(queue, loop)
    callbacks: list[Any] = [
        token_collector,
        text_collector,
        tool_collector,
        progress_collector,
        phase_collector,
    ]
    return callbacks, token_collector, tool_collector, progress_collector, phase_collector, graph_tracker


def _sse_event(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, cls=_NumpyEncoder)
    return f"event: {event}\ndata: {payload}\n\n"


def _to_artifact_payload(artifact_dict: dict[str, Any]):
    from backend.api.models import ArtifactPayload
    return ArtifactPayload(**artifact_dict)


def _build_response(
    session_id: str,
    text: str,
    reasoning: str | None,
    artifacts: list[dict[str, Any]],
    duration_ms: int,
    model_name: str,
    include_reasoning: bool = False,
    force_reasoning: bool = False,
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
            values.update(artifact.get("data", {}).get("data", {}))
        elif artifact_type == "json":
            json_count += 1
    return QueryResponse(
        session_id=session_id,
        text=text,
        reasoning=reasoning if (include_reasoning or force_reasoning) else None,
        artifacts=artifacts,
        values=values or None,
        metrics=QueryMetrics(
            duration_ms=duration_ms,
            artifact_count=len(artifacts),
            table_count=table_count,
            plot_count=plot_count,
            value_count=value_count,
            json_count=json_count,
            model=model_name,
        ),
    )


def _fallback_text(query: str, reason: str) -> str:
    normalized = query.strip().lower()
    if CHAT_FALLBACK_RE.search(normalized):
        if "как дела" in normalized:
            return (
                "Все в порядке. Сейчас сервис ограничен, но я на связи и готов помочь."
            )
        return "Привет. Я на связи, но сейчас аналитический контур временно ограничен."
    if query.strip():
        short_query = query.strip()
        if len(short_query) > 220:
            short_query = short_query[:220] + "..."
        reason_text = (
            "таймаут выполнения"
            if reason == "timeout"
            else "внутренняя техническая ошибка"
        )
        return (
            "Я получил ваш запрос, но не смог завершить полноценный анализ "
            f"({reason_text}). Попробуйте повторить запрос.\n\n"
            f"Запрос: {short_query}"
        )
    return "Я получил запрос, но не смог сформировать содержательный ответ."


def _build_fallback_response(
    session_id: str,
    query: str,
    reason: str,
    duration_ms: int,
    model_name: str,
    include_reasoning: bool = False,
    force_reasoning: bool = False,
) -> QueryResponse:
    fallback_reasoning = (
        "Fallback response generated due to timeout."
        if reason == "timeout"
        else "Fallback response generated due to runtime error."
    )
    return _build_response(
        session_id=session_id,
        text=_fallback_text(query, reason),
        reasoning=fallback_reasoning,
        artifacts=[],
        duration_ms=duration_ms,
        model_name=model_name,
        include_reasoning=include_reasoning,
        force_reasoning=force_reasoning,
    )


def _persist_fallback_response(
    store, auth_db, session_id, user_id, query_text, error_text,
    reasoning=None, auto_title=None,
):
    _ = user_id
    store.add_chat_message(session_id, "user", query_text)
    store.add_chat_message(session_id, "ai", error_text, reasoning=reasoning)
    auth_db.update_session_after_reply(session_id, error_text, auto_title=auto_title)


def _build_reasoning_trace(
    *,
    response_text: str,
    response_reasoning: str | None,
    route: str | None,
    tool_collector,
    use_history: bool,
    duration_ms: int,
    has_dataset: bool,
) -> str | None:
    normalized_route = (route or "").strip().lower()
    if normalized_route not in {"chat", "analysis", "rag", "summary"}:
        normalized_route = "analysis" if has_dataset else "chat"

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

    end_events = [
        event
        for event in tool_collector.events
        if str(event.get("phase")) == "end"
    ]
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
            event_line = (
                f"{idx}. `{tool_name}` -> status: `{status}`, artifacts: `{payload}`"
            )
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


def _extract_tool_code_preview(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        for key in ("code", "input", "query"):
            candidate = parsed.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return text


def _trim_preview(text: str, limit: int = 1200) -> str:
    clean = str(text or "").strip()
    if len(clean) <= limit:
        return clean
    return f"{clean[:limit]}..."


def _build_live_reasoning_event(event: dict[str, Any], index: int) -> str | None:
    phase = str(event.get("phase", "")).strip().lower()
    tool_name = str(event.get("tool_name", "unknown")).strip() or "unknown"

    # Special rendering for skill loader — compact inline message, no code block.
    if tool_name == "get_tool_instructions":
        skill_id = ""
        raw_preview = str(event.get("input_preview", "")).strip()
        try:
            parsed = json.loads(raw_preview)
            skill_id = str(parsed.get("tool_name", "")).strip()
        except Exception:
            skill_id = raw_preview.strip("'\"")
        label = f"`{skill_id}`" if skill_id else "скил"
        if phase == "start":
            return f"📚 Загружаю инструкцию: {label}"
        if phase == "end":
            status = str(event.get("status", "ok")).strip()
            icon = "✅" if status == "ok" else "❌"
            return f"{icon} Инструкция {label} загружена"
        return None

    # Meta-tools that return plain text (no artifacts) — show compact one-liner,
    # never show misleading "status=empty_output, artifacts=none".
    _META_TOOLS = {"planner_tool", "review_tool"}
    if tool_name in _META_TOOLS:
        if phase == "start":
            return f"🧠 `{tool_name}` запущен"
        if phase == "end":
            status = str(event.get("status", "")).strip()
            if status == "error":
                error_text = str(event.get("error", "")).strip()
                hint = f": {_trim_preview(error_text.splitlines()[-1], 120)}" if error_text else ""
                return f"❌ `{tool_name}` завершен с ошибкой{hint}"
            return f"✅ `{tool_name}` завершен"
        return None

    if phase == "start":
        raw_input = _extract_tool_code_preview(str(event.get("input_preview", "")))
        if not raw_input:
            return f"### Live Tool #{index}\n`{tool_name}` запущен."
        return (
            f"### Live Tool #{index}\n"
            f"`{tool_name}` запущен.\n\n"
            "```python\n"
            f"{_trim_preview(raw_input, 900)}\n"
            "```"
        )

    if phase == "end":
        status = str(event.get("status", "ok")).strip() or "ok"
        artifact_keys = event.get("artifact_keys")
        artifacts = "none"
        if isinstance(artifact_keys, list) and artifact_keys:
            artifacts = ", ".join(str(item) for item in artifact_keys[:6])

        lines = [
            f"### Live Tool #{index}",
            f"`{tool_name}` завершен: status=`{status}`, artifacts=`{artifacts}`.",
        ]
        error_text = str(event.get("error", "")).strip()
        if error_text:
            lines.append(f"- error: `{_trim_preview(error_text.splitlines()[-1], 220)}`")
        code_preview = str(event.get("code_preview", "")).strip()
        if code_preview:
            lines.append("")
            lines.append("```python")
            lines.append(_trim_preview(code_preview, 900))
            lines.append("```")
        return "\n".join(lines)

    return None


def _build_live_agent_progress_event(event: dict[str, Any], index: int) -> str | None:
    title = str(event.get("title", "")).strip() or f"Ход анализа #{index}"
    details = _trim_preview(str(event.get("details", "")).strip(), 1400)
    step_index = event.get("step_index")
    max_steps = event.get("max_steps")

    lines = [f"### {title}"]
    if isinstance(step_index, int) and isinstance(max_steps, int) and max_steps > 0:
        lines.append(f"Шаг: `{step_index}/{max_steps}`")
    if details:
        lines.append(details)
    return "\n".join(lines)


# ── Route handlers ────────────────────────────────────────────────────────────

async def _execute_query(
    session_id: str,
    payload: QueryRequest,
    *,
    persist: bool,
    current_user: AuthUser,
    callbacks: list[object] | None = None,
) -> QueryResponse:
    state = _load_owned_session(session_id, current_user)
    if _session_source_type(state) == "csv":
        state = _ensure_csv_runtime_state(session_id, state)
    selected_skill_ids = _effective_selected_skill_ids(state, payload)
    df = _active_session_dataframe(state, session_id)
    session_source = _session_runtime_source_payload(state)
    session_db_connection_id = _session_db_connection_id(state)
    has_active_source = (
        df is not None
        or session_db_connection_id is not None
        or bool(session_source.get("csv_loaded"))
    )

    from backend.artifacts.execution import ExecutionStore
    exec_store = ExecutionStore(session_id=session_id)
    text_collector = _LLMTextCollector()
    tool_collector = _ToolCollector(source_context=session_source, execution_store=exec_store)
    active_callbacks = list(callbacks or [])
    active_callbacks.extend([text_collector, tool_collector])
    started_at = time.perf_counter()
    _auth_db.touch_session(session_id)
    trace_context = _build_trace_context_fn(
        session_id=session_id,
        user_id=current_user.id,
        username=current_user.username,
        request_kind="query" if persist else "evaluate",
        use_history=payload.use_history,
        include_reasoning=payload.include_reasoning,
        db_connection_id=session_db_connection_id,
        csv_session_id=session_source.get("csv_session_id"),
        csv_duckdb_loaded=bool(session_source.get("csv_loaded")),
    )
    runtime_settings = _effective_runtime_settings(
        current_user.id, analysis_depth_override=payload.analysis_depth
    )
    allowed_tool_keys = _enabled_tool_keys_for_user(current_user.id)
    user_memory = _user_memory_service.load(current_user.id)
    from backend.sessions.session_memory import SessionMemory as _SessionMemory
    session_memory = _SessionMemory(notes=state.session_memory or "")
    runtime_runner = _AgentRunner(
        runtime_settings,
        db_runtime_service=_db_runtime_service,
        search_service=_search_integration_service,
        forecast_service=_forecast_integration_service,
        anomaly_planfact_service=_anomaly_planfact_integration_service,
        rag_service=_rag_service,
        allowed_tool_keys=allowed_tool_keys,
        user_memory=user_memory,
        session_memory=session_memory,
        skill_registry=_runner.skill_registry,
    )

    try:
        runtime_runner.skill_registry.resolve_selection(selected_skill_ids)
        with _query_trace_context_fn(
            session_id=session_id,
            user_id=current_user.id,
            username=current_user.username,
            request_kind="query" if persist else "evaluate",
            use_history=payload.use_history,
            include_reasoning=payload.include_reasoning,
            query=payload.query,
            db_connection_id=session_db_connection_id,
            csv_session_id=session_source.get("csv_session_id"),
            csv_duckdb_loaded=bool(session_source.get("csv_loaded")),
        ):
            with anyio.fail_after(runtime_settings.backend_query_timeout_sec):
                response = await anyio.to_thread.run_sync(
                    runtime_runner.run_query,
                    df,
                    payload.query,
                    state.chat_history,
                    payload.use_history,
                    payload.include_reasoning,
                    active_callbacks,
                    trace_context,
                    session_source,
                    selected_skill_ids,
                )
    except SkillSelectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except TimeoutError:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        fallback = _build_fallback_response(
            session_id=session_id,
            query=payload.query,
            reason="timeout",
            duration_ms=duration_ms,
            model_name=runtime_settings.llm_model,
            include_reasoning=payload.include_reasoning,
            force_reasoning=persist,
        )
        if persist:
            _persist_fallback_response(
                _store, _auth_db, session_id, current_user.id,
                payload.query, fallback.text, reasoning=fallback.reasoning,
            )
        return fallback
    except Exception:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        fallback = _build_fallback_response(
            session_id=session_id,
            query=payload.query,
            reason="runtime_error",
            duration_ms=duration_ms,
            model_name=runtime_settings.llm_model,
            include_reasoning=payload.include_reasoning,
            force_reasoning=persist,
        )
        if persist:
            _persist_fallback_response(
                _store, _auth_db, session_id, current_user.id,
                payload.query, fallback.text, reasoning=fallback.reasoning,
            )
        return fallback

    try:
        if runtime_runner._user_memory_buffer:  # noqa: SLF001
            _mem_llm = runtime_runner._build_llm(  # noqa: SLF001
                role="chat", include_reasoning=False, max_tokens_override=800
            )
            _user_memory_service.schedule_consolidation(
                current_user.id,
                list(runtime_runner._user_memory_buffer),  # noqa: SLF001
                _mem_llm.invoke,
            )
    except Exception:
        pass

    try:
        if runtime_runner._session_memory_buffer:  # noqa: SLF001
            for note in runtime_runner._session_memory_buffer:  # noqa: SLF001
                _store.append_session_memory(session_id, note)
    except Exception:
        pass

    from backend.artifacts.bridge import execution_to_api_payload
    artifacts = [execution_to_api_payload(a) for a in response.artifacts]
    duration_ms = int((time.perf_counter() - started_at) * 1000)
    effective_reasoning = _build_reasoning_trace(
        response_text=response.final_text,
        response_reasoning=response.reasoning,
        route=response.route,
        tool_collector=tool_collector,
        use_history=payload.use_history,
        duration_ms=duration_ms,
        has_dataset=has_active_source,
    )
    if persist:
        _store.set_selected_skill_ids(session_id, selected_skill_ids)
        _store.add_chat_message(session_id, "user", payload.query)
        _store.add_chat_message(
            session_id,
            "ai",
            response.final_text,
            artifacts=artifacts,
            reasoning=effective_reasoning,
        )
        _store.add_artifacts(session_id, response.artifacts)
        _auth_db.update_session_after_reply(
            session_id, response.final_text, auto_title=None
        )

    return _build_response(
        session_id,
        response.final_text,
        effective_reasoning,
        artifacts,
        duration_ms,
        runtime_settings.llm_model,
        include_reasoning=payload.include_reasoning,
        force_reasoning=persist,
    )


@router.post("/sessions/{session_id}/query", response_model=QueryResponse)
async def query(
    session_id: str,
    payload: QueryRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> QueryResponse:
    return await _execute_query(
        session_id,
        payload,
        persist=True,
        current_user=current_user,
    )


@router.post("/sessions/{session_id}/evaluate", response_model=QueryResponse)
async def evaluate(
    session_id: str,
    payload: QueryRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> QueryResponse:
    return await _execute_query(
        session_id,
        payload,
        persist=False,
        current_user=current_user,
    )


@router.post("/sessions/{session_id}/query/stream")
async def query_stream(
    session_id: str,
    payload: QueryRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> StreamingResponse:
    state = _load_owned_session(session_id, current_user)
    if _session_source_type(state) == "csv":
        state = _ensure_csv_runtime_state(session_id, state)
    selected_skill_ids = _effective_selected_skill_ids(state, payload)
    df = _active_session_dataframe(state, session_id)
    session_source = _session_runtime_source_payload(state)
    session_db_connection_id = _session_db_connection_id(state)
    has_active_source = (
        df is not None
        or session_db_connection_id is not None
        or bool(session_source.get("csv_loaded"))
    )

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    agent_finished = asyncio.Event()

    from backend.artifacts.execution import ExecutionStore
    exec_store = ExecutionStore(session_id=session_id)
    callbacks, token_collector, tool_collector, progress_collector, phase_collector, _graph_tracker = (
        _build_stream_callbacks(
            queue=queue,
            loop=loop,
            session_source=session_source,
            exec_store=exec_store,
        )
    )
    started_at = time.perf_counter()
    _auth_db.touch_session(session_id)
    trace_context = _build_trace_context_fn(
        session_id=session_id,
        user_id=current_user.id,
        username=current_user.username,
        request_kind="stream",
        use_history=payload.use_history,
        include_reasoning=payload.include_reasoning,
        db_connection_id=session_db_connection_id,
        csv_session_id=session_source.get("csv_session_id"),
        csv_duckdb_loaded=bool(session_source.get("csv_loaded")),
    )
    runtime_settings = _effective_runtime_settings(
        current_user.id, analysis_depth_override=payload.analysis_depth
    )
    allowed_tool_keys = _enabled_tool_keys_for_user(current_user.id)
    user_memory = _user_memory_service.load(current_user.id)
    from backend.sessions.session_memory import SessionMemory as _SessionMemory
    session_memory = _SessionMemory(notes=state.session_memory or "")
    runtime_runner = _AgentRunner(
        runtime_settings,
        db_runtime_service=_db_runtime_service,
        search_service=_search_integration_service,
        forecast_service=_forecast_integration_service,
        anomaly_planfact_service=_anomaly_planfact_integration_service,
        rag_service=_rag_service,
        allowed_tool_keys=allowed_tool_keys,
        user_memory=user_memory,
        session_memory=session_memory,
        skill_registry=_runner.skill_registry,
    )

    async def run_agent() -> None:
        try:
            runtime_runner.skill_registry.resolve_selection(selected_skill_ids)
            with _query_trace_context_fn(
                session_id=session_id,
                user_id=current_user.id,
                username=current_user.username,
                request_kind="stream",
                use_history=payload.use_history,
                include_reasoning=payload.include_reasoning,
                query=payload.query,
                db_connection_id=session_db_connection_id,
                csv_session_id=session_source.get("csv_session_id"),
                csv_duckdb_loaded=bool(session_source.get("csv_loaded")),
            ):
                with anyio.fail_after(runtime_settings.backend_query_timeout_sec):
                    response = await anyio.to_thread.run_sync(
                        runtime_runner.run_query,
                        df,
                        payload.query,
                        state.chat_history,
                        payload.use_history,
                        payload.include_reasoning,
                        callbacks,
                        trace_context,
                        session_source,
                        selected_skill_ids,
                    )
            try:
                if runtime_runner._user_memory_buffer:  # noqa: SLF001
                    _mem_llm = runtime_runner._build_llm(  # noqa: SLF001
                role="chat", include_reasoning=False, max_tokens_override=800
            )
                    _user_memory_service.schedule_consolidation(
                        current_user.id,
                        list(runtime_runner._user_memory_buffer),  # noqa: SLF001
                        _mem_llm.invoke,
                    )
            except Exception:
                pass

            try:
                if runtime_runner._session_memory_buffer:  # noqa: SLF001
                    for note in runtime_runner._session_memory_buffer:  # noqa: SLF001
                        _store.append_session_memory(session_id, note)
            except Exception:
                pass

            duration_ms = int((time.perf_counter() - started_at) * 1000)
            streamed_reasoning = token_collector.collected_reasoning()
            merged_reasoning = _merge_reasoning_text(
                response.reasoning,
                streamed_reasoning,
            )
            try:
                from backend.artifacts.bridge import execution_to_api_payload

                artifacts = [execution_to_api_payload(a) for a in response.artifacts]
                effective_reasoning = _build_reasoning_trace(
                    response_text=response.final_text,
                    response_reasoning=merged_reasoning,
                    route=response.route,
                    tool_collector=tool_collector,
                    use_history=payload.use_history,
                    duration_ms=duration_ms,
                    has_dataset=has_active_source,
                )
                _store.set_selected_skill_ids(session_id, selected_skill_ids)
                _store.add_chat_message(session_id, "user", payload.query)
                _store.add_chat_message(
                    session_id,
                    "ai",
                    response.final_text,
                    artifacts=artifacts,
                    reasoning=effective_reasoning,
                )
                _store.add_artifacts(session_id, response.artifacts)
                _auth_db.update_session_after_reply(
                    session_id, response.final_text, auto_title=None
                )
            except Exception:
                logger.exception(
                    "query_stream post-processing failed; returning agent response without persistence "
                    "session_id=%s user_id=%s",
                    session_id,
                    current_user.id,
                )
                artifacts = []
                effective_reasoning = merged_reasoning or response.reasoning

            final_payload = _build_response(
                session_id,
                response.final_text,
                effective_reasoning,
                artifacts,
                duration_ms,
                runtime_settings.llm_model,
                include_reasoning=payload.include_reasoning,
                force_reasoning=True,
            ).model_dump()
            # Attach execution graph to final payload.
            gt = phase_collector.graph_tracker
            if gt is not None and gt:
                final_payload["execution_graph"] = gt.snapshot()
            await queue.put(("final", final_payload))
        except SkillSelectionError as exc:
            await queue.put(_sse_event("error", {"detail": str(exc)}))
            return
        except TimeoutError:
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            fallback_payload = _build_fallback_response(
                session_id=session_id,
                query=payload.query,
                reason="timeout",
                duration_ms=duration_ms,
                model_name=runtime_settings.llm_model,
                include_reasoning=payload.include_reasoning,
                force_reasoning=True,
            )
            _persist_fallback_response(
                _store, _auth_db, session_id, current_user.id,
                payload.query, fallback_payload.text,
                reasoning=fallback_payload.reasoning,
            )
            await queue.put(("final", fallback_payload.model_dump()))
        except Exception:
            logger.exception(
                "query_stream failed; returning fallback response session_id=%s user_id=%s",
                session_id,
                current_user.id,
            )
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            fallback_payload = _build_fallback_response(
                session_id=session_id,
                query=payload.query,
                reason="runtime_error",
                duration_ms=duration_ms,
                model_name=runtime_settings.llm_model,
                include_reasoning=payload.include_reasoning,
                force_reasoning=True,
            )
            _persist_fallback_response(
                _store, _auth_db, session_id, current_user.id,
                payload.query, fallback_payload.text,
                reasoning=fallback_payload.reasoning,
            )
            await queue.put(("final", fallback_payload.model_dump()))
        finally:
            agent_finished.set()
            await queue.put(("done", None))

    emit_progress = payload.include_reasoning

    async def emit_live_reasoning() -> None:
        emitted_progress_events = 0
        emitted_tool_events = 0
        emitted_phase_events = 0
        emitted_graph_version = 0
        _loop_count = 0
        while True:
            _loop_count += 1
            emitted_any = False

            while emitted_phase_events < len(phase_collector.events):
                current = phase_collector.events[emitted_phase_events]
                emitted_phase_events += 1
                await queue.put(("phase", current))
                emitted_any = True

            # Emit execution graph updates.
            gt = phase_collector.graph_tracker
            if gt is not None:
                gv = phase_collector._graph_version  # noqa: SLF001
                if gv > emitted_graph_version:
                    emitted_graph_version = gv
                    await queue.put(("execution_graph", gt.snapshot()))
                    emitted_any = True

            if emit_progress:
                while emitted_progress_events < len(progress_collector.events):
                    current = progress_collector.events[emitted_progress_events]
                    emitted_progress_events += 1
                    text = _build_live_agent_progress_event(current, emitted_progress_events)
                    if text:
                        await queue.put(("reasoning", text))
                        emitted_any = True

                while emitted_tool_events < len(tool_collector.events):
                    current = tool_collector.events[emitted_tool_events]
                    emitted_tool_events += 1
                    text = _build_live_reasoning_event(current, emitted_tool_events)
                    if text:
                        await queue.put(("reasoning", text))
                        emitted_any = True

            if agent_finished.is_set():
                all_drained = emitted_phase_events >= len(phase_collector.events)
                if emit_progress:
                    all_drained = all_drained and (
                        emitted_tool_events >= len(tool_collector.events)
                        and emitted_progress_events >= len(progress_collector.events)
                    )
                if all_drained:
                    logger.warning("REASONING_TASK: exiting after %d loops, agent_finished=True", _loop_count)
                    break

            if not emitted_any:
                await asyncio.sleep(0.02)

    async def event_generator():
        yield _sse_event("start", {"session_id": session_id})
        agent_task = asyncio.create_task(run_agent())
        reasoning_task = asyncio.create_task(emit_live_reasoning())
        deferred_final: list[tuple[str, Any]] = []
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
                    yield _sse_event(extra_event, extra_data)
                for final_event, final_data in deferred_final:
                    yield _sse_event(final_event, final_data)
                break
            if event == "final":
                deferred_final.append((event, data))
                continue
            yield _sse_event(event, data)
        await agent_task
        await reasoning_task

    return StreamingResponse(event_generator(), media_type="text/event-stream")
