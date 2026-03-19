from __future__ import annotations

import asyncio
import io
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import anyio
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.agent_runner import AgentRunner
from backend.auth_db import AuthDB, AuthUser
from backend.callbacks import (
    AgentProgressCollector,
    LLMTextCollector,
    PhaseCollector,
    PhaseTokenStreamHandler,
    TokenStreamCallbackHandler,
    ToolCollector,
)
from backend.config import settings
from backend.db_connections_service import DBConnectionsService
from backend.db_runtime_service import DBRuntimeService
from backend.models import (
    AdminCreateUserRequest,
    AdminUpdateUserRequest,
    AuthChangePasswordRequest,
    AuthLoginRequest,
    AuthRegisterRequest,
    AuthResponse,
    AuthUserResponse,
    CreateSessionResponse,
    DBConnectionCreateRequest,
    DBConnectionResponse,
    DBConnectionSchemaResponse,
    DBConnectionTableResponse,
    DBConnectionTestResponse,
    DBConnectionUpdateRequest,
    MessageResponse,
    PhoenixOverviewResponse,
    QueryMetrics,
    QueryRequest,
    QueryResponse,
    SessionBindDBConnectionSourceRequest,
    SessionSourceStateResponse,
    SessionTitleUpdateRequest,
    SessionStateResponse,
    SessionSummaryResponse,
    UserSettingsResponse,
    UserSettingsUpdateRequest,
    UploadResponse,
)
from backend.observability import initialize_phoenix
from backend.observability import build_trace_context, query_trace_context
from backend.observability_service import PhoenixObservabilityService
from backend.serialization import serialize_artifact
from backend.session_store import SessionState, SessionStore


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    await anyio.to_thread.run_sync(runner.warmup)
    yield


app = FastAPI(title="LLM Data Analyst Backend", version="0.2.0", lifespan=_lifespan)

origins = (
    ["*"]
    if settings.cors_allow_origins.strip() == "*"
    else [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = SessionStore(settings.storage_dir, settings.session_ttl_days)
auth_db = AuthDB(settings.auth_db_path, settings.auth_token_ttl_days)
auth_db.ensure_default_admin(
    settings.auth_default_admin_username,
    settings.auth_default_admin_password,
)
db_connections_service = DBConnectionsService(auth_db, settings)
db_runtime_service = DBRuntimeService(db_connections_service)
runner = AgentRunner(settings, db_runtime_service=db_runtime_service)
phoenix_observability_service = PhoenixObservabilityService(settings)
initialize_phoenix()

CHAT_FALLBACK_RE = re.compile(
    r"^(привет|здравствуй|здравствуйте|добрый|как дела|что нового|кто ты|помоги|hello|hi)\b",
    re.IGNORECASE,
)



def _to_user_response(user: AuthUser) -> AuthUserResponse:
    return AuthUserResponse(
        id=user.id,
        username=user.username,
        is_admin=user.is_admin,
        created_at=user.created_at,
    )


def _to_db_connection_response(connection) -> DBConnectionResponse:
    return DBConnectionResponse(
        id=connection.id,
        name=connection.name,
        db_type=connection.db_type,
        host=connection.host,
        port=connection.port,
        database=connection.database,
        username=connection.username,
        options_json=connection.options_json,
        password_present=connection.password_present,
        last_test_at=connection.last_test_at,
        last_test_ok=connection.last_test_ok,
        last_error=connection.last_error,
        created_at=connection.created_at,
        updated_at=connection.updated_at,
    )


def _build_internal_db_tool_context(user_id: int, connection_id: str) -> dict[str, Any]:
    """Internal example contract for future DB-aware tools/runtime consumers."""
    return db_runtime_service.build_demo_tool_contract(
        user_id=user_id,
        connection_id=connection_id,
    )


def _session_source_payload(state: SessionState) -> dict[str, Any]:
    return {
        "source_type": state.source_type,
        "source_ref_id": state.source_ref_id,
        "source_label": state.source_label,
        "source_mode": state.source_mode,
    }


def _to_session_source_response(state: SessionState) -> SessionSourceStateResponse:
    return SessionSourceStateResponse(**_session_source_payload(state))


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
    return store.get_dataframe(session_id)


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.strip().partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _require_token(authorization: str | None = Header(default=None)) -> str:
    token = _extract_bearer_token(authorization)
    if token is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return token


def get_current_user(token: str = Depends(_require_token)) -> AuthUser:
    user = auth_db.get_user_by_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


def get_admin_user(current_user: AuthUser = Depends(get_current_user)) -> AuthUser:
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def _to_settings_response(user_id: int) -> UserSettingsResponse:
    settings_row = auth_db.get_user_settings(user_id)
    return UserSettingsResponse(
        theme=settings_row.theme,
        default_include_reasoning=settings_row.default_include_reasoning,
        default_answer_style=settings_row.default_answer_style,
        analysis_depth=settings_row.analysis_depth,
        llm_temperature_chat=settings_row.llm_temperature_chat,
        llm_temperature_tool=settings_row.llm_temperature_tool,
        llm_max_tokens_default=settings_row.llm_max_tokens_default,
        llm_max_tokens_reasoning=settings_row.llm_max_tokens_reasoning,
        backend_query_timeout_sec=settings_row.backend_query_timeout_sec,
        agent_max_steps=settings_row.agent_max_steps,
        agent_step_timeout_sec=settings_row.agent_step_timeout_sec,
        agent_inner_recursion_limit=settings_row.agent_inner_recursion_limit,
    )


def _effective_runtime_settings(user_id: int, *, analysis_depth_override: str | None = None):
    user_runtime = auth_db.get_user_settings(user_id)
    depth = analysis_depth_override or user_runtime.analysis_depth
    return replace(
        settings,
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


def _load_owned_session(session_id: str, current_user: AuthUser) -> SessionState:
    if not auth_db.is_session_owner(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    state = store.load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


def _collect_user_queries_for_title(state: SessionState) -> list[str]:
    queries: list[str] = []
    for message in state.chat_history:
        if str(message.get("role", "")).strip().lower() != "user":
            continue
        content = str(message.get("content", "")).strip()
        if content:
            queries.append(content)
    return queries


def _dataset_hint_for_title(
    state: SessionState, df: pd.DataFrame | None
) -> str | None:
    raw_name = str(state.dataset_name or "").strip()
    if raw_name:
        stem = Path(raw_name).name.rsplit(".", 1)[0].strip()
        if stem:
            return stem
    if df is not None:
        return f"dataset {len(df)}x{len(df.columns)}"
    return None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/runtime/model")
def runtime_model(current_user: AuthUser = Depends(get_current_user)) -> dict[str, str]:
    _ = current_user
    return {
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
    }


@app.get("/observability/phoenix", response_model=PhoenixOverviewResponse)
def phoenix_overview(
    current_user: AuthUser = Depends(get_current_user),
) -> PhoenixOverviewResponse:
    _ = current_user
    return phoenix_observability_service.build_overview()


@app.post("/auth/register", response_model=AuthResponse)
def auth_register(payload: AuthRegisterRequest) -> AuthResponse:
    try:
        user = auth_db.create_user(payload.username, payload.password, is_admin=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=409, detail="Username already exists")

    token = auth_db.create_access_token(user.id)
    return AuthResponse(access_token=token, user=_to_user_response(user))


@app.post("/auth/login", response_model=AuthResponse)
def auth_login(payload: AuthLoginRequest) -> AuthResponse:
    user = auth_db.authenticate(payload.username, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = auth_db.create_access_token(user.id)
    return AuthResponse(access_token=token, user=_to_user_response(user))


@app.get("/auth/me", response_model=AuthUserResponse)
def auth_me(current_user: AuthUser = Depends(get_current_user)) -> AuthUserResponse:
    return _to_user_response(current_user)


@app.post("/auth/change-password", response_model=MessageResponse)
def auth_change_password(
    payload: AuthChangePasswordRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> MessageResponse:
    try:
        auth_db.update_password_with_current(
            current_user.id,
            payload.current_password,
            payload.new_password,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return MessageResponse(message="Password updated")


@app.get("/auth/settings", response_model=UserSettingsResponse)
def auth_get_settings(
    current_user: AuthUser = Depends(get_current_user),
) -> UserSettingsResponse:
    return _to_settings_response(current_user.id)


@app.patch("/auth/settings", response_model=UserSettingsResponse)
def auth_update_settings(
    payload: UserSettingsUpdateRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> UserSettingsResponse:
    try:
        updated = auth_db.update_user_settings(
            current_user.id,
            theme=payload.theme,
            default_include_reasoning=payload.default_include_reasoning,
            default_answer_style=payload.default_answer_style,
            analysis_depth=payload.analysis_depth,
            llm_temperature_chat=payload.llm_temperature_chat,
            llm_temperature_tool=payload.llm_temperature_tool,
            llm_max_tokens_default=payload.llm_max_tokens_default,
            llm_max_tokens_reasoning=payload.llm_max_tokens_reasoning,
            backend_query_timeout_sec=payload.backend_query_timeout_sec,
            agent_max_steps=payload.agent_max_steps,
            agent_step_timeout_sec=payload.agent_step_timeout_sec,
            agent_inner_recursion_limit=payload.agent_inner_recursion_limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return UserSettingsResponse(
        theme=updated.theme,
        default_include_reasoning=updated.default_include_reasoning,
        default_answer_style=updated.default_answer_style,
        analysis_depth=updated.analysis_depth,
        llm_temperature_chat=updated.llm_temperature_chat,
        llm_temperature_tool=updated.llm_temperature_tool,
        llm_max_tokens_default=updated.llm_max_tokens_default,
        llm_max_tokens_reasoning=updated.llm_max_tokens_reasoning,
        backend_query_timeout_sec=updated.backend_query_timeout_sec,
        agent_max_steps=updated.agent_max_steps,
        agent_step_timeout_sec=updated.agent_step_timeout_sec,
        agent_inner_recursion_limit=updated.agent_inner_recursion_limit,
    )


@app.get("/db-connections", response_model=list[DBConnectionResponse])
def list_db_connections(
    current_user: AuthUser = Depends(get_current_user),
) -> list[DBConnectionResponse]:
    rows = db_connections_service.list_connections(current_user.id)
    return [_to_db_connection_response(row) for row in rows]


@app.post("/db-connections", response_model=DBConnectionResponse, status_code=201)
def create_db_connection(
    payload: DBConnectionCreateRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> DBConnectionResponse:
    created = db_connections_service.create_connection(
        current_user.id,
        name=payload.name,
        db_type=payload.db_type,
        host=payload.host,
        port=payload.port,
        database=payload.database,
        username=payload.username,
        password=payload.password,
        options_json=payload.options_json,
    )
    return _to_db_connection_response(created)


@app.get("/db-connections/{connection_id}", response_model=DBConnectionResponse)
def get_db_connection(
    connection_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> DBConnectionResponse:
    row = db_connections_service.get_connection(current_user.id, connection_id)
    return _to_db_connection_response(row)


@app.patch("/db-connections/{connection_id}", response_model=DBConnectionResponse)
def update_db_connection(
    connection_id: str,
    payload: DBConnectionUpdateRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> DBConnectionResponse:
    updated = db_connections_service.update_connection(
        current_user.id,
        connection_id,
        name=payload.name,
        db_type=payload.db_type,
        host=payload.host,
        port=payload.port,
        database=payload.database,
        username=payload.username,
        password=payload.password,
        clear_password=payload.clear_password,
        options_json=payload.options_json,
        options_json_set="options_json" in payload.model_fields_set,
    )
    return _to_db_connection_response(updated)


@app.delete("/db-connections/{connection_id}", response_model=MessageResponse)
def delete_db_connection(
    connection_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> MessageResponse:
    db_connections_service.delete_connection(current_user.id, connection_id)
    return MessageResponse(message="DB connection deleted")


@app.post("/db-connections/{connection_id}/test", response_model=DBConnectionTestResponse)
def test_db_connection(
    connection_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> DBConnectionTestResponse:
    tested = db_connections_service.test_connection(current_user.id, connection_id)
    if tested.last_test_at is None or tested.last_test_ok is None:
        raise HTTPException(
            status_code=500,
            detail="Connection test state was not persisted.",
        )
    return DBConnectionTestResponse(
        ok=tested.last_test_ok,
        checked_at=tested.last_test_at,
        last_test_at=tested.last_test_at,
        last_test_ok=tested.last_test_ok,
        error=tested.last_error,
    )


@app.get("/db-connections/{connection_id}/schemas", response_model=list[DBConnectionSchemaResponse])
def list_db_connection_schemas(
    connection_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> list[DBConnectionSchemaResponse]:
    try:
        items = db_runtime_service.list_schemas(
            user_id=current_user.id,
            connection_id=connection_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [
        DBConnectionSchemaResponse(
            name=item.name,
            display_name=item.display_name,
        )
        for item in items
    ]


@app.get("/db-connections/{connection_id}/tables", response_model=list[DBConnectionTableResponse])
def list_db_connection_tables(
    connection_id: str,
    schema: str,
    current_user: AuthUser = Depends(get_current_user),
) -> list[DBConnectionTableResponse]:
    try:
        items = db_runtime_service.list_tables(
            user_id=current_user.id,
            connection_id=connection_id,
            schema=schema,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [
        DBConnectionTableResponse(
            schema=item.schema,
            name=item.name,
            table_type=item.table_type,
            qualified_name=item.qualified_name,
        )
        for item in items
    ]


@app.get("/admin/users", response_model=list[AuthUserResponse])
def admin_list_users(
    _: AuthUser = Depends(get_admin_user),
) -> list[AuthUserResponse]:
    return [_to_user_response(user) for user in auth_db.list_users()]


@app.post("/admin/users", response_model=AuthUserResponse)
def admin_create_user(
    payload: AdminCreateUserRequest,
    _: AuthUser = Depends(get_admin_user),
) -> AuthUserResponse:
    try:
        created = auth_db.create_user(
            payload.username,
            payload.password,
            is_admin=payload.is_admin,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=409, detail="Username already exists")
    return _to_user_response(created)


@app.patch("/admin/users/{user_id}", response_model=AuthUserResponse)
def admin_update_user(
    user_id: int,
    payload: AdminUpdateUserRequest,
    current_admin: AuthUser = Depends(get_admin_user),
) -> AuthUserResponse:
    if payload.is_admin is False and user_id == current_admin.id:
        raise HTTPException(
            status_code=400,
            detail="Нельзя снять роль администратора у текущего пользователя",
        )

    if payload.password is not None:
        try:
            updated = auth_db.set_user_password(user_id, payload.password)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not updated:
            raise HTTPException(status_code=404, detail="User not found")

    if payload.is_admin is not None:
        try:
            updated = auth_db.set_user_admin(user_id, payload.is_admin)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not updated:
            raise HTTPException(status_code=404, detail="User not found")

    user = auth_db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return _to_user_response(user)


@app.delete("/admin/users/{user_id}", response_model=MessageResponse)
def admin_delete_user(
    user_id: int,
    current_admin: AuthUser = Depends(get_admin_user),
) -> MessageResponse:
    if user_id == current_admin.id:
        raise HTTPException(
            status_code=400,
            detail="Нельзя удалить текущего пользователя",
        )
    try:
        deleted = auth_db.delete_user(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")
    return MessageResponse(message="User deleted")


@app.post("/auth/logout", response_model=MessageResponse)
def auth_logout(token: str = Depends(_require_token)) -> MessageResponse:
    auth_db.revoke_token(token)
    return MessageResponse(message="Logged out")


@app.get("/sessions", response_model=list[SessionSummaryResponse])
def list_sessions(
    current_user: AuthUser = Depends(get_current_user),
) -> list[SessionSummaryResponse]:
    rows = auth_db.list_sessions(current_user.id)
    return [SessionSummaryResponse(**row) for row in rows]


@app.post("/sessions", response_model=CreateSessionResponse)
def create_session(
    enable_auto_title: bool = False,
    current_user: AuthUser = Depends(get_current_user),
) -> CreateSessionResponse:
    state = store.create_session()
    auth_db.register_session(
        state.session_id,
        current_user.id,
        allow_auto_title=enable_auto_title,
    )
    return CreateSessionResponse(session_id=state.session_id)


@app.delete("/sessions/{session_id}", response_model=MessageResponse)
def delete_session(
    session_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> MessageResponse:
    if not auth_db.delete_session(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    store.delete_session(session_id)
    return MessageResponse(message="Session deleted")


@app.patch("/sessions/{session_id}/title", response_model=SessionSummaryResponse)
def update_session_title(
    session_id: str,
    payload: SessionTitleUpdateRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> SessionSummaryResponse:
    if not auth_db.is_session_owner(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")

    ok = auth_db.set_session_title(session_id, current_user.id, payload.title)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid title")

    meta = auth_db.get_session_metadata(session_id, current_user.id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionSummaryResponse(**meta)


@app.post("/sessions/{session_id}/title/generate", response_model=SessionSummaryResponse)
def generate_session_title(
    session_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> SessionSummaryResponse:
    state = _load_owned_session(session_id, current_user)
    meta = auth_db.get_session_metadata(session_id, current_user.id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Session not found")

    is_custom_title = auth_db.is_session_title_custom(session_id, current_user.id)
    if is_custom_title is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if is_custom_title:
        return SessionSummaryResponse(**meta)

    user_queries = _collect_user_queries_for_title(state)
    if not user_queries:
        return SessionSummaryResponse(**meta)

    df = _active_session_dataframe(state, session_id)
    dataset_hint = _dataset_hint_for_title(state, df)
    title_context_query = "\n".join(user_queries[-6:])

    trace_context = build_trace_context(
        session_id=session_id,
        user_id=current_user.id,
        username=current_user.username,
        request_kind="title_generate",
        use_history=False,
        include_reasoning=False,
    )
    generated_title: str | None = None
    try:
        with query_trace_context(
            session_id=session_id,
            user_id=current_user.id,
            username=current_user.username,
            request_kind="title_generate",
            use_history=False,
            include_reasoning=False,
            query=title_context_query,
        ):
            generated_title = runner.generate_chat_title(
                dataset_name=dataset_hint,
                user_queries=user_queries,
                trace_context=trace_context,
            )
    except Exception:
        generated_title = None

    if generated_title:
        auth_db.set_session_title(session_id, current_user.id, generated_title)
        updated = auth_db.get_session_metadata(session_id, current_user.id)
        if updated is not None:
            return SessionSummaryResponse(**updated)
    return SessionSummaryResponse(**meta)


@app.get("/sessions/{session_id}", response_model=SessionStateResponse)
def get_session(
    session_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> SessionStateResponse:
    state = _load_owned_session(session_id, current_user)
    meta = auth_db.get_session_metadata(session_id, current_user.id)
    title = "Новый чат"
    if meta is not None:
        title = str(meta.get("title") or "Новый чат")
    return SessionStateResponse(
        session_id=state.session_id,
        title=title,
        chat_history=state.chat_history,
        artifacts=state.artifacts,
        has_dataset=bool(state.df_path),
        dataset_name=state.dataset_name,
        source_type=state.source_type,
        source_ref_id=state.source_ref_id,
        source_label=state.source_label,
        source_mode=state.source_mode,
    )


@app.post(
    "/sessions/{session_id}/source/db-connection",
    response_model=SessionSourceStateResponse,
)
def bind_session_db_connection_source(
    session_id: str,
    payload: SessionBindDBConnectionSourceRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> SessionSourceStateResponse:
    state = _load_owned_session(session_id, current_user)
    connection = db_connections_service.get_connection(
        current_user.id,
        payload.connection_id,
    )
    store.bind_db_connection_source(
        session_id,
        connection_id=connection.id,
        label=connection.name,
        source_mode=payload.source_mode,
    )
    refreshed = _load_owned_session(session_id, current_user)
    return _to_session_source_response(refreshed)


@app.post(
    "/sessions/{session_id}/source/clear",
    response_model=SessionSourceStateResponse,
)
def clear_session_source(
    session_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> SessionSourceStateResponse:
    _load_owned_session(session_id, current_user)
    store.set_source(
        session_id,
        source_type=None,
        source_ref_id=None,
        source_label=None,
        source_mode=None,
    )
    refreshed = _load_owned_session(session_id, current_user)
    return _to_session_source_response(refreshed)


@app.post(
    "/sessions/{session_id}/source/csv",
    response_model=SessionSourceStateResponse,
)
def bind_session_csv_source(
    session_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> SessionSourceStateResponse:
    state = _load_owned_session(session_id, current_user)
    if not state.df_path or not state.dataset_name:
        raise HTTPException(
            status_code=400,
            detail="No CSV dataset is attached to this session",
        )
    store.bind_csv_source(session_id, filename=state.dataset_name)
    refreshed = _load_owned_session(session_id, current_user)
    return _to_session_source_response(refreshed)


@app.post("/sessions/{session_id}/data", response_model=UploadResponse)
async def upload_data(
    session_id: str,
    file: UploadFile = File(...),
    current_user: AuthUser = Depends(get_current_user),
) -> UploadResponse:
    _load_owned_session(session_id, current_user)

    content = await file.read()
    max_bytes = settings.max_dataset_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="Dataset exceeds size limit")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}")

    store.save_dataframe(session_id, df)
    store.set_dataset_name(session_id, file.filename)
    store.bind_csv_source(session_id, filename=file.filename)
    auth_db.mark_session_has_dataset(session_id, True)
    return UploadResponse(session_id=session_id, rows=len(df), columns=len(df.columns))


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
    tool_collector: ToolCollector,
    use_history: bool,
    duration_ms: int,
    has_dataset: bool,
) -> str | None:
    normalized_route = (route or "").strip().lower()
    if normalized_route not in {"chat", "analysis"}:
        normalized_route = "analysis" if has_dataset else "chat"

    unique_tools: list[str] = []
    seen_tools: set[str] = set()
    for name in tool_collector.tool_names:
        normalized = str(name).strip()
        if not normalized or normalized in seen_tools:
            continue
        seen_tools.add(normalized)
        unique_tools.append(normalized)

    lines: list[str] = [
        "### Reason-Action Trace",
        f"- Route: `{normalized_route}`",
        f"- Dataset attached: `{'yes' if has_dataset else 'no'}`",
        f"- Use history: `{'yes' if use_history else 'no'}`",
        f"- Tool calls: `{tool_collector.tool_calls}`",
        f"- Duration: `{duration_ms} ms`",
    ]
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

    cleaned_model_reasoning = (response_reasoning or "").strip()
    if cleaned_model_reasoning:
        if len(cleaned_model_reasoning) > 4000:
            cleaned_model_reasoning = f"{cleaned_model_reasoning[:4000]}..."
        lines.append("")
        lines.append("### Model reasoning")
        lines.append(cleaned_model_reasoning)
    elif not response_text.strip():
        lines.append("")
        lines.append("### Notes")
        lines.append("Fallback path used: model returned empty text.")

    reasoning = "\n".join(lines).strip()
    return reasoning or None


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


async def _execute_query(
    session_id: str,
    payload: QueryRequest,
    *,
    persist: bool,
    current_user: AuthUser,
    callbacks: list[object] | None = None,
) -> QueryResponse:
    state = _load_owned_session(session_id, current_user)
    df = _active_session_dataframe(state, session_id)
    session_source = _session_source_payload(state)
    session_db_connection_id = _session_db_connection_id(state)
    has_active_source = df is not None or session_db_connection_id is not None

    text_collector = LLMTextCollector()
    tool_collector = ToolCollector()
    active_callbacks = list(callbacks or [])
    active_callbacks.extend([text_collector, tool_collector])
    started_at = time.perf_counter()
    auth_db.touch_session(session_id)
    trace_context = build_trace_context(
        session_id=session_id,
        user_id=current_user.id,
        username=current_user.username,
        request_kind="query" if persist else "evaluate",
        use_history=payload.use_history,
        include_reasoning=payload.include_reasoning,
        db_connection_id=session_db_connection_id,
    )
    runtime_settings = _effective_runtime_settings(
        current_user.id, analysis_depth_override=payload.analysis_depth
    )
    runtime_runner = AgentRunner(
        runtime_settings,
        db_runtime_service=db_runtime_service,
    )

    try:
        with query_trace_context(
            session_id=session_id,
            user_id=current_user.id,
            username=current_user.username,
            request_kind="query" if persist else "evaluate",
            use_history=payload.use_history,
            include_reasoning=payload.include_reasoning,
            query=payload.query,
            db_connection_id=session_db_connection_id,
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
                )
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
                store, auth_db, session_id, current_user.id,
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
                store, auth_db, session_id, current_user.id,
                payload.query, fallback.text, reasoning=fallback.reasoning,
            )
        return fallback

    artifacts = [serialize_artifact(a) for a in response.artifacts]
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
        store.add_chat_message(session_id, "user", payload.query)
        store.add_chat_message(
            session_id,
            "ai",
            response.final_text,
            artifacts=artifacts,
            reasoning=effective_reasoning,
        )
        store.add_artifacts(session_id, response.artifacts)
        auth_db.update_session_after_reply(
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


@app.post("/sessions/{session_id}/query", response_model=QueryResponse)
async def query(
    session_id: str,
    payload: QueryRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> QueryResponse:
    return await _execute_query(
        session_id,
        payload,
        persist=True,
        current_user=current_user,
    )


@app.post("/sessions/{session_id}/evaluate", response_model=QueryResponse)
async def evaluate(
    session_id: str,
    payload: QueryRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> QueryResponse:
    return await _execute_query(
        session_id,
        payload,
        persist=False,
        current_user=current_user,
    )


@app.post("/sessions/{session_id}/query/stream")
async def query_stream(
    session_id: str,
    payload: QueryRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> StreamingResponse:
    state = _load_owned_session(session_id, current_user)
    df = _active_session_dataframe(state, session_id)
    session_source = _session_source_payload(state)
    session_db_connection_id = _session_db_connection_id(state)
    has_active_source = df is not None or session_db_connection_id is not None

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    agent_finished = asyncio.Event()

    text_collector = LLMTextCollector()
    tool_collector = ToolCollector()
    progress_collector = AgentProgressCollector()
    phase_collector = PhaseCollector()
    token_collector = TokenStreamCallbackHandler(queue, loop)
    phase_token_handler = PhaseTokenStreamHandler(queue, loop)
    callbacks = [token_collector, text_collector, tool_collector, progress_collector, phase_collector, phase_token_handler]
    started_at = time.perf_counter()
    auth_db.touch_session(session_id)
    trace_context = build_trace_context(
        session_id=session_id,
        user_id=current_user.id,
        username=current_user.username,
        request_kind="stream",
        use_history=payload.use_history,
        include_reasoning=payload.include_reasoning,
        db_connection_id=session_db_connection_id,
    )
    runtime_settings = _effective_runtime_settings(
        current_user.id, analysis_depth_override=payload.analysis_depth
    )
    runtime_runner = AgentRunner(
        runtime_settings,
        db_runtime_service=db_runtime_service,
    )

    async def run_agent() -> None:
        try:
            with query_trace_context(
                session_id=session_id,
                user_id=current_user.id,
                username=current_user.username,
                request_kind="stream",
                use_history=payload.use_history,
                include_reasoning=payload.include_reasoning,
                query=payload.query,
                db_connection_id=session_db_connection_id,
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
                    )

            artifacts = [serialize_artifact(a) for a in response.artifacts]
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
            store.add_chat_message(session_id, "user", payload.query)
            store.add_chat_message(
                session_id,
                "ai",
                response.final_text,
                artifacts=artifacts,
                reasoning=effective_reasoning,
            )
            store.add_artifacts(session_id, response.artifacts)
            auth_db.update_session_after_reply(
                session_id, response.final_text, auto_title=None
            )
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
            await queue.put(("final", final_payload))
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
                store, auth_db, session_id, current_user.id,
                payload.query, fallback_payload.text,
                reasoning=fallback_payload.reasoning,
            )
            await queue.put(("final", fallback_payload.model_dump()))
        except Exception:
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
                store, auth_db, session_id, current_user.id,
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
        while True:
            emitted_any = False

            while emitted_phase_events < len(phase_collector.events):
                current = phase_collector.events[emitted_phase_events]
                emitted_phase_events += 1
                await queue.put(("phase", current))
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
                if (
                    emitted_tool_events >= len(tool_collector.events)
                    and emitted_progress_events >= len(progress_collector.events)
                    and emitted_phase_events >= len(phase_collector.events)
                ):
                    break

            if not emitted_any:
                await asyncio.sleep(0.15)

    async def event_generator():
        yield _sse_event("start", {"session_id": session_id})
        if payload.include_reasoning:
            yield _sse_event(
                "reasoning",
                "### Думаю\nПолучил задачу. Формирую план анализа и последовательность шагов.",
            )
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


def _sse_event(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"
