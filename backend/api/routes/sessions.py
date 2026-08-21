from __future__ import annotations

import secrets
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

from backend.agent.services.chat_title import ChatTitleRequest, ChatTitleService
from backend.api.deps import get_current_user
from backend.api.models import (
    CreateSessionResponse,
    MessageResponse,
    SessionSourceResponse,
    SessionStateResponse,
    SessionSummaryResponse,
    SessionTitleUpdateRequest,
)
from backend.auth.auth_db import AuthDB, AuthUser
from backend.core.config import settings
from backend.data_access.semantic_catalog_service import SemanticCatalogService
from backend.notebook.manifest_store import ManifestStore
from backend.sessions.session_store import SessionState, SessionStore

router = APIRouter(tags=["Сессии"])

# Singletons set during app startup
_auth_db: AuthDB = None  # type: ignore
_store: SessionStore = None  # type: ignore
_title_service: ChatTitleService = None  # type: ignore
_build_trace_context_fn = None  # type: ignore
_query_trace_context_fn = None  # type: ignore
_manifest_store: ManifestStore = None  # type: ignore
_semantic_catalog_service: SemanticCatalogService | None = None


def setup(
    auth_db: AuthDB,
    store: SessionStore,
    title_service: ChatTitleService,
    build_trace_context_fn,
    query_trace_context_fn,
    manifest_store: ManifestStore | None = None,
    semantic_catalog_service: SemanticCatalogService | None = None,
) -> None:
    global _auth_db, _store, _title_service, _build_trace_context_fn, _query_trace_context_fn
    global _manifest_store, _semantic_catalog_service
    _auth_db = auth_db
    _store = store
    _title_service = title_service
    _manifest_store = manifest_store
    _semantic_catalog_service = semantic_catalog_service
    _build_trace_context_fn = build_trace_context_fn
    _query_trace_context_fn = query_trace_context_fn


def _clear_session_semantic_catalog(
    session_id: str,
    user_id: int,
    *,
    preserve_shared_csv: bool,
) -> None:
    if _semantic_catalog_service is None:
        return
    state = _store.load_session(session_id)
    if state is None or _session_source_type(state) not in {"csv", "planfact"}:
        return
    if preserve_shared_csv and _session_source_type(state) == "csv":
        for row in _auth_db.list_sessions(user_id):
            other_id = str(row.get("session_id") or "")
            if not other_id or other_id == session_id:
                continue
            other = _store.load_session(other_id)
            if (
                other is not None
                and _session_source_type(other) == "csv"
                and other.source_ref_id == state.source_ref_id
            ):
                return
    _semantic_catalog_service.clear_for_session(session_id=session_id, user_id=user_id)


def _load_owned_session(session_id: str, current_user: AuthUser) -> SessionState:
    if not _auth_db.is_session_owner(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    state = _store.load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


def _session_source_type(state: SessionState) -> str:
    return str(state.source_type or "").strip().lower()


def _active_session_dataframe(
    state: SessionState, session_id: str
) -> pd.DataFrame | None:
    if _session_source_type(state) != "csv":
        return None
    return _store.get_dataframe(session_id)


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


def _strip_thinking_from_history(
    chat_history: list[dict],
) -> list[dict]:
    """Remove reasoning_steps and pre_reasoning from AI messages.

    Called when settings.llm_show_think=False so that think blocks
    are not exposed to the frontend on page refresh.
    Data is preserved in storage — only the API response is filtered.
    """
    result = []
    for message in chat_history:
        if str(message.get("role", "")).strip().lower() not in ("ai", "assistant"):
            result.append(message)
            continue
        filtered = {k: v for k, v in message.items() if k != "reasoning_steps"}
        tools = filtered.get("tools")
        if tools:
            filtered["tools"] = [
                {k: v for k, v in tool.items() if k != "pre_reasoning"}
                for tool in tools
            ]
        result.append(filtered)
    return result


@router.get("/sessions", response_model=list[SessionSummaryResponse])
def list_sessions(
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[SessionSummaryResponse]:
    rows = _auth_db.list_sessions(current_user.id)
    return [SessionSummaryResponse(**row) for row in rows]


@router.post("/sessions", response_model=CreateSessionResponse)
def create_session(
    current_user: Annotated[AuthUser, Depends(get_current_user)],
    enable_auto_title: bool = False,
) -> CreateSessionResponse:
    session_id = secrets.token_hex(16)
    _auth_db.register_session(
        session_id,
        current_user.id,
        allow_auto_title=enable_auto_title,
    )
    try:
        state = _store.create_session(session_id=session_id)
    except Exception:
        _auth_db.delete_session(session_id, current_user.id)
        raise
    return CreateSessionResponse(session_id=state.session_id)


@router.delete("/sessions", response_model=MessageResponse)
def delete_all_sessions(
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> MessageResponse:
    """Delete every session that belongs to the current user."""
    session_ids = [str(row["session_id"]) for row in _auth_db.list_sessions(current_user.id)]
    for sid in session_ids:
        _clear_session_semantic_catalog(sid, current_user.id, preserve_shared_csv=False)
    session_ids = _auth_db.delete_all_sessions(current_user.id)
    for sid in session_ids:
        _store.delete_session(sid)
    return MessageResponse(message=f"Deleted {len(session_ids)} session(s)")


@router.delete("/sessions/{session_id}", response_model=MessageResponse)
def delete_session(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> MessageResponse:
    if not _auth_db.is_session_owner(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    _clear_session_semantic_catalog(session_id, current_user.id, preserve_shared_csv=True)
    if not _auth_db.delete_session(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    _store.delete_session(session_id)
    return MessageResponse(message="Session deleted")


@router.patch("/sessions/{session_id}/title", response_model=SessionSummaryResponse)
def update_session_title(
    session_id: str,
    payload: SessionTitleUpdateRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SessionSummaryResponse:
    if not _auth_db.is_session_owner(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")

    ok = _auth_db.set_session_title(session_id, current_user.id, payload.title)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid title")

    meta = _auth_db.get_session_metadata(session_id, current_user.id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionSummaryResponse(**meta)


@router.post("/sessions/{session_id}/title/generate", response_model=SessionSummaryResponse)
def generate_session_title(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SessionSummaryResponse:
    state = _load_owned_session(session_id, current_user)
    meta = _auth_db.get_session_metadata(session_id, current_user.id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Session not found")

    is_custom_title = _auth_db.is_session_title_custom(session_id, current_user.id)
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

    trace_context = _build_trace_context_fn(
        session_id=session_id,
        user_id=current_user.id,
        username=current_user.username,
        request_kind="title_generate",
        use_history=False,
        include_reasoning=False,
    )
    generated_title: str | None = None
    try:
        with _query_trace_context_fn(
            session_id=session_id,
            user_id=current_user.id,
            username=current_user.username,
            request_kind="title_generate",
            use_history=False,
            include_reasoning=False,
            query=title_context_query,
        ):
            generated_title = _title_service.generate(
                ChatTitleRequest(
                    dataset_name=dataset_hint,
                    user_queries=user_queries,
                    trace_context=trace_context,
                )
            )
    except Exception:
        generated_title = None

    if generated_title:
        _auth_db.set_session_title(session_id, current_user.id, generated_title)
        updated = _auth_db.get_session_metadata(session_id, current_user.id)
        if updated is not None:
            return SessionSummaryResponse(**updated)
    return SessionSummaryResponse(**meta)


@router.get("/sessions/{session_id}", response_model=SessionStateResponse)
def get_session(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SessionStateResponse:
    state = _load_owned_session(session_id, current_user)
    meta = _auth_db.get_session_metadata(session_id, current_user.id)
    title = "Новый чат"
    if meta is not None:
        title = str(meta.get("title") or "Новый чат")
    # Build multi-source list from manifest (if available).
    sources: list[SessionSourceResponse] = []
    if _manifest_store is not None:
        manifest = _manifest_store.load(state.session_id)
        sources = [
            SessionSourceResponse(
                alias=s.alias,
                source_type=s.source_type,
                display_name=s.display_name,
                variable_name=s.variable_name,
                file_name=s.file_name,
                connection_id=s.connection_id,
                connection_name=s.connection_name,
                bound_at=s.bound_at,
                csv_table_names=list(s.csv_table_names or []),
                schema_hint=s.schema_hint,
            )
            for s in manifest.sources
        ]

    chat_history = state.chat_history
    if not settings.llm_show_think:
        chat_history = _strip_thinking_from_history(chat_history)

    return SessionStateResponse(
        session_id=state.session_id,
        title=title,
        chat_history=chat_history,
        artifacts=state.artifacts,
        has_dataset=bool(state.df_path),
        dataset_name=state.dataset_name,
        source_type=state.source_type,
        source_ref_id=state.source_ref_id,
        source_label=state.source_label,
        source_mode=state.source_mode,
        selected_skill_ids=list(state.selected_skill_ids or []),
        sources=sources,
        session_memory=state.session_memory or "",
        context_usage=state.context_usage,
    )


@router.delete("/sessions/{session_id}/messages/last", response_model=MessageResponse)
def delete_last_session_messages(
    session_id: str,
    message_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> MessageResponse:
    """Delete the message identified by *message_id* and all subsequent messages.

    Used to remove a user+assistant exchange before regeneration so the
    history stays consistent on both client and server.
    """
    if not _auth_db.is_session_owner(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    removed = _store.delete_messages_from_id(session_id, message_id)
    return MessageResponse(message=f"Removed {removed} message(s)")


@router.get("/sessions/{session_id}/notebook")
def get_session_notebook(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> PlainTextResponse:
    """Return the session notebook markdown (persisted by sandbox)."""
    if not _auth_db.is_session_owner(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    notebook_path = _store._session_dir(session_id) / "notebook.md"  # noqa: SLF001
    if not notebook_path.exists():
        return PlainTextResponse("", media_type="text/markdown")
    content = notebook_path.read_text(encoding="utf-8")
    return PlainTextResponse(content, media_type="text/markdown")


@router.get("/sessions/{session_id}/notebook/cells")
def get_session_notebook_cells(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[dict]:
    """Return the session notebook as a structured list of cells (JSON)."""
    from backend.tools.sandbox_manager import SandboxManager
    if not _auth_db.is_session_owner(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    sandbox = SandboxManager.get_instance().get(session_id)
    if sandbox is None:
        return []
    return sandbox.get_notebook_cells()
