from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from backend.auth.auth_db import AuthUser, AuthDB
from backend.sessions.session_store import SessionStore, SessionState
from backend.api.deps import get_current_user
from backend.api.models import (
    CreateSessionResponse,
    MessageResponse,
    SessionStateResponse,
    SessionSummaryResponse,
    SessionTitleUpdateRequest,
)

router = APIRouter(tags=["Сессии"])

# Singletons set during app startup
_auth_db: AuthDB = None  # type: ignore
_store: SessionStore = None  # type: ignore
_runner = None  # type: ignore
_build_trace_context_fn = None  # type: ignore
_query_trace_context_fn = None  # type: ignore


def setup(
    auth_db: AuthDB,
    store: SessionStore,
    runner,
    build_trace_context_fn,
    query_trace_context_fn,
) -> None:
    global _auth_db, _store, _runner, _build_trace_context_fn, _query_trace_context_fn
    _auth_db = auth_db
    _store = store
    _runner = runner
    _build_trace_context_fn = build_trace_context_fn
    _query_trace_context_fn = query_trace_context_fn


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


@router.get("/sessions", response_model=list[SessionSummaryResponse])
def list_sessions(
    current_user: AuthUser = Depends(get_current_user),
) -> list[SessionSummaryResponse]:
    rows = _auth_db.list_sessions(current_user.id)
    return [SessionSummaryResponse(**row) for row in rows]


@router.post("/sessions", response_model=CreateSessionResponse)
def create_session(
    enable_auto_title: bool = False,
    current_user: AuthUser = Depends(get_current_user),
) -> CreateSessionResponse:
    state = _store.create_session()
    _auth_db.register_session(
        state.session_id,
        current_user.id,
        allow_auto_title=enable_auto_title,
    )
    return CreateSessionResponse(session_id=state.session_id)


@router.delete("/sessions/{session_id}", response_model=MessageResponse)
def delete_session(
    session_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> MessageResponse:
    if not _auth_db.delete_session(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    _store.delete_session(session_id)
    return MessageResponse(message="Session deleted")


@router.patch("/sessions/{session_id}/title", response_model=SessionSummaryResponse)
def update_session_title(
    session_id: str,
    payload: SessionTitleUpdateRequest,
    current_user: AuthUser = Depends(get_current_user),
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
    current_user: AuthUser = Depends(get_current_user),
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
            generated_title = _runner.generate_chat_title(
                dataset_name=dataset_hint,
                user_queries=user_queries,
                trace_context=trace_context,
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
    current_user: AuthUser = Depends(get_current_user),
) -> SessionStateResponse:
    state = _load_owned_session(session_id, current_user)
    meta = _auth_db.get_session_metadata(session_id, current_user.id)
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
        selected_skill_ids=list(state.selected_skill_ids or []),
    )


@router.get("/sessions/{session_id}/notebook")
def get_session_notebook(
    session_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> PlainTextResponse:
    """Return the session notebook markdown (persisted by sandbox)."""
    if not _auth_db.is_session_owner(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    notebook_path = _store._session_dir(session_id) / "notebook.md"
    if not notebook_path.exists():
        return PlainTextResponse("", media_type="text/markdown")
    content = notebook_path.read_text(encoding="utf-8")
    return PlainTextResponse(content, media_type="text/markdown")


@router.get("/sessions/{session_id}/notebook/cells")
def get_session_notebook_cells(
    session_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> list[dict]:
    """Return the session notebook as a structured list of cells (JSON)."""
    from backend.tools.sandbox_manager import SandboxManager
    if not _auth_db.is_session_owner(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    sandbox = SandboxManager.get_instance().get(session_id)
    if sandbox is None:
        return []
    return sandbox.get_notebook_cells()

