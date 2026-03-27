from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from backend.auth.auth_db import AuthUser, AuthDB
from backend.sessions.session_store import SessionStore, SessionState
from backend.api.deps import get_current_user
from backend.api.models import (
    SessionBindDBConnectionSourceRequest,
    SessionSourceStateResponse,
    SourceDescriptorResponse,
)

router = APIRouter(tags=["Источники"])

# Singletons set during app startup
_auth_db: AuthDB = None  # type: ignore
_store: SessionStore = None  # type: ignore
_db_connections_service = None  # type: ignore
_integration_source_descriptors_fn = None  # type: ignore


def setup(
    auth_db: AuthDB,
    store: SessionStore,
    db_connections_service,
    integration_source_descriptors_fn,
) -> None:
    global _auth_db, _store, _db_connections_service, _integration_source_descriptors_fn
    _auth_db = auth_db
    _store = store
    _db_connections_service = db_connections_service
    _integration_source_descriptors_fn = integration_source_descriptors_fn


def _load_owned_session(session_id: str, current_user: AuthUser) -> SessionState:
    if not _auth_db.is_session_owner(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    state = _store.load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


def _to_session_source_response(state: SessionState) -> SessionSourceStateResponse:
    return SessionSourceStateResponse(
        source_type=state.source_type,
        source_ref_id=state.source_ref_id,
        source_label=state.source_label,
        source_mode=state.source_mode,
    )


@router.get("/sources", response_model=list[SourceDescriptorResponse])
def list_available_sources(
    current_user: AuthUser = Depends(get_current_user),
) -> list[SourceDescriptorResponse]:
    _ = current_user
    return [SourceDescriptorResponse(**item) for item in _integration_source_descriptors_fn()]


@router.post(
    "/sessions/{session_id}/source/db-connection",
    response_model=SessionSourceStateResponse,
)
def bind_session_db_connection_source(
    session_id: str,
    payload: SessionBindDBConnectionSourceRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> SessionSourceStateResponse:
    _load_owned_session(session_id, current_user)
    connection = _db_connections_service.get_connection(
        current_user.id,
        payload.connection_id,
    )
    _store.bind_db_connection_source(
        session_id,
        connection_id=connection.id,
        label=connection.name,
        source_mode=payload.source_mode,
    )
    refreshed = _load_owned_session(session_id, current_user)
    return _to_session_source_response(refreshed)


@router.post(
    "/sessions/{session_id}/source/clear",
    response_model=SessionSourceStateResponse,
)
def clear_session_source(
    session_id: str,
    current_user: AuthUser = Depends(get_current_user),
) -> SessionSourceStateResponse:
    _load_owned_session(session_id, current_user)
    _store.set_source(
        session_id,
        source_type=None,
        source_ref_id=None,
        source_label=None,
        source_mode=None,
    )
    refreshed = _load_owned_session(session_id, current_user)
    return _to_session_source_response(refreshed)


@router.post(
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
    _store.bind_csv_source(session_id, filename=state.dataset_name)
    refreshed = _load_owned_session(session_id, current_user)
    return _to_session_source_response(refreshed)


