from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import get_current_user
from backend.api.models import (
    SessionBindDBConnectionSourceRequest,
    SessionSourceResponse,
    SessionSourceStateResponse,
    SourceDescriptorResponse,
)
from backend.auth.auth_db import AuthDB, AuthUser
from backend.core.config import settings
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.notebook.cell_builder import build_source_binding_cell
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.orchestrator import CellOp, NotebookEdit, NotebookOrchestrator
from backend.notebook.session_source import (
    SessionSource,
    alias_to_variable_name,
    make_source_alias,
)
from backend.sessions.session_store import SessionState, SessionStore

router = APIRouter(tags=["Источники"])

# Singletons set during app startup
_auth_db: AuthDB = None  # type: ignore
_store: SessionStore = None  # type: ignore
_db_connections_service = None  # type: ignore
_integration_source_descriptors_fn = None  # type: ignore
_csv_runtime: CSVSessionRuntime = None  # type: ignore
_manifest_store: ManifestStore = None  # type: ignore
_orchestrator: NotebookOrchestrator = None  # type: ignore


def setup(
    auth_db: AuthDB,
    store: SessionStore,
    db_connections_service,
    integration_source_descriptors_fn,
    csv_runtime: CSVSessionRuntime,
    manifest_store: ManifestStore,
    notebook_orchestrator: NotebookOrchestrator,
) -> None:
    global _auth_db, _store, _db_connections_service
    global _integration_source_descriptors_fn, _csv_runtime
    global _manifest_store, _orchestrator
    _auth_db = auth_db
    _store = store
    _db_connections_service = db_connections_service
    _integration_source_descriptors_fn = integration_source_descriptors_fn
    _csv_runtime = csv_runtime
    _manifest_store = manifest_store
    _orchestrator = notebook_orchestrator


def _load_owned_session(session_id: str, current_user: AuthUser) -> SessionState:
    if not _auth_db.is_session_owner(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    state = _store.load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


def _session_source_type(state: SessionState) -> str:
    return str(state.source_type or "").strip().lower()


def _ensure_csv_runtime_state(session_id: str, state: SessionState) -> SessionState:
    if _session_source_type(state) != "csv":
        return state

    if state.csv_loaded and state.csv_session_id:
        return state

    if not state.df_path:
        raise HTTPException(
            status_code=400,
            detail="CSV dataset is not attached to this session",
        )

    df = _store.get_dataframe(session_id)
    if df is None:
        raise HTTPException(
            status_code=400,
            detail="Failed to load CSV dataframe for this session",
        )

    try:
        csv_info = _csv_runtime.register_dataframe(
            session_id=session_id,
            table_name=state.dataset_name or "uploaded.csv",
            df=df,
            ttl_seconds=settings.csv_session_ttl_sec,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize CSV runtime: {exc}",
        ) from exc

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


def _to_session_source_response(state: SessionState) -> SessionSourceStateResponse:
    return SessionSourceStateResponse(
        source_type=state.source_type,
        source_ref_id=state.source_ref_id,
        source_label=state.source_label,
        source_mode=state.source_mode,
    )


# ── Legacy single-source endpoints (backward compat) ────────────────────────


@router.get("/sources", response_model=list[SourceDescriptorResponse])
def list_available_sources(
    current_user: Annotated[AuthUser, Depends(get_current_user)],
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
    current_user: Annotated[AuthUser, Depends(get_current_user)],
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

    # Also register in manifest.
    _add_source_to_manifest(
        session_id,
        source_type="db_connection",
        display_name=connection.name,
        connection_id=connection.id,
        connection_name=connection.name,
    )

    refreshed = _load_owned_session(session_id, current_user)
    return _to_session_source_response(refreshed)


@router.post(
    "/sessions/{session_id}/source/clear",
    response_model=SessionSourceStateResponse,
)
def clear_session_source(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
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
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SessionSourceStateResponse:
    state = _load_owned_session(session_id, current_user)
    if not state.df_path or not state.dataset_name:
        raise HTTPException(
            status_code=400,
            detail="No CSV dataset is attached to this session",
        )
    _store.bind_csv_source(session_id, filename=state.dataset_name)
    refreshed = _load_owned_session(session_id, current_user)
    refreshed = _ensure_csv_runtime_state(session_id, refreshed)
    return _to_session_source_response(refreshed)


# ── Multi-source endpoints ──────────────────────────────────────────────────


@router.get(
    "/sessions/{session_id}/sources",
    response_model=list[SessionSourceResponse],
)
def list_session_sources(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[SessionSourceResponse]:
    """List all sources bound to a session."""
    _load_owned_session(session_id, current_user)
    manifest = _manifest_store.load(session_id)
    return [
        SessionSourceResponse(
            alias=s.alias,
            source_type=s.source_type,
            display_name=s.display_name,
            variable_name=s.variable_name,
            file_name=s.file_name,
            connection_id=s.connection_id,
            connection_name=s.connection_name,
            bound_at=s.bound_at,
            schema_hint=s.schema_hint,
        )
        for s in manifest.sources
    ]


@router.delete(
    "/sessions/{session_id}/sources/{alias}",
    response_model=list[SessionSourceResponse],
)
def remove_session_source(
    session_id: str,
    alias: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[SessionSourceResponse]:
    """Remove a specific source from a session by alias."""
    _load_owned_session(session_id, current_user)
    manifest = _manifest_store.load(session_id)

    removed = manifest.remove_source(alias)
    if removed is None:
        raise HTTPException(status_code=404, detail=f"Source '{alias}' not found")

    _manifest_store.save(session_id, manifest)

    return list_session_sources(session_id, current_user)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _add_source_to_manifest(
    session_id: str,
    *,
    source_type: str,
    display_name: str,
    file_name: str | None = None,
    parquet_path: str | None = None,
    connection_id: str | None = None,
    connection_name: str | None = None,
    csv_session_id: str | None = None,
    csv_table_names: list[str] | None = None,
    csv_expires_at: int | None = None,
    schema_hint: dict[str, str] | None = None,
) -> SessionSource:
    """Add a source to the manifest and create a source_binding cell."""
    manifest = _manifest_store.load(session_id)

    existing_aliases = [s.alias for s in manifest.sources]
    alias = make_source_alias(display_name, source_type, existing_aliases)
    var_name = alias_to_variable_name(alias)

    source = SessionSource(
        alias=alias,
        source_type=source_type,
        display_name=display_name,
        variable_name=var_name,
        file_name=file_name,
        parquet_path=parquet_path,
        connection_id=connection_id,
        connection_name=connection_name,
        csv_session_id=csv_session_id,
        csv_table_names=csv_table_names or [],
        csv_expires_at=csv_expires_at,
        schema_hint=schema_hint or {},
    )
    manifest.add_source(source)
    _manifest_store.save(session_id, manifest)

    # Create source_binding notebook cell.
    if source_type == "csv":
        load_code = f'{var_name} = pd.read_parquet("{parquet_path or "data.parquet"}")'
    else:
        load_code = f'{var_name} = _restore_db_connection("{alias}")'

    cell = build_source_binding_cell(
        alias=alias,
        variable_name=var_name,
        source_type=source_type,
        display_name=display_name,
        load_code=load_code,
    )
    _orchestrator.apply(session_id, NotebookEdit(op=CellOp.INSERT, cell=cell))

    return source
