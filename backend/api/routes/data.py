from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from backend.auth.auth_db import AuthUser, AuthDB
from backend.sessions.session_store import SessionStore, SessionState
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.api.deps import get_current_user
from backend.api.models import UploadResponse
from backend.core.config import settings

router = APIRouter(tags=["Данные"])

# Singletons set during app startup
_auth_db: AuthDB = None  # type: ignore
_store: SessionStore = None  # type: ignore
_csv_runtime: CSVSessionRuntime = None  # type: ignore


def setup(auth_db: AuthDB, store: SessionStore, csv_runtime: CSVSessionRuntime) -> None:
    global _auth_db, _store, _csv_runtime
    _auth_db = auth_db
    _store = store
    _csv_runtime = csv_runtime


def _load_owned_session(session_id: str, current_user: AuthUser) -> SessionState:
    if not _auth_db.is_session_owner(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    state = _store.load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


@router.post("/sessions/{session_id}/data", response_model=UploadResponse)
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
        df, csv_info = _csv_runtime.register_csv_bytes(
            session_id=session_id,
            file_name=file.filename or "uploaded.csv",
            csv_bytes=content,
            ttl_seconds=settings.csv_session_ttl_sec,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}")

    _store.save_dataframe(session_id, df)
    _store.set_dataset_name(session_id, file.filename)
    _store.bind_csv_source(session_id, filename=file.filename)
    _store.set_csv_runtime_state(
        session_id,
        csv_loaded=True,
        csv_session_id=csv_info.session_id,
        csv_table_names=list(csv_info.table_names),
        csv_expires_at=csv_info.expires_at,
    )
    _auth_db.mark_session_has_dataset(session_id, True)
    return UploadResponse(
        session_id=session_id,
        rows=len(df),
        columns=len(df.columns),
    )
