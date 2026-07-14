from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.api.deps import get_current_user
from backend.api.models import (
    RagDocumentDeleteResponse,
    RagDocumentsResponse,
    RagDocumentUploadResponse,
    RagTrackStatusResponse,
    SessionSourceStateResponse,
)
from backend.auth.auth_db import AuthDB, AuthUser
from backend.core.config import settings
from backend.integrations.rag import RAGIntegrationError, RAGService
from backend.sessions.session_store import SessionState, SessionStore

router = APIRouter(tags=["База знаний"])

_auth_db: AuthDB = None  # type: ignore
_store: SessionStore = None  # type: ignore
_rag_service: RAGService = None  # type: ignore

_ALLOWED_EXTENSIONS = {
    ".csv",
    ".docx",
    ".htm",
    ".html",
    ".json",
    ".md",
    ".markdown",
    ".pdf",
    ".txt",
}


def setup(
    auth_db: AuthDB,
    store: SessionStore,
    rag_service: RAGService,
) -> None:
    global _auth_db, _store, _rag_service
    _auth_db = auth_db
    _store = store
    _rag_service = rag_service


def _load_owned_session(session_id: str, current_user: AuthUser) -> SessionState:
    if not _auth_db.is_session_owner(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    state = _store.load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


def _raise_rag_error(exc: Exception) -> None:
    raise HTTPException(status_code=502, detail=str(exc)) from exc


def _to_upload_response(payload: dict[str, object]) -> RagDocumentUploadResponse:
    return RagDocumentUploadResponse(
        status=str(payload.get("status") or "success"),
        message=str(payload.get("message") or ""),
        track_id=str(payload.get("track_id") or ""),
    )


@router.post(
    "/sessions/{session_id}/rag/documents",
    response_model=RagDocumentUploadResponse,
)
async def upload_rag_document(
    session_id: str,
    file: Annotated[UploadFile, File()],
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> RagDocumentUploadResponse:
    _load_owned_session(session_id, current_user)
    file_name = (file.filename or "document.txt").strip()
    extension = Path(file_name).suffix.lower()
    if extension not in _ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(_ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported knowledge-base file type. Allowed: {allowed}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    max_bytes = settings.max_dataset_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="Document exceeds size limit")

    try:
        result = _rag_service.upload_document(
            file_name=file_name,
            content=content,
            content_type=file.content_type or "application/octet-stream",
        )
    except RAGIntegrationError as exc:
        _raise_rag_error(exc)
    return _to_upload_response(result)


@router.get(
    "/sessions/{session_id}/rag/uploads/{track_id}",
    response_model=RagTrackStatusResponse,
)
def get_rag_upload_status(
    session_id: str,
    track_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> RagTrackStatusResponse:
    _load_owned_session(session_id, current_user)
    try:
        return RagTrackStatusResponse(**_rag_service.get_track_status(track_id))
    except RAGIntegrationError as exc:
        _raise_rag_error(exc)


@router.get(
    "/sessions/{session_id}/rag/documents",
    response_model=RagDocumentsResponse,
)
def list_rag_documents(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> RagDocumentsResponse:
    _load_owned_session(session_id, current_user)
    try:
        return RagDocumentsResponse(**_rag_service.list_documents())
    except RAGIntegrationError as exc:
        _raise_rag_error(exc)


@router.delete(
    "/sessions/{session_id}/rag/documents/{document_id}",
    response_model=RagDocumentDeleteResponse,
)
def delete_rag_document(
    session_id: str,
    document_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> RagDocumentDeleteResponse:
    _load_owned_session(session_id, current_user)
    try:
        return RagDocumentDeleteResponse(**_rag_service.delete_document(document_id))
    except RAGIntegrationError as exc:
        _raise_rag_error(exc)


@router.post(
    "/sessions/{session_id}/source/rag",
    response_model=SessionSourceStateResponse,
)
def bind_session_rag_source(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SessionSourceStateResponse:
    _load_owned_session(session_id, current_user)
    if not _rag_service.is_enabled:
        raise HTTPException(
            status_code=400,
            detail="RAG integration is disabled or not configured.",
        )
    _store.set_source(
        session_id,
        source_type="rag",
        source_ref_id="rag",
        source_label="База знаний",
        source_mode="lightrag",
    )
    refreshed = _load_owned_session(session_id, current_user)
    return SessionSourceStateResponse(
        source_type=refreshed.source_type,
        source_ref_id=refreshed.source_ref_id,
        source_label=refreshed.source_label,
        source_mode=refreshed.source_mode,
    )
