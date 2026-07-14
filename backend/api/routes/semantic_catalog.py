from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from pydantic import BaseModel

from backend.api.deps import get_current_user
from backend.auth.auth_db import AuthDB, AuthUser
from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticColumn,
    SemanticContextResult,
    SemanticMetric,
    SemanticMetricCreate,
    SemanticMetricUpdate,
    SemanticColumnPatch,
    SemanticRelationship,
    SemanticRelationshipCreate,
    SemanticRelationshipUpdate,
    SemanticSearchRequest,
    SemanticTable,
    SemanticTablePatch,
    SemanticTerm,
    SemanticTermCreate,
    SemanticTermUpdate,
)
from backend.data_access.semantic_models import utc_now_iso
from backend.data_access.semantic_generation_service import (
    SemanticCatalogGenerationRequest,
    SemanticCatalogGenerationResponse,
)
from backend.sessions.session_store import SessionState, SessionStore

router = APIRouter(tags=["Semantic Layer"])
logger = logging.getLogger(__name__)

_auth_db: AuthDB = None  # type: ignore
_store: SessionStore = None  # type: ignore
_semantic_catalog_service = None  # type: ignore
_semantic_generation_service = None  # type: ignore


class SemanticCatalogStatusResponse(BaseModel):
    status: str
    catalog_id: str | None = None
    source_fingerprint: str | None = None
    updated_at: str | None = None
    error: str | None = None


class SemanticCatalogGenerationAcceptedResponse(BaseModel):
    accepted: bool = True
    status: str = "indexing"


def setup(
    *,
    auth_db: AuthDB,
    store: SessionStore,
    semantic_catalog_service,
    semantic_generation_service=None,
) -> None:
    global _auth_db, _store, _semantic_catalog_service, _semantic_generation_service
    _auth_db = auth_db
    _store = store
    _semantic_catalog_service = semantic_catalog_service
    _semantic_generation_service = semantic_generation_service


def _load_owned_session(session_id: str, current_user: AuthUser) -> SessionState:
    if not _auth_db.is_session_owner(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    state = _store.load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


def _load_catalog(session_id: str, current_user: AuthUser) -> SemanticCatalog:
    _load_owned_session(session_id, current_user)
    catalog = _semantic_catalog_service.load_for_session(
        session_id=session_id,
        user_id=current_user.id,
    )
    if catalog is None:
        raise HTTPException(status_code=404, detail="Semantic catalog not found")
    return catalog


@router.get("/sessions/{session_id}/semantic-catalog", response_model=SemanticCatalog)
def get_semantic_catalog(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticCatalog:
    return _load_catalog(session_id, current_user)


@router.get(
    "/sessions/{session_id}/semantic-catalog/status",
    response_model=SemanticCatalogStatusResponse,
)
def get_semantic_catalog_status(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticCatalogStatusResponse:
    _load_owned_session(session_id, current_user)
    catalog = _semantic_catalog_service.load_for_session(
        session_id=session_id,
        user_id=current_user.id,
    )
    if catalog is None:
        return SemanticCatalogStatusResponse(status="empty")
    return SemanticCatalogStatusResponse(
        status=catalog.status,
        catalog_id=catalog.catalog_id,
        source_fingerprint=catalog.source_fingerprint,
        updated_at=catalog.updated_at,
        error=catalog.error,
    )


@router.post("/sessions/{session_id}/semantic-catalog/refresh", response_model=SemanticCatalog)
def refresh_semantic_catalog(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticCatalog:
    _load_owned_session(session_id, current_user)
    return _semantic_catalog_service.refresh(session_id=session_id, user_id=current_user.id)


@router.post(
    "/sessions/{session_id}/semantic-catalog/generate",
    response_model=SemanticCatalogGenerationResponse | SemanticCatalogGenerationAcceptedResponse,
)
def generate_semantic_catalog(
    session_id: str,
    background_tasks: BackgroundTasks,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
    payload: SemanticCatalogGenerationRequest | None = None,
    background: bool = False,
) -> SemanticCatalogGenerationResponse | SemanticCatalogGenerationAcceptedResponse:
    if _semantic_generation_service is None:
        raise HTTPException(status_code=503, detail="Semantic generation is not configured")
    state = _load_owned_session(session_id, current_user)
    if str(state.source_type or "").lower() not in {"db_connection", "csv"}:
        raise HTTPException(
            status_code=400,
            detail="AI semantic generation is available only for database or uploaded CSV/XLSX sources",
        )
    if background:
        catalog = _semantic_catalog_service.load_for_session(session_id=session_id, user_id=current_user.id)
        if catalog is not None:
            catalog.status = "indexing"
            catalog.error = None
            catalog.updated_at = utc_now_iso()
            _semantic_catalog_service.save_runtime_status(catalog)
        background_tasks.add_task(
            _run_semantic_generation_background,
            session_id,
            current_user.id,
            payload,
        )
        return SemanticCatalogGenerationAcceptedResponse()
    try:
        return _semantic_generation_service.generate(
            session_id=session_id,
            user_id=current_user.id,
            request=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _run_semantic_generation_background(
    session_id: str,
    user_id: int,
    payload: SemanticCatalogGenerationRequest | None,
) -> None:
    try:
        _semantic_generation_service.generate(session_id=session_id, user_id=user_id, request=payload)
    except Exception as exc:
        logger.exception("Background semantic generation failed for session %s", session_id)
        catalog = _semantic_catalog_service.load_for_session(session_id=session_id, user_id=user_id)
        if catalog is not None:
            catalog.status = "failed"
            catalog.error = str(exc)
            catalog.updated_at = utc_now_iso()
            _semantic_catalog_service.save_runtime_status(catalog)


@router.get("/sessions/{session_id}/semantic-catalog/tables", response_model=list[SemanticTable])
def list_semantic_tables(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[SemanticTable]:
    return _load_catalog(session_id, current_user).tables


@router.get("/sessions/{session_id}/semantic-catalog/tables/{table_id}", response_model=SemanticTable)
def get_semantic_table(
    session_id: str,
    table_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticTable:
    catalog = _load_catalog(session_id, current_user)
    table = next(
        (
            item
            for item in catalog.tables
            if item.table_id == table_id
            or item.qualified_name == table_id
            or item.table_name == table_id
        ),
        None,
    )
    if table is None:
        raise HTTPException(status_code=404, detail="Semantic table not found")
    return table


@router.patch("/sessions/{session_id}/semantic-catalog/tables/{table_id}", response_model=SemanticTable)
def patch_semantic_table(
    session_id: str,
    table_id: str,
    payload: SemanticTablePatch,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticTable:
    _load_owned_session(session_id, current_user)
    try:
        return _semantic_catalog_service.patch_table(
            session_id=session_id,
            user_id=current_user.id,
            table_id=table_id,
            payload=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/semantic-catalog/columns", response_model=list[SemanticColumn])
def list_semantic_columns(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[SemanticColumn]:
    return _load_catalog(session_id, current_user).columns


@router.patch("/sessions/{session_id}/semantic-catalog/columns/{column_id}", response_model=SemanticColumn)
def patch_semantic_column(
    session_id: str,
    column_id: str,
    payload: SemanticColumnPatch,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticColumn:
    _load_owned_session(session_id, current_user)
    try:
        return _semantic_catalog_service.patch_column(
            session_id=session_id,
            user_id=current_user.id,
            column_id=column_id,
            payload=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/semantic-catalog/metrics", response_model=list[SemanticMetric])
def list_semantic_metrics(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[SemanticMetric]:
    return _load_catalog(session_id, current_user).metrics


@router.post("/sessions/{session_id}/semantic-catalog/metrics", response_model=SemanticMetric)
def create_semantic_metric(
    session_id: str,
    payload: SemanticMetricCreate,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticMetric:
    _load_owned_session(session_id, current_user)
    try:
        return _semantic_catalog_service.create_metric(
            session_id=session_id,
            user_id=current_user.id,
            payload=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch(
    "/sessions/{session_id}/semantic-catalog/metrics/{metric_id}",
    response_model=SemanticMetric,
)
def update_semantic_metric(
    session_id: str,
    metric_id: str,
    payload: SemanticMetricUpdate,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticMetric:
    _load_owned_session(session_id, current_user)
    try:
        return _semantic_catalog_service.update_metric(
            session_id=session_id,
            user_id=current_user.id,
            metric_id=metric_id,
            payload=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete(
    "/sessions/{session_id}/semantic-catalog/metrics/{metric_id}",
    status_code=204,
    response_class=Response,
    response_model=None,
)
def delete_semantic_metric(
    session_id: str,
    metric_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> None:
    _load_owned_session(session_id, current_user)
    _semantic_catalog_service.delete_metric(
        session_id=session_id,
        user_id=current_user.id,
        metric_id=metric_id,
    )


@router.get(
    "/sessions/{session_id}/semantic-catalog/relationships",
    response_model=list[SemanticRelationship],
)
def list_semantic_relationships(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[SemanticRelationship]:
    return _load_catalog(session_id, current_user).relationships


@router.post(
    "/sessions/{session_id}/semantic-catalog/relationships",
    response_model=SemanticRelationship,
)
def create_semantic_relationship(
    session_id: str,
    payload: SemanticRelationshipCreate,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticRelationship:
    _load_owned_session(session_id, current_user)
    try:
        return _semantic_catalog_service.create_relationship(
            session_id=session_id,
            user_id=current_user.id,
            payload=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch(
    "/sessions/{session_id}/semantic-catalog/relationships/{relationship_id}",
    response_model=SemanticRelationship,
)
def update_semantic_relationship(
    session_id: str,
    relationship_id: str,
    payload: SemanticRelationshipUpdate,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticRelationship:
    _load_owned_session(session_id, current_user)
    try:
        return _semantic_catalog_service.update_relationship(
            session_id=session_id,
            user_id=current_user.id,
            relationship_id=relationship_id,
            payload=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete(
    "/sessions/{session_id}/semantic-catalog/relationships/{relationship_id}",
    status_code=204,
    response_class=Response,
    response_model=None,
)
def delete_semantic_relationship(
    session_id: str,
    relationship_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> None:
    _load_owned_session(session_id, current_user)
    _semantic_catalog_service.delete_relationship(
        session_id=session_id,
        user_id=current_user.id,
        relationship_id=relationship_id,
    )


@router.get("/sessions/{session_id}/semantic-catalog/terms", response_model=list[SemanticTerm])
def list_semantic_terms(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[SemanticTerm]:
    return _load_catalog(session_id, current_user).terms


@router.post("/sessions/{session_id}/semantic-catalog/terms", response_model=SemanticTerm)
def create_semantic_term(
    session_id: str,
    payload: SemanticTermCreate,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticTerm:
    _load_owned_session(session_id, current_user)
    try:
        return _semantic_catalog_service.create_term(
            session_id=session_id,
            user_id=current_user.id,
            payload=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch(
    "/sessions/{session_id}/semantic-catalog/terms/{term_id}",
    response_model=SemanticTerm,
)
def update_semantic_term(
    session_id: str,
    term_id: str,
    payload: SemanticTermUpdate,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticTerm:
    _load_owned_session(session_id, current_user)
    try:
        return _semantic_catalog_service.update_term(
            session_id=session_id,
            user_id=current_user.id,
            term_id=term_id,
            payload=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete(
    "/sessions/{session_id}/semantic-catalog/terms/{term_id}",
    status_code=204,
    response_class=Response,
    response_model=None,
)
def delete_semantic_term(
    session_id: str,
    term_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> None:
    _load_owned_session(session_id, current_user)
    _semantic_catalog_service.delete_term(
        session_id=session_id,
        user_id=current_user.id,
        term_id=term_id,
    )


@router.post("/sessions/{session_id}/semantic-catalog/search", response_model=SemanticContextResult)
def search_semantic_catalog(
    session_id: str,
    payload: SemanticSearchRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticContextResult:
    _load_owned_session(session_id, current_user)
    return _semantic_catalog_service.search(
        session_id=session_id,
        user_id=current_user.id,
        query=payload.query,
        top_k=payload.top_k,
    )
