from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from pydantic import BaseModel

from backend.api.deps import get_current_user
from backend.auth.auth_db import AuthDB, AuthUser
from backend.data_access.semantic_generation_service import (
    SemanticCatalogGenerationRequest,
    SemanticCatalogGenerationResponse,
)
from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticCatalogOperation,
    SemanticColumn,
    SemanticColumnPatch,
    SemanticContextResult,
    SemanticMetric,
    SemanticMetricCreate,
    SemanticMetricUpdate,
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
from backend.data_access.semantic_scenario_models import (
    SemanticScenarioApplyRequest,
    SemanticScenarioApplyResponse,
    SemanticScenarioRequest,
    SemanticScenarioReview,
)
from backend.sessions.session_store import SessionState, SessionStore

router = APIRouter(tags=["Semantic Layer"])
logger = logging.getLogger(__name__)

_auth_db: AuthDB = None  # type: ignore
_store: SessionStore = None  # type: ignore
_semantic_catalog_service = None  # type: ignore
_semantic_generation_service = None  # type: ignore
_semantic_scenario_service = None  # type: ignore
_db_runtime_service = None  # type: ignore


class SemanticCatalogStatusResponse(BaseModel):
    status: str
    catalog_id: str | None = None
    source_fingerprint: str | None = None
    updated_at: str | None = None
    error: str | None = None
    operation: SemanticCatalogOperation | None = None


class SemanticCatalogGenerationAcceptedResponse(BaseModel):
    accepted: bool = True
    status: str = "indexing"
    operation_id: int | None = None


def setup(
    *,
    auth_db: AuthDB,
    store: SessionStore,
    semantic_catalog_service,
    semantic_generation_service=None,
    semantic_scenario_service=None,
    db_runtime_service=None,
) -> None:
    global _auth_db, _store, _semantic_catalog_service, _semantic_generation_service
    global _semantic_scenario_service, _db_runtime_service
    _auth_db = auth_db
    _store = store
    _semantic_catalog_service = semantic_catalog_service
    _semantic_generation_service = semantic_generation_service
    _semantic_scenario_service = semantic_scenario_service
    _db_runtime_service = db_runtime_service


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


def _load_connection_runtime(connection_id: str, current_user: AuthUser):
    if _db_runtime_service is None:
        raise HTTPException(status_code=503, detail="DB runtime service is not configured")
    try:
        return _db_runtime_service.get_runtime_config(
            user_id=current_user.id,
            connection_id=connection_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="DB connection not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _require_connection_owner(connection_id: str, current_user: AuthUser):
    connection = _auth_db.get_db_connection(current_user.id, connection_id)
    if connection is None:
        raise HTTPException(status_code=404, detail="DB connection not found")
    if connection.user_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Only the DB connection owner can change its semantic catalog",
        )
    return _load_connection_runtime(connection_id, current_user)


def _require_semantic_editor(session_id: str, current_user: AuthUser) -> SessionState:
    state = _load_owned_session(session_id, current_user)
    if str(state.source_type or "").lower() == "db_connection" and state.source_ref_id:
        _require_connection_owner(str(state.source_ref_id), current_user)
    return state


@router.post(
    "/sessions/{session_id}/semantic-catalog/scenario-reviews",
    response_model=SemanticScenarioReview,
)
def analyze_semantic_scenarios(
    session_id: str,
    payload: SemanticScenarioRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticScenarioReview:
    _load_owned_session(session_id, current_user)
    if _semantic_scenario_service is None:
        raise HTTPException(status_code=503, detail="Semantic scenario service is not configured")
    try:
        return _semantic_scenario_service.analyze(
            session_id=session_id,
            user_id=current_user.id,
            request=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/sessions/{session_id}/semantic-catalog/scenario-reviews/{review_id:path}",
    response_model=SemanticScenarioReview,
)
def get_semantic_scenario_review(
    session_id: str,
    review_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticScenarioReview:
    _load_owned_session(session_id, current_user)
    if _semantic_scenario_service is None:
        raise HTTPException(status_code=503, detail="Semantic scenario service is not configured")
    try:
        return _semantic_scenario_service.get(
            session_id=session_id,
            user_id=current_user.id,
            review_id=review_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/sessions/{session_id}/semantic-catalog/scenario-reviews/{review_id:path}/apply",
    response_model=SemanticScenarioApplyResponse,
)
def apply_semantic_scenario_review(
    session_id: str,
    review_id: str,
    payload: SemanticScenarioApplyRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticScenarioApplyResponse:
    _require_semantic_editor(session_id, current_user)
    if _semantic_scenario_service is None:
        raise HTTPException(status_code=503, detail="Semantic scenario service is not configured")
    try:
        return _semantic_scenario_service.apply(
            session_id=session_id,
            user_id=current_user.id,
            review_id=review_id,
            request=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/db-connections/{connection_id}/semantic-catalog/status",
    response_model=SemanticCatalogStatusResponse,
)
def get_connection_semantic_catalog_status(
    connection_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticCatalogStatusResponse:
    _load_connection_runtime(connection_id, current_user)
    catalog = _semantic_catalog_service.status_for_connection(connection_id=connection_id)
    operation = _semantic_catalog_service.latest_operation_for_connection(
        connection_id=connection_id,
    )
    return SemanticCatalogStatusResponse(
        status=catalog.status,
        catalog_id=catalog.catalog_id,
        source_fingerprint=catalog.source_fingerprint,
        updated_at=catalog.updated_at,
        error=catalog.error,
        operation=operation,
    )


@router.get("/db-connections/{connection_id}/semantic-catalog", response_model=SemanticCatalog)
def get_connection_semantic_catalog(
    connection_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticCatalog:
    _load_connection_runtime(connection_id, current_user)
    catalog = _semantic_catalog_service.load_for_connection(
        connection_id=connection_id,
        user_id=current_user.id,
    )
    if catalog is None:
        raise HTTPException(status_code=404, detail="Semantic catalog not found")
    return catalog


@router.delete(
    "/db-connections/{connection_id}/semantic-catalog",
    status_code=204,
    response_class=Response,
    response_model=None,
)
def clear_connection_semantic_catalog(
    connection_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> None:
    _require_connection_owner(connection_id, current_user)
    try:
        _semantic_catalog_service.clear_for_connection(
            connection_id=connection_id,
            user_id=current_user.id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/db-connections/{connection_id}/semantic-catalog/build",
    response_model=SemanticCatalogGenerationAcceptedResponse,
    status_code=202,
)
def build_connection_semantic_catalog(
    connection_id: str,
    background_tasks: BackgroundTasks,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticCatalogGenerationAcceptedResponse:
    runtime = _require_connection_owner(connection_id, current_user)
    try:
        pending, operation = _semantic_catalog_service.claim_connection_build(
            connection_id=connection_id,
            user_id=current_user.id,
            source_label=str(getattr(runtime, "name", "") or ""),
            force=True,
        )
        if operation is not None:
            background_tasks.add_task(
                _run_connection_build_background,
                connection_id,
                current_user.id,
                pending.source_label,
                operation.operation_id,
            )
        return SemanticCatalogGenerationAcceptedResponse(
            status="indexing" if operation is not None else pending.status,
            operation_id=operation.operation_id if operation is not None else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/db-connections/{connection_id}/semantic-catalog/refresh",
    response_model=SemanticCatalogGenerationAcceptedResponse,
    status_code=202,
)
def refresh_connection_semantic_catalog(
    connection_id: str,
    background_tasks: BackgroundTasks,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticCatalogGenerationAcceptedResponse:
    return build_connection_semantic_catalog(connection_id, background_tasks, current_user)


def _run_connection_build_background(
    connection_id: str,
    user_id: int,
    source_label: str,
    operation_id: int,
) -> None:
    source_key = f"db_connection:{connection_id}"
    try:
        runtime = _db_runtime_service.get_runtime_config(
            user_id=user_id,
            connection_id=connection_id,
        )
        _semantic_catalog_service.build_for_connection(
            user_id=user_id,
            runtime=runtime,
            source_label=source_label,
            operation_id=operation_id,
        )
    except Exception as exc:
        logger.exception("Background semantic build failed for connection %s", connection_id)
        _semantic_catalog_service.mark_build_failed(
            source_key=source_key,
            error=str(exc),
            operation_id=operation_id,
        )


@router.get(
    "/db-connections/{connection_id}/semantic-catalog/metrics",
    response_model=list[SemanticMetric],
)
def list_connection_semantic_metrics(
    connection_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[SemanticMetric]:
    _load_connection_runtime(connection_id, current_user)
    catalog = _semantic_catalog_service.load_for_connection(
        connection_id=connection_id,
        user_id=current_user.id,
    )
    if catalog is None:
        raise HTTPException(status_code=404, detail="Semantic catalog not found")
    return catalog.metrics


@router.post(
    "/db-connections/{connection_id}/semantic-catalog/metrics",
    response_model=SemanticMetric,
)
def create_connection_semantic_metric(
    connection_id: str,
    payload: SemanticMetricCreate,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticMetric:
    _require_connection_owner(connection_id, current_user)
    try:
        return _semantic_catalog_service.create_metric_for_connection(
            connection_id=connection_id,
            user_id=current_user.id,
            payload=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch(
    "/db-connections/{connection_id}/semantic-catalog/metrics/{metric_id}",
    response_model=SemanticMetric,
)
def update_connection_semantic_metric(
    connection_id: str,
    metric_id: str,
    payload: SemanticMetricUpdate,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticMetric:
    _require_connection_owner(connection_id, current_user)
    try:
        return _semantic_catalog_service.update_metric_for_connection(
            connection_id=connection_id,
            user_id=current_user.id,
            metric_id=metric_id,
            payload=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete(
    "/db-connections/{connection_id}/semantic-catalog/metrics/{metric_id}",
    status_code=204,
    response_class=Response,
    response_model=None,
)
def delete_connection_semantic_metric(
    connection_id: str,
    metric_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> None:
    _require_connection_owner(connection_id, current_user)
    try:
        _semantic_catalog_service.delete_metric_for_connection(
            connection_id=connection_id,
            user_id=current_user.id,
            metric_id=metric_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/db-connections/{connection_id}/semantic-catalog/relationships",
    response_model=SemanticRelationship,
)
def create_connection_semantic_relationship(
    connection_id: str,
    payload: SemanticRelationshipCreate,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticRelationship:
    _require_connection_owner(connection_id, current_user)
    try:
        return _semantic_catalog_service.create_relationship_for_connection(
            connection_id=connection_id,
            user_id=current_user.id,
            payload=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch(
    "/db-connections/{connection_id}/semantic-catalog/relationships/{relationship_id}",
    response_model=SemanticRelationship,
)
def update_connection_semantic_relationship(
    connection_id: str,
    relationship_id: str,
    payload: SemanticRelationshipUpdate,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticRelationship:
    _require_connection_owner(connection_id, current_user)
    try:
        return _semantic_catalog_service.update_relationship_for_connection(
            connection_id=connection_id,
            user_id=current_user.id,
            relationship_id=relationship_id,
            payload=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete(
    "/db-connections/{connection_id}/semantic-catalog/relationships/{relationship_id}",
    status_code=204,
    response_class=Response,
    response_model=None,
)
def delete_connection_semantic_relationship(
    connection_id: str,
    relationship_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> None:
    _require_connection_owner(connection_id, current_user)
    try:
        _semantic_catalog_service.delete_relationship_for_connection(
            connection_id=connection_id,
            user_id=current_user.id,
            relationship_id=relationship_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/db-connections/{connection_id}/semantic-catalog/search",
    response_model=SemanticContextResult,
)
def search_connection_semantic_catalog(
    connection_id: str,
    payload: SemanticSearchRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticContextResult:
    _load_connection_runtime(connection_id, current_user)
    return _semantic_catalog_service.search_for_connection(
        connection_id=connection_id,
        user_id=current_user.id,
        query=payload.query,
        top_k=payload.top_k,
    )


@router.get("/sessions/{session_id}/semantic-catalog", response_model=SemanticCatalog)
def get_semantic_catalog(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticCatalog:
    return _load_catalog(session_id, current_user)


@router.delete(
    "/sessions/{session_id}/semantic-catalog",
    status_code=204,
    response_class=Response,
    response_model=None,
)
def clear_semantic_catalog(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> None:
    _require_semantic_editor(session_id, current_user)
    try:
        _semantic_catalog_service.clear_for_session(
            session_id=session_id,
            user_id=current_user.id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/sessions/{session_id}/semantic-catalog/status",
    response_model=SemanticCatalogStatusResponse,
)
def get_semantic_catalog_status(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticCatalogStatusResponse:
    _load_owned_session(session_id, current_user)
    operation = _semantic_catalog_service.latest_operation_for_session(
        session_id=session_id,
        user_id=current_user.id,
    )
    catalog = _semantic_catalog_service.load_for_session(
        session_id=session_id,
        user_id=current_user.id,
    )
    if catalog is None:
        state = _store.load_session(session_id)
        is_db_source = str(getattr(state, "source_type", "") or "").lower() == "db_connection"
        if is_db_source and getattr(state, "source_ref_id", None):
            connection_catalog = _semantic_catalog_service.status_for_connection(
                connection_id=str(state.source_ref_id),
            )
            return SemanticCatalogStatusResponse(
                status=connection_catalog.status,
                catalog_id=connection_catalog.catalog_id,
                source_fingerprint=connection_catalog.source_fingerprint,
                updated_at=connection_catalog.updated_at,
                error=connection_catalog.error,
                operation=(
                    _semantic_catalog_service.latest_operation_for_connection(
                        connection_id=str(state.source_ref_id),
                    )
                ),
            )
        return SemanticCatalogStatusResponse(status="empty", operation=operation)
    return SemanticCatalogStatusResponse(
        status=catalog.status,
        catalog_id=catalog.catalog_id,
        source_fingerprint=catalog.source_fingerprint,
        updated_at=catalog.updated_at,
        error=catalog.error,
        operation=operation,
    )


@router.post(
    "/sessions/{session_id}/semantic-catalog/refresh",
    response_model=SemanticCatalogGenerationAcceptedResponse,
    status_code=202,
)
def refresh_semantic_catalog(
    session_id: str,
    background_tasks: BackgroundTasks,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SemanticCatalogGenerationAcceptedResponse:
    state = _require_semantic_editor(session_id, current_user)
    if state.source_type == "db_connection" and state.source_ref_id:
        return build_connection_semantic_catalog(
            state.source_ref_id,
            background_tasks,
            current_user,
        )
    pending, operation = _semantic_catalog_service.claim_session_build(
        session_id=session_id,
        user_id=current_user.id,
        force=True,
    )
    if operation is not None:
        background_tasks.add_task(
            _run_session_build_background,
            session_id,
            current_user.id,
            pending.source_key,
            operation.operation_id,
        )
    return SemanticCatalogGenerationAcceptedResponse(
        status="indexing" if operation is not None else pending.status,
        operation_id=operation.operation_id if operation is not None else None,
    )


def _run_session_build_background(
    session_id: str,
    user_id: int,
    source_key: str,
    operation_id: int,
) -> None:
    try:
        _semantic_catalog_service.refresh(
            session_id=session_id,
            user_id=user_id,
            operation_id=operation_id,
        )
    except Exception as exc:
        logger.exception("Background semantic build failed for session %s", session_id)
        _semantic_catalog_service.mark_build_failed(
            source_key=source_key,
            error=str(exc),
            operation_id=operation_id,
        )


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
    state = _require_semantic_editor(session_id, current_user)
    if str(state.source_type or "").lower() not in {"db_connection", "csv"}:
        raise HTTPException(
            status_code=400,
            detail="AI semantic generation is available only for database or uploaded CSV/XLSX sources",
        )
    if background:
        catalog = _semantic_catalog_service.load_for_session(session_id=session_id, user_id=current_user.id)
        if catalog is None or catalog.status in {"not_built", "pending", "indexing", "failed"}:
            raise HTTPException(status_code=409, detail="Build the semantic catalog before AI generation")
        pending, operation = _semantic_catalog_service.claim_session_build(
            session_id=session_id,
            user_id=current_user.id,
            operation_type="generate",
        )
        if operation is None:
            raise HTTPException(status_code=409, detail="A semantic operation is already running")
        background_tasks.add_task(
            _run_semantic_generation_background,
            session_id,
            current_user.id,
            payload,
            operation.operation_id,
        )
        return SemanticCatalogGenerationAcceptedResponse(
            status=pending.status,
            operation_id=operation.operation_id,
        )
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
    operation_id: int,
) -> None:
    try:
        _semantic_generation_service.generate(
            session_id=session_id,
            user_id=user_id,
            request=payload,
            operation_id=operation_id,
        )
    except Exception as exc:
        logger.exception("Background semantic generation failed for session %s", session_id)
        _semantic_catalog_service.fail_operation(operation_id=operation_id, error=str(exc))


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
            if item.table_id == table_id or item.qualified_name == table_id or item.table_name == table_id
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
    _require_semantic_editor(session_id, current_user)
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
    _require_semantic_editor(session_id, current_user)
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
    _require_semantic_editor(session_id, current_user)
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
    _require_semantic_editor(session_id, current_user)
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
    _require_semantic_editor(session_id, current_user)
    try:
        _semantic_catalog_service.delete_metric(
            session_id=session_id,
            user_id=current_user.id,
            metric_id=metric_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
    _require_semantic_editor(session_id, current_user)
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
    _require_semantic_editor(session_id, current_user)
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
    _require_semantic_editor(session_id, current_user)
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
    _require_semantic_editor(session_id, current_user)
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
    _require_semantic_editor(session_id, current_user)
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
    _require_semantic_editor(session_id, current_user)
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
