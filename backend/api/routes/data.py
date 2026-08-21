from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import duckdb
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)

from backend.api.deps import get_current_user
from backend.api.models import BatchUploadResponse, UploadResponse
from backend.auth.auth_db import AuthDB, AuthUser
from backend.auth.blob_store import PostgresBlobStore
from backend.core.config import settings
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.data_access.planfact_source_service import (
    PlanfactSourceError,
    PlanfactSourceService,
    PlanfactUploadFile,
)
from backend.data_access.tabular_preprocessing import TabularPreprocessingOptions
from backend.data_access.tabular_upload_service import (
    TabularUploadError,
    TabularUploadFile,
    TabularUploadResult,
    TabularUploadService,
)
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.orchestrator import NotebookOrchestrator
from backend.sessions.session_store import SessionState, SessionStore

router = APIRouter(tags=["Данные"])

# Singletons set during app startup
_auth_db: AuthDB = None  # type: ignore
_store: SessionStore = None  # type: ignore
_csv_runtime: CSVSessionRuntime = None  # type: ignore
_manifest_store: ManifestStore = None  # type: ignore
_orchestrator: NotebookOrchestrator = None  # type: ignore
_storage_dir: Path | None = None
_semantic_catalog_service = None  # type: ignore
_blob_store: PostgresBlobStore | None = None


def setup(
    auth_db: AuthDB,
    store: SessionStore,
    csv_runtime: CSVSessionRuntime,
    manifest_store: ManifestStore,
    notebook_orchestrator: NotebookOrchestrator,
    storage_dir: str | Path | None = None,
    semantic_catalog_service=None,
    blob_store: PostgresBlobStore | None = None,
) -> None:
    global _auth_db, _store, _csv_runtime, _manifest_store, _orchestrator, _storage_dir
    global _semantic_catalog_service, _blob_store
    _auth_db = auth_db
    _store = store
    _csv_runtime = csv_runtime
    _manifest_store = manifest_store
    _orchestrator = notebook_orchestrator
    _storage_dir = Path(storage_dir) if storage_dir is not None else Path(settings.storage_dir)
    _semantic_catalog_service = semantic_catalog_service
    _blob_store = blob_store


def _load_owned_session(session_id: str, current_user: AuthUser) -> SessionState:
    if not _auth_db.is_session_owner(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    state = _store.load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


def _ensure_safe_readonly_sql(sql: str) -> str:
    clean = str(sql or "").strip()
    if not clean:
        raise HTTPException(status_code=400, detail="sql must not be empty")
    try:
        statements = duckdb.extract_statements(clean)
    except duckdb.ParserException as exc:
        raise HTTPException(status_code=400, detail=f"Invalid SQL: {exc}") from exc
    if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
        raise HTTPException(status_code=400, detail="Only read-only queries are allowed")
    return clean


def _df_to_json_rows(df) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _upload_service() -> TabularUploadService:
    return TabularUploadService(
        store=_store,
        csv_runtime=_csv_runtime,
        manifest_store=_manifest_store,
        notebook_orchestrator=_orchestrator,
        storage_dir=_storage_dir or Path(settings.storage_dir),
        blob_store=_blob_store,
    )


def _planfact_service() -> PlanfactSourceService:
    return PlanfactSourceService(
        store=_store,
        csv_runtime=_csv_runtime,
        manifest_store=_manifest_store,
        storage_dir=_storage_dir or Path(settings.storage_dir),
        blob_store=_blob_store,
    )


async def _read_tabular_uploads(files: list[UploadFile]) -> list[TabularUploadFile]:
    max_bytes = settings.max_dataset_mb * 1024 * 1024
    uploads: list[TabularUploadFile] = []
    for file in files:
        file_name = (file.filename or "uploaded.csv").strip()
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"Uploaded file '{file_name}' is empty")
        if len(content) > max_bytes:
            raise HTTPException(status_code=413, detail=f"Dataset '{file_name}' exceeds size limit")
        uploads.append(
            TabularUploadFile(
                file_name=file_name,
                content=content,
                content_type=file.content_type,
            )
        )
    return uploads


async def _read_planfact_upload(file: UploadFile) -> PlanfactUploadFile:
    max_bytes = settings.max_dataset_mb * 1024 * 1024
    file_name = (file.filename or "uploaded.xlsx").strip()
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"Uploaded file '{file_name}' is empty")
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Dataset '{file_name}' exceeds size limit")
    return PlanfactUploadFile(
        file_name=file_name,
        content=content,
        content_type=file.content_type,
    )


def _build_csv_semantic_catalog(
    session_id: str,
    user_id: int,
    source_key: str,
    operation_id: int,
) -> None:
    try:
        from backend.data_access.catalog_refresh import refresh_session_catalog

        refresh_session_catalog(
            _store,
            session_id,
            csv_runtime=_csv_runtime,
        )
        _semantic_catalog_service.refresh(
            session_id=session_id,
            user_id=user_id,
            operation_id=operation_id,
        )
    except Exception as exc:
        _semantic_catalog_service.mark_build_failed(
            source_key=source_key,
            error=str(exc),
            operation_id=operation_id,
        )


def _queue_csv_semantic_build(
    background_tasks: BackgroundTasks,
    *,
    session_id: str,
    user_id: int,
) -> None:
    if _semantic_catalog_service is None or not settings.semantic_layer_enabled:
        return
    pending, operation = _semantic_catalog_service.claim_session_build(
        session_id=session_id,
        user_id=user_id,
    )
    if operation is not None:
        background_tasks.add_task(
            _build_csv_semantic_catalog,
            session_id,
            user_id,
            pending.source_key,
            operation.operation_id,
        )


def _batch_response(result) -> BatchUploadResponse:
    return BatchUploadResponse.model_validate(result.model_dump())


def _parse_preprocessing_options(raw: str | None) -> TabularPreprocessingOptions:
    if raw is None or not str(raw).strip():
        return TabularPreprocessingOptions()
    try:
        return TabularPreprocessingOptions.model_validate_json(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid preprocessing options: {exc}") from exc


def _mark_dataset_loaded(session_id: str) -> None:
    _auth_db.mark_session_has_dataset(session_id, True)


def _handle_planfact_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, PlanfactSourceError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=400, detail=f"Failed to process planfact source: {exc}")


async def _ingest_tabular_uploads(
    *,
    session_id: str,
    files: list[UploadFile],
    background_tasks: BackgroundTasks,
    current_user: AuthUser,
    preprocessing_options: str | None,
    failure_message: str,
) -> TabularUploadResult:
    _load_owned_session(session_id, current_user)
    try:
        result = _upload_service().ingest_files(
            session_id=session_id,
            files=await _read_tabular_uploads(files),
            ttl_seconds=settings.csv_session_ttl_sec,
            max_bytes_per_file=settings.max_dataset_mb * 1024 * 1024,
            preprocessing_options=_parse_preprocessing_options(preprocessing_options),
            user_id=current_user.id,
        )
    except TabularUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{failure_message}: {exc}") from exc

    _mark_dataset_loaded(session_id)
    _queue_csv_semantic_build(
        background_tasks,
        session_id=session_id,
        user_id=current_user.id,
    )
    return result


@router.post("/sessions/{session_id}/data", response_model=UploadResponse)
async def upload_data(
    session_id: str,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File()],
    current_user: Annotated[AuthUser, Depends(get_current_user)],
    preprocessing_options: Annotated[str | None, Form()] = None,
) -> UploadResponse:
    result = await _ingest_tabular_uploads(
        session_id=session_id,
        files=[file],
        background_tasks=background_tasks,
        current_user=current_user,
        preprocessing_options=preprocessing_options,
        failure_message="Failed to read tabular file",
    )
    first = result.files[0]

    return UploadResponse(
        session_id=session_id,
        rows=first.rows,
        columns=first.columns,
    )


@router.post("/sessions/{session_id}/data/batch", response_model=BatchUploadResponse)
async def upload_data_batch(
    session_id: str,
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File()],
    current_user: Annotated[AuthUser, Depends(get_current_user)],
    preprocessing_options: Annotated[str | None, Form()] = None,
) -> BatchUploadResponse:
    result = await _ingest_tabular_uploads(
        session_id=session_id,
        files=files,
        background_tasks=background_tasks,
        current_user=current_user,
        preprocessing_options=preprocessing_options,
        failure_message="Failed to read tabular files",
    )
    return _batch_response(result)


@router.post("/sessions/{session_id}/source/planfact/detect")
async def detect_planfact_source(
    session_id: str,
    plan_file: Annotated[UploadFile, File()],
    fact_file: Annotated[UploadFile, File()],
    current_user: Annotated[AuthUser, Depends(get_current_user)],
    mapping_file: Annotated[UploadFile | None, File()] = None,
) -> dict:
    _load_owned_session(session_id, current_user)
    try:
        plan_upload = await _read_planfact_upload(plan_file)
        fact_upload = await _read_planfact_upload(fact_file)
        mapping_upload = (
            await _read_planfact_upload(mapping_file) if mapping_file is not None else None
        )
        result = _planfact_service().detect(
            session_id=session_id,
            plan_file=plan_upload,
            fact_file=fact_upload,
            mapping_file=mapping_upload,
            user_id=current_user.id,
        )
    except Exception as exc:
        raise _handle_planfact_error(exc) from exc
    return result.model_dump()


@router.post("/sessions/{session_id}/source/planfact/confirm")
async def confirm_planfact_source(
    session_id: str,
    background_tasks: BackgroundTasks,
    config: Annotated[dict, Body()],
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    _load_owned_session(session_id, current_user)
    try:
        result = _planfact_service().confirm(
            session_id=session_id,
            config=config,
            ttl_seconds=settings.csv_session_ttl_sec,
            user_id=current_user.id,
        )
    except Exception as exc:
        raise _handle_planfact_error(exc) from exc

    _mark_dataset_loaded(session_id)
    _queue_csv_semantic_build(
        background_tasks,
        session_id=session_id,
        user_id=current_user.id,
    )
    return result.model_dump()


@router.get("/sessions/{session_id}/source/planfact/config")
async def get_planfact_config(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    _load_owned_session(session_id, current_user)
    try:
        return _planfact_service().get_config(session_id)
    except Exception as exc:
        raise _handle_planfact_error(exc) from exc


@router.patch("/sessions/{session_id}/source/planfact/config")
async def patch_planfact_config(
    session_id: str,
    patch: Annotated[dict, Body()],
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> dict:
    _load_owned_session(session_id, current_user)
    try:
        return _planfact_service().update_config(session_id, patch, user_id=current_user.id)
    except Exception as exc:
        raise _handle_planfact_error(exc) from exc


# Predict-service calls this endpoint without a bearer token and passes the CSV
# runtime session_id, matching /csv/query below.
@router.get("/csv/schema")
async def csv_schema(
    session_id: Annotated[str, Query()],
) -> dict:
    try:
        info = _csv_runtime.get_session_info(session_id)
        tables = _csv_runtime.list_tables(session_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"CSV session not found: {exc}") from exc

    if not tables:
        raise HTTPException(status_code=404, detail="No tables found in CSV session")

    main_table = str(tables[0].get("table_name") or "").strip()
    if not main_table:
        raise HTTPException(status_code=500, detail="CSV session returned invalid table metadata")

    columns_meta = _csv_runtime.describe_table(session_id, main_table)

    return {
        "session_id": session_id,
        "table_name": main_table,
        "table_names": [str(t.get("table_name") or "") for t in tables if t.get("table_name")],
        "columns": [
            {
                "name": str(col.get("column_name") or ""),
                "type": str(col.get("data_type") or ""),
                "nullable": bool(col.get("is_nullable")),
                "ordinal_position": int(col.get("ordinal_position") or 0),
            }
            for col in columns_meta
        ],
        "expires_at": int(info.expires_at),
    }


@router.get("/csv/query")
async def query_table(
    session_id: Annotated[str, Query()],
    sql: Annotated[str, Query()],
) -> dict:
    safe_sql = _ensure_safe_readonly_sql(sql)

    try:
        df = _csv_runtime.query_dataframe(session_id, safe_sql)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to execute query: {exc}") from exc

    rows = _df_to_json_rows(df)

    return {
        "result": rows,
        "row_count": len(rows),
        "columns": [str(c) for c in df.columns],
    }
