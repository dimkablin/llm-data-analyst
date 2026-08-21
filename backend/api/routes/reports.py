from __future__ import annotations

import tempfile
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.api.deps import get_current_user
from backend.auth.auth_db import AuthDB, AuthUser
from backend.auth.blob_store import BlobWrite, PostgresBlobStore
from backend.core.config import settings
from backend.core.json_utils import make_json_safe
from backend.data_access.csv_runtime_state_service import CSVRuntimeStateService
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.notebook.manifest_store import ManifestStore
from backend.services.planfact_export_validation import build_artifact_validation
from backend.services.report_export import build_board_export
from backend.sessions.session_store import SessionStore

router = APIRouter(tags=["Отчеты"])

_auth_db: AuthDB = None  # type: ignore
_store: SessionStore = None  # type: ignore
_csv_runtime: CSVSessionRuntime = None  # type: ignore
_manifest_store: ManifestStore = None  # type: ignore
_blob_store: PostgresBlobStore | None = None

_PLANFACT_COLUMN_LABELS = {
    "cfo": "ЦФО",
    "article": "Статья",
    "article_key": "Код статьи",
    "extra_key": "Доп. ключ",
    "service_content": "Содержание услуги",
    "plan_counterparty": "Контрагент плана",
    "fact_counterparty": "Контрагент факта",
    "fact_contract": "Договор",
    "period": "Период",
    "period_month": "Номер месяца",
    "plan_article": "Статья плана",
    "fact_article": "Статья факта",
    "plan_amount": "План",
    "fact_amount": "Факт",
    "variance_amount": "Отклонение",
    "variance_pct": "Отклонение %",
    "fact_rows": "Строк факта",
    "original_article_key": "Исходный код статьи",
    "matched_plan_article_key": "Сопоставленный код статьи",
    "matched_plan_article": "Сопоставленная статья",
    "article_match_type": "Тип сопоставления",
    "article_match_confidence": "Уверенность сопоставления",
    "plan_source_row_id": "Строка источника",
    "fact_source_row_id": "Строка источника",
}


def setup(
    *,
    auth_db: AuthDB,
    store: SessionStore,
    csv_runtime: CSVSessionRuntime,
    manifest_store: ManifestStore,
    blob_store: PostgresBlobStore | None = None,
) -> None:
    global _auth_db, _store, _csv_runtime, _manifest_store, _blob_store
    _auth_db = auth_db
    _store = store
    _csv_runtime = csv_runtime
    _manifest_store = manifest_store
    _blob_store = blob_store


class BoardExportSection(BaseModel):
    label: str = Field(default="", max_length=500)
    artifact_ids: list[str] = Field(default_factory=list)


class BoardExportRequest(BaseModel):
    format: Literal["docx", "pdf", "xlsx"] = "docx"
    title: str = Field(default="Отчёт по визуализациям", max_length=240)
    session_id: str | None = Field(default=None, max_length=128)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    sections: list[BoardExportSection] = Field(default_factory=list)


def _table_payload(dataframe) -> dict[str, Any]:
    frame = dataframe.copy()
    frame = frame.rename(columns=_PLANFACT_COLUMN_LABELS)
    split = make_json_safe(frame)
    return {
        "columns": list(split.get("columns") or []),
        "rows": list(split.get("data") or []),
    }


def _rename_payload_columns(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "columns": [
            _PLANFACT_COLUMN_LABELS.get(str(column), str(column)) for column in payload.get("columns") or []
        ],
        "rows": make_json_safe(payload.get("rows") or []),
    }


def _primary_rows_payload(
    *,
    runtime_session_id: str,
    table: str,
    id_column: str,
    links: dict[int, set[int]],
) -> dict[str, Any] | None:
    if not links:
        return None
    ids = ", ".join(str(int(row_id)) for row_id in sorted(links))
    try:
        frame = _csv_runtime.query_dataframe(
            runtime_session_id,
            f'SELECT * FROM "{table}" WHERE "{id_column}" IN ({ids}) ORDER BY "{id_column}"',
        )
    except Exception:
        return None
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        row_id = int(record[id_column])
        rows.extend(
            {"Строка результата": result_id, **record} for result_id in sorted(links.get(row_id) or [])
        )
    return _table_payload(pd.DataFrame(rows))


def _planfact_validation_tables(
    session_id: str,
    current_user: AuthUser,
    artifacts: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not _auth_db.is_session_owner(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    state = _store.load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if str(state.source_type or "").strip().lower() != "planfact":
        return {}

    state = CSVRuntimeStateService(
        store=_store,
        csv_runtime=_csv_runtime,
        manifest_store=_manifest_store,
        storage_dir=settings.storage_dir,
        blob_store=_blob_store,
    ).ensure_csv_runtime(
        session_id=session_id,
        ttl_seconds=settings.csv_session_ttl_sec,
    )
    runtime_session_id = str(state.csv_session_id or session_id)
    for artifact in artifacts:
        validation = build_artifact_validation(
            artifact,
            query_dataframe=lambda sql: _csv_runtime.query_dataframe(runtime_session_id, sql),
        )
        if validation is None:
            continue
        tables = {title: _rename_payload_columns(payload) for title, payload in validation["tables"].items()}
        plan = _primary_rows_payload(
            runtime_session_id=runtime_session_id,
            table="planfact_plan_raw",
            id_column="plan_source_row_id",
            links=validation["plan_links"],
        )
        fact = _primary_rows_payload(
            runtime_session_id=runtime_session_id,
            table="planfact_fact_raw",
            id_column="fact_source_row_id",
            links=validation["fact_links"],
        )
        if plan:
            tables["Первичка план"] = plan
        if fact:
            tables["Первичка факт"] = fact
        return tables
    return {
        "Проверка недоступна": {
            "columns": ["Сообщение"],
            "rows": [
                [
                    "Автоматическая проверка поддерживает обычные выборки и SUM, AVG, COUNT, "
                    "MIN, MAX с WHERE и GROUP BY по таблицам план-факта."
                ]
            ],
        }
    }


@router.get("/reports/download/{blob_id}/{file_name}")
def download_report(
    blob_id: str,
    file_name: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
):
    if _blob_store is None:
        raise HTTPException(status_code=503, detail="Report storage unavailable")
    blob = _blob_store.get(user_id=current_user.id, blob_id=blob_id, kind="report")
    if blob is None or blob.logical_name != Path(file_name).name:
        raise HTTPException(status_code=404, detail="Report not found")
    return StreamingResponse(
        BytesIO(blob.content),
        media_type=blob.media_type,
        headers={"Content-Disposition": f'attachment; filename="{blob.logical_name}"'},
    )


@router.post("/reports/board-export")
def export_board_report(
    payload: BoardExportRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
):
    if not payload.artifacts:
        raise HTTPException(status_code=400, detail="Нет артефактов для экспорта")
    if _blob_store is None:
        raise HTTPException(status_code=503, detail="Report storage unavailable")
    if payload.session_id and not _auth_db.is_session_owner(payload.session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        validation_tables = (
            _planfact_validation_tables(payload.session_id, current_user, payload.artifacts)
            if payload.format == "xlsx" and payload.session_id
            else {}
        )
        with tempfile.TemporaryDirectory(prefix="analyst-board-report-") as temp_dir:
            result = build_board_export(
                title=payload.title.strip() or "Отчёт по визуализациям",
                artifacts=payload.artifacts,
                output_dir=Path(temp_dir),
                export_format=payload.format,
                sections=[section.model_dump() for section in payload.sections],
                planfact_validation_tables=validation_tables,
            )
            content = Path(result.file_path).read_bytes()
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Не удалось собрать отчёт: {exc}") from exc

    media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if payload.format == "pdf":
        media_type = "application/pdf"
    elif payload.format == "xlsx":
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    _blob_store.put_many(
        user_id=current_user.id,
        session_id=payload.session_id,
        kind="report",
        items=[BlobWrite(logical_name=result.file_name, media_type=media_type, content=content)],
    )
    return StreamingResponse(
        BytesIO(content),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{result.file_name}"'},
    )
