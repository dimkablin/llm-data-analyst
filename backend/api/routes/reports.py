from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.api.deps import get_current_user
from backend.auth.auth_db import AuthUser
from backend.core.config import settings
from backend.services.report_export import build_board_export

router = APIRouter(tags=["Отчеты"])


class BoardExportSection(BaseModel):
    label: str = Field(default="", max_length=500)
    artifact_ids: list[str] = Field(default_factory=list)


class BoardExportRequest(BaseModel):
    format: Literal["docx", "pdf", "xlsx"] = "docx"
    title: str = Field(default="Отчёт по визуализациям", max_length=240)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    sections: list[BoardExportSection] = Field(default_factory=list)


def _report_export_dir() -> Path:
    return Path(settings.storage_dir) / "report_exports"


@router.get("/reports/download/{file_name}")
def download_report(
    file_name: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
):
    # Пока current_user нужен минимум для проверки авторизации.
    # Если get_current_user не пройдет — файл не будет отдан.
    _ = current_user

    export_root = _report_export_dir().resolve()
    file_path = (export_root / file_name).resolve()

    # Защита от path traversal: нельзя выйти за пределы report_exports.
    if export_root not in file_path.parents and file_path != export_root:
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Report not found")

    media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if file_path.suffix.lower() == ".pdf":
        media_type = "application/pdf"
    elif file_path.suffix.lower() == ".xlsx":
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type=media_type,
    )


@router.post("/reports/board-export")
def export_board_report(
    payload: BoardExportRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
):
    _ = current_user
    if not payload.artifacts:
        raise HTTPException(status_code=400, detail="Нет артефактов для экспорта")

    try:
        result = build_board_export(
            title=payload.title.strip() or "Отчёт по визуализациям",
            artifacts=payload.artifacts,
            output_dir=_report_export_dir(),
            export_format=payload.format,
            sections=[section.model_dump() for section in payload.sections],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Не удалось собрать отчёт: {exc}") from exc

    file_path = Path(result.file_path).resolve()
    export_root = _report_export_dir().resolve()
    if export_root not in file_path.parents:
        raise HTTPException(status_code=500, detail="Invalid export path")

    media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if payload.format == "pdf":
        media_type = "application/pdf"
    elif payload.format == "xlsx":
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return FileResponse(
        path=str(file_path),
        filename=result.file_name,
        media_type=media_type,
    )
