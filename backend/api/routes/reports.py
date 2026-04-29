from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from backend.api.deps import get_current_user
from backend.auth.auth_db import AuthUser
from backend.core.config import settings

router = APIRouter(tags=["Отчеты"])


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

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
