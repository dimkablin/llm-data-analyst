from __future__ import annotations

from fastapi import APIRouter, Depends
from backend.auth.auth_db import AuthUser
from backend.api.deps import get_current_user
from backend.core.config import settings

router = APIRouter(tags=["Сервис"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/runtime/model")
def runtime_model(current_user: AuthUser = Depends(get_current_user)) -> dict[str, str]:
    _ = current_user
    return {
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
    }


