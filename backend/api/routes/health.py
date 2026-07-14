from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from backend.api.deps import get_current_user
from backend.api.models import RuntimeModelProfileResponse
from backend.auth.auth_db import AuthUser
from backend.core.config import settings
from backend.core.public_identity import runtime_model_payload

router = APIRouter(tags=["Сервис"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/runtime/model", response_model=RuntimeModelProfileResponse)
def runtime_model(
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> RuntimeModelProfileResponse:
    max_context_tokens = settings.llm_num_ctx if settings.llm_num_ctx > 0 else None
    payload = runtime_model_payload(
        is_admin=current_user.is_admin,
        provider=settings.llm_provider,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        max_context_tokens=max_context_tokens,
        context_window_source="settings" if max_context_tokens else "unavailable",
    )
    return RuntimeModelProfileResponse.model_validate(payload)
