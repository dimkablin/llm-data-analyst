from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from backend.api.deps import get_current_user
from backend.api.models import PhoenixOverviewResponse
from backend.auth.auth_db import AuthUser

router = APIRouter(tags=["Наблюдаемость"])

# Singleton set during app startup
_phoenix_observability_service = None  # type: ignore


def setup(phoenix_observability_service) -> None:
    global _phoenix_observability_service
    _phoenix_observability_service = phoenix_observability_service


@router.get("/observability/phoenix", response_model=PhoenixOverviewResponse)
def phoenix_overview(
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> PhoenixOverviewResponse:
    _ = current_user
    return _phoenix_observability_service.build_overview()


