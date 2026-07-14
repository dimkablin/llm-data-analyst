from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from backend.api.deps import get_current_user
from backend.api.models import PhoenixOverviewResponse
from backend.auth.auth_db import AuthUser
from backend.core.public_identity import PUBLIC_ASSISTANT_MODEL
from backend.observability.models import (
    PhoenixTraceDetailResponse,
    PhoenixTraceRow,
    PhoenixTracesResponse,
)

router = APIRouter(tags=["Наблюдаемость"])

# Singleton set during app startup
_phoenix_observability_service = None  # type: ignore


def setup(phoenix_observability_service) -> None:
    global _phoenix_observability_service
    _phoenix_observability_service = phoenix_observability_service


def _redact_phoenix_models(
    overview: PhoenixOverviewResponse,
    *,
    is_admin: bool,
) -> PhoenixOverviewResponse:
    if is_admin:
        return overview
    return overview.model_copy(
        update={
            "traces": [
                trace.model_copy(update={"model": PUBLIC_ASSISTANT_MODEL})
                if trace.model
                else trace
                for trace in overview.traces
            ],
            "token_usage": [
                row.model_copy(update={"model": PUBLIC_ASSISTANT_MODEL})
                if row.model
                else row
                for row in overview.token_usage
            ],
        }
    )


def _redact_trace_row(
    row: PhoenixTraceRow,
    *,
    is_admin: bool,
) -> PhoenixTraceRow:
    if is_admin or row.model is None:
        return row
    return row.model_copy(update={"model": PUBLIC_ASSISTANT_MODEL})


@router.get("/observability/phoenix", response_model=PhoenixOverviewResponse)
def phoenix_overview(
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> PhoenixOverviewResponse:
    overview = _phoenix_observability_service.build_overview()
    return _redact_phoenix_models(overview, is_admin=current_user.is_admin)


@router.get("/observability/phoenix/traces", response_model=PhoenixTracesResponse)
def phoenix_traces(
    current_user: Annotated[AuthUser, Depends(get_current_user)],
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> PhoenixTracesResponse:
    result = _phoenix_observability_service.get_traces(limit=limit, offset=offset)
    if not current_user.is_admin:
        result.traces = [
            _redact_trace_row(r, is_admin=False) for r in result.traces
        ]
    return result


@router.get(
    "/observability/phoenix/traces/by-session/{session_id}",
    response_model=PhoenixTraceDetailResponse,
)
def phoenix_trace_by_session(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> PhoenixTraceDetailResponse:
    return _phoenix_observability_service.get_session_spans(session_id)


@router.get(
    "/observability/phoenix/traces/{trace_id}",
    response_model=PhoenixTraceDetailResponse,
)
def phoenix_trace_detail(
    trace_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> PhoenixTraceDetailResponse:
    return _phoenix_observability_service.get_trace_spans(trace_id)


