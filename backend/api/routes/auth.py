from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import _require_token, get_current_user
from backend.api.models import (
    AuthChangePasswordRequest,
    AuthLoginRequest,
    AuthRegisterRequest,
    AuthResponse,
    AuthUserResponse,
    MessageResponse,
    ToolAvailabilityResponse,
    ToolEnabledUpdateRequest,
    UserMemoryResponse,
    UserMemoryUpdateRequest,
    UserSettingsResponse,
    UserSettingsUpdateRequest,
)
from backend.auth.auth_db import AuthDB, AuthUser

router = APIRouter(tags=["Аутентификация"])

# Singletons set during app startup
_auth_db: AuthDB = None  # type: ignore
_user_memory_service = None  # type: ignore
_tool_catalog_response_fn = None  # type: ignore
_tool_catalog_payload_fn = None  # type: ignore
_known_tool_keys = None  # type: ignore


def setup(
    auth_db: AuthDB,
    user_memory_service,
    tool_catalog_response_fn,
    tool_catalog_payload_fn,
    known_tool_keys,
) -> None:
    global _auth_db, _user_memory_service, _tool_catalog_response_fn
    global _tool_catalog_payload_fn, _known_tool_keys
    _auth_db = auth_db
    _user_memory_service = user_memory_service
    _tool_catalog_response_fn = tool_catalog_response_fn
    _tool_catalog_payload_fn = tool_catalog_payload_fn
    _known_tool_keys = known_tool_keys


def _to_user_response(user: AuthUser) -> AuthUserResponse:
    return AuthUserResponse(
        id=user.id,
        username=user.username,
        is_admin=user.is_admin,
        created_at=user.created_at,
    )


def _to_settings_response(user_id: int) -> UserSettingsResponse:
    settings_row = _auth_db.get_user_settings(user_id)
    return UserSettingsResponse(
        theme=settings_row.theme,
        default_include_reasoning=settings_row.default_include_reasoning,
        default_answer_style=settings_row.default_answer_style,
        analysis_depth=settings_row.analysis_depth,
        llm_temperature_chat=settings_row.llm_temperature_chat,
        llm_temperature_tool=settings_row.llm_temperature_tool,
        llm_max_tokens_default=settings_row.llm_max_tokens_default,
        llm_max_tokens_reasoning=settings_row.llm_max_tokens_reasoning,
        backend_query_timeout_sec=settings_row.backend_query_timeout_sec,
        agent_max_steps=settings_row.agent_max_steps,
        agent_step_timeout_sec=settings_row.agent_step_timeout_sec,
        agent_inner_recursion_limit=settings_row.agent_inner_recursion_limit,
        ui_scale=settings_row.ui_scale,
    )


@router.post("/auth/register", response_model=AuthResponse)
def auth_register(payload: AuthRegisterRequest) -> AuthResponse:
    try:
        user = _auth_db.create_user(payload.username, payload.password, is_admin=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(status_code=409, detail="Username already exists") from None

    token = _auth_db.create_access_token(user.id)
    return AuthResponse(access_token=token, user=_to_user_response(user))


@router.post("/auth/login", response_model=AuthResponse)
def auth_login(payload: AuthLoginRequest) -> AuthResponse:
    user = _auth_db.authenticate(payload.username, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = _auth_db.create_access_token(user.id)
    return AuthResponse(access_token=token, user=_to_user_response(user))


@router.get("/auth/me", response_model=AuthUserResponse)
def auth_me(current_user: Annotated[AuthUser, Depends(get_current_user)]) -> AuthUserResponse:
    return _to_user_response(current_user)


@router.post("/auth/change-password", response_model=MessageResponse)
def auth_change_password(
    payload: AuthChangePasswordRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> MessageResponse:
    try:
        _auth_db.update_password_with_current(
            current_user.id,
            payload.current_password,
            payload.new_password,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return MessageResponse(message="Password updated")


@router.get("/auth/settings", response_model=UserSettingsResponse)
def auth_get_settings(
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> UserSettingsResponse:
    return _to_settings_response(current_user.id)


@router.patch("/auth/settings", response_model=UserSettingsResponse)
def auth_update_settings(
    payload: UserSettingsUpdateRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> UserSettingsResponse:
    try:
        updated = _auth_db.update_user_settings(
            current_user.id,
            theme=payload.theme,
            default_include_reasoning=payload.default_include_reasoning,
            default_answer_style=payload.default_answer_style,
            analysis_depth=payload.analysis_depth,
            llm_temperature_chat=payload.llm_temperature_chat,
            llm_temperature_tool=payload.llm_temperature_tool,
            llm_max_tokens_default=payload.llm_max_tokens_default,
            llm_max_tokens_reasoning=payload.llm_max_tokens_reasoning,
            backend_query_timeout_sec=payload.backend_query_timeout_sec,
            agent_max_steps=payload.agent_max_steps,
            agent_step_timeout_sec=payload.agent_step_timeout_sec,
            agent_inner_recursion_limit=payload.agent_inner_recursion_limit,
            ui_scale=payload.ui_scale,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return UserSettingsResponse(
        theme=updated.theme,
        default_include_reasoning=updated.default_include_reasoning,
        default_answer_style=updated.default_answer_style,
        analysis_depth=updated.analysis_depth,
        llm_temperature_chat=updated.llm_temperature_chat,
        llm_temperature_tool=updated.llm_temperature_tool,
        llm_max_tokens_default=updated.llm_max_tokens_default,
        llm_max_tokens_reasoning=updated.llm_max_tokens_reasoning,
        backend_query_timeout_sec=updated.backend_query_timeout_sec,
        agent_max_steps=updated.agent_max_steps,
        agent_step_timeout_sec=updated.agent_step_timeout_sec,
        agent_inner_recursion_limit=updated.agent_inner_recursion_limit,
        ui_scale=updated.ui_scale,
    )


@router.get("/auth/memory", response_model=UserMemoryResponse)
def auth_get_memory(
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> UserMemoryResponse:
    mem = _user_memory_service.load(current_user.id)
    return UserMemoryResponse(profile=mem.profile, notes=mem.notes)


@router.patch("/auth/memory", response_model=UserMemoryResponse)
def auth_update_memory(
    payload: UserMemoryUpdateRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> UserMemoryResponse:
    if payload.profile is not None:
        _user_memory_service.set_profile(current_user.id, payload.profile)
    if payload.notes is not None:
        _user_memory_service.set_notes(current_user.id, payload.notes)
    mem = _user_memory_service.load(current_user.id)
    return UserMemoryResponse(profile=mem.profile, notes=mem.notes)


@router.get("/auth/tools", response_model=list[ToolAvailabilityResponse])
def auth_get_tools(
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[ToolAvailabilityResponse]:
    return _tool_catalog_response_fn(current_user.id)


@router.patch("/auth/tools/{tool_key}", response_model=ToolAvailabilityResponse)
def auth_update_tool(
    tool_key: str,
    payload: ToolEnabledUpdateRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> ToolAvailabilityResponse:
    clean_tool_key = str(tool_key or "").strip()
    if clean_tool_key not in _known_tool_keys:
        raise HTTPException(status_code=404, detail="Tool not found")
    _auth_db.set_user_tool_enabled(current_user.id, clean_tool_key, payload.enabled)
    rows = _tool_catalog_response_fn(current_user.id)
    for item in rows:
        if item.tool_key == clean_tool_key:
            return item
    raise HTTPException(status_code=500, detail="Tool state was not persisted")


@router.post("/auth/logout", response_model=MessageResponse)
def auth_logout(token: Annotated[str, Depends(_require_token)]) -> MessageResponse:
    _auth_db.revoke_token(token)
    return MessageResponse(message="Logged out")


