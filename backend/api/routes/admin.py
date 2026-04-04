from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import get_admin_user
from backend.api.models import (
    AdminCreateUserRequest,
    AdminUpdateUserRequest,
    AuthUserResponse,
    MessageResponse,
)
from backend.auth.auth_db import AuthDB, AuthUser

router = APIRouter(tags=["Администрирование"])

# Singleton set during app startup
_auth_db: AuthDB = None  # type: ignore


def setup(auth_db: AuthDB) -> None:
    global _auth_db
    _auth_db = auth_db


def _to_user_response(user: AuthUser) -> AuthUserResponse:
    return AuthUserResponse(
        id=user.id,
        username=user.username,
        is_admin=user.is_admin,
        created_at=user.created_at,
    )


@router.get("/admin/users", response_model=list[AuthUserResponse])
def admin_list_users(
    _: Annotated[AuthUser, Depends(get_admin_user)],
) -> list[AuthUserResponse]:
    return [_to_user_response(user) for user in _auth_db.list_users()]


@router.post("/admin/users", response_model=AuthUserResponse)
def admin_create_user(
    payload: AdminCreateUserRequest,
    _: Annotated[AuthUser, Depends(get_admin_user)],
) -> AuthUserResponse:
    try:
        created = _auth_db.create_user(
            payload.username,
            payload.password,
            is_admin=payload.is_admin,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(status_code=409, detail="Username already exists") from None
    return _to_user_response(created)


@router.patch("/admin/users/{user_id}", response_model=AuthUserResponse)
def admin_update_user(
    user_id: int,
    payload: AdminUpdateUserRequest,
    current_admin: Annotated[AuthUser, Depends(get_admin_user)],
) -> AuthUserResponse:
    if payload.is_admin is False and user_id == current_admin.id:
        raise HTTPException(
            status_code=400,
            detail="Нельзя снять роль администратора у текущего пользователя",
        )

    if payload.password is not None:
        try:
            updated = _auth_db.set_user_password(user_id, payload.password)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not updated:
            raise HTTPException(status_code=404, detail="User not found")

    if payload.is_admin is not None:
        try:
            updated = _auth_db.set_user_admin(user_id, payload.is_admin)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not updated:
            raise HTTPException(status_code=404, detail="User not found")

    user = _auth_db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return _to_user_response(user)


@router.delete("/admin/users/{user_id}", response_model=MessageResponse)
def admin_delete_user(
    user_id: int,
    current_admin: Annotated[AuthUser, Depends(get_admin_user)],
) -> MessageResponse:
    if user_id == current_admin.id:
        raise HTTPException(
            status_code=400,
            detail="Нельзя удалить текущего пользователя",
        )
    try:
        deleted = _auth_db.delete_user(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")
    return MessageResponse(message="User deleted")


