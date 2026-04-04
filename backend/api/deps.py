from __future__ import annotations

from fastapi import Depends, Header, HTTPException

from backend.auth.auth_db import AuthUser

# This will be set by api/app.py during initialization
_auth_db = None


def set_auth_db(db) -> None:
    global _auth_db
    _auth_db = db


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.strip().partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _require_token(authorization: str | None = Header(default=None)) -> str:
    token = _extract_bearer_token(authorization)
    if token is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return token


def get_current_user(token: str = Depends(_require_token)) -> AuthUser:
    user = _auth_db.get_user_by_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


def get_admin_user(current_user: AuthUser = Depends(get_current_user)) -> AuthUser:
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


