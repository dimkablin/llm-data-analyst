from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import get_admin_user, get_current_user
from backend.api.models import MessageResponse
from backend.auth.auth_db import AuthDB, AuthUser
from backend.mcp.models import (
    AdminMCPServerConfigResponse,
    MCPServerCatalogItem,
    MCPServerCreateRequest,
    MCPServerEnabledUpdateRequest,
    MCPServerUpdateRequest,
)
from backend.mcp.service import MCPServerService

router = APIRouter(tags=["MCP"])

_auth_db: AuthDB = None  # type: ignore
_mcp_service: MCPServerService = None  # type: ignore


def setup(auth_db: AuthDB, mcp_service: MCPServerService) -> None:
    global _auth_db, _mcp_service
    _auth_db = auth_db
    _mcp_service = mcp_service


def _admin_response(config) -> AdminMCPServerConfigResponse:
    return AdminMCPServerConfigResponse(
        **config.model_dump(),
        secret_configured=bool(config.bearer_token),
    )


@router.get("/mcp/servers", response_model=list[MCPServerCatalogItem])
def list_mcp_servers(
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[MCPServerCatalogItem]:
    return _mcp_service.list_catalog(
        user_settings=_auth_db.list_user_mcp_server_settings(current_user.id),
    )


@router.patch("/mcp/servers/{server_id}", response_model=MCPServerCatalogItem)
def update_mcp_server_enabled(
    server_id: str,
    payload: MCPServerEnabledUpdateRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> MCPServerCatalogItem:
    clean_server_id = str(server_id or "").strip()
    current = _mcp_service.get_catalog_item(
        clean_server_id,
        user_settings=_auth_db.list_user_mcp_server_settings(current_user.id),
    )
    if current is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    _auth_db.set_user_mcp_server_enabled(
        current_user.id,
        clean_server_id,
        payload.enabled,
    )
    updated = _mcp_service.get_catalog_item(
        clean_server_id,
        user_settings=_auth_db.list_user_mcp_server_settings(current_user.id),
    )
    if updated is None:
        raise HTTPException(status_code=500, detail="MCP server state was not persisted")
    return updated


@router.get("/admin/mcp/servers", response_model=list[AdminMCPServerConfigResponse])
def admin_list_mcp_servers(
    _: Annotated[AuthUser, Depends(get_admin_user)],
) -> list[AdminMCPServerConfigResponse]:
    return [_admin_response(config) for config in _mcp_service.list_configs()]


@router.post("/admin/mcp/servers", response_model=AdminMCPServerConfigResponse)
def admin_upsert_mcp_server(
    payload: MCPServerCreateRequest,
    current_admin: Annotated[AuthUser, Depends(get_admin_user)],
) -> AdminMCPServerConfigResponse:
    saved = _mcp_service.upsert_config(payload, updated_by=current_admin.id)
    return _admin_response(saved)


@router.patch("/admin/mcp/servers/{server_id}", response_model=AdminMCPServerConfigResponse)
def admin_update_mcp_server(
    server_id: str,
    payload: MCPServerUpdateRequest,
    current_admin: Annotated[AuthUser, Depends(get_admin_user)],
) -> AdminMCPServerConfigResponse:
    updated = _mcp_service.update_config(
        server_id,
        payload,
        updated_by=current_admin.id,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return _admin_response(updated)


@router.post("/admin/mcp/servers/{server_id}/test", response_model=MessageResponse)
def admin_test_mcp_server(
    server_id: str,
    payload: MCPServerUpdateRequest,
    _: Annotated[AuthUser, Depends(get_admin_user)],
) -> MessageResponse:
    try:
        tool_count = _mcp_service.test_connection(server_id, payload)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MCP connection failed: {exc}") from exc
    if tool_count is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return MessageResponse(message=f"MCP server connected; tools: {tool_count}")


@router.delete("/admin/mcp/servers/{server_id}", response_model=MessageResponse)
def admin_delete_mcp_server(
    server_id: str,
    _: Annotated[AuthUser, Depends(get_admin_user)],
) -> MessageResponse:
    deleted = _mcp_service.delete_config(server_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return MessageResponse(message="MCP server deleted")
