from __future__ import annotations

from fastapi import HTTPException

from backend.api.routes import mcp_servers as mcp_servers_route
from backend.auth.auth_db import AuthUser
from backend.mcp.models import (
    MCPServerCatalogItem,
    MCPServerConfig,
    MCPServerCreateRequest,
    MCPServerTransport,
)


class _FakeAuthDB:
    def __init__(self) -> None:
        self.user_settings: dict[str, bool] = {}

    def list_user_mcp_server_settings(self, user_id: int) -> dict[str, bool]:
        assert user_id == 7
        return dict(self.user_settings)

    def set_user_mcp_server_enabled(self, user_id: int, server_id: str, enabled: bool) -> None:
        assert user_id == 7
        self.user_settings[server_id] = enabled


class _FakeMCPService:
    def __init__(self) -> None:
        self.created_by: int | None = None
        self.deleted: str | None = None
        self.catalog = [
            MCPServerCatalogItem(
                server_id="finance-research",
                name="Finance Research",
                description=None,
                transport=MCPServerTransport.streamable_http,
                enabled_globally=True,
                available_globally=True,
                status="available",
                enabled_by_default=True,
                enabled_for_user=True,
                effective_enabled=True,
                tools=[],
                tool_count=0,
                last_error=None,
            )
        ]

    def list_catalog(self, user_settings: dict[str, bool]) -> list[MCPServerCatalogItem]:
        return [
            item.model_copy(
                update={
                    "enabled_for_user": user_settings.get(
                        item.server_id,
                        item.enabled_by_default,
                    )
                }
            )
            for item in self.catalog
        ]

    def get_catalog_item(
        self,
        server_id: str,
        user_settings: dict[str, bool],
    ) -> MCPServerCatalogItem | None:
        for item in self.list_catalog(user_settings):
            if item.server_id == server_id:
                return item
        return None

    def list_configs(self) -> list[MCPServerConfig]:
        return [
            MCPServerConfig(
                server_id="finance-research",
                name="Finance Research",
                transport=MCPServerTransport.streamable_http,
                url="http://127.0.0.1:8765/mcp",
            )
        ]

    def upsert_config(
        self,
        payload: MCPServerCreateRequest,
        *,
        updated_by: int,
    ) -> MCPServerConfig:
        self.created_by = updated_by
        return MCPServerConfig(**payload.model_dump(), updated_by=updated_by)

    def delete_config(self, server_id: str) -> bool:
        self.deleted = server_id
        return server_id == "finance-research"


def _user(*, is_admin: bool = False) -> AuthUser:
    return AuthUser(id=7, username="analyst", is_admin=is_admin, created_at="now")


def test_user_mcp_server_route_lists_and_updates_own_toggle() -> None:
    auth_db = _FakeAuthDB()
    service = _FakeMCPService()
    mcp_servers_route.setup(auth_db=auth_db, mcp_service=service)

    [initial] = mcp_servers_route.list_mcp_servers(_user())
    updated = mcp_servers_route.update_mcp_server_enabled(
        "finance-research",
        mcp_servers_route.MCPServerEnabledUpdateRequest(enabled=False),
        _user(),
    )

    assert initial.server_id == "finance-research"
    assert updated.enabled_for_user is False
    assert auth_db.user_settings == {"finance-research": False}


def test_user_mcp_server_route_rejects_unknown_server_toggle() -> None:
    mcp_servers_route.setup(auth_db=_FakeAuthDB(), mcp_service=_FakeMCPService())

    try:
        mcp_servers_route.update_mcp_server_enabled(
            "missing",
            mcp_servers_route.MCPServerEnabledUpdateRequest(enabled=True),
            _user(),
        )
    except HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("unknown MCP server must return 404")


def test_admin_mcp_server_route_can_create_and_delete_configs() -> None:
    service = _FakeMCPService()
    mcp_servers_route.setup(auth_db=_FakeAuthDB(), mcp_service=service)

    created = mcp_servers_route.admin_upsert_mcp_server(
        MCPServerCreateRequest(
            server_id="finance-research",
            name="Finance Research",
            transport=MCPServerTransport.streamable_http,
            url="http://127.0.0.1:8765/mcp",
        ),
        _user(is_admin=True),
    )
    deleted = mcp_servers_route.admin_delete_mcp_server(
        "finance-research",
        _user(is_admin=True),
    )

    assert created.updated_by == 7
    assert service.created_by == 7
    assert deleted.ok is True
    assert service.deleted == "finance-research"
