from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from backend.mcp.models import (
    MCPServerCatalogItem,
    MCPServerConfig,
    MCPServerCreateRequest,
    MCPServerUpdateRequest,
    MCPToolDescriptor,
)


class MCPToolProvider(Protocol):
    def list_tools(self, config: MCPServerConfig) -> list[MCPToolDescriptor]: ...

    def call_tool(
        self,
        *,
        config: MCPServerConfig,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any: ...


class MCPServerService(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    auth_db: Any | None = None
    provider: Any
    static_configs: list[MCPServerConfig] = Field(default_factory=list)

    def __init__(
        self,
        *,
        auth_db: Any | None = None,
        provider: MCPToolProvider | None = None,
        configs: Iterable[MCPServerConfig] | None = None,
    ) -> None:
        from backend.mcp.transport import SDKMCPToolProvider

        super().__init__(
            auth_db=auth_db,
            provider=provider or SDKMCPToolProvider(),
            static_configs=list(configs or []),
        )

    def list_configs(self) -> list[MCPServerConfig]:
        if self.auth_db is not None:
            return self.auth_db.list_mcp_server_configs()
        return list(self.static_configs)

    def get_config(self, server_id: str) -> MCPServerConfig | None:
        clean_server_id = str(server_id or "").strip()
        if not clean_server_id:
            return None
        if self.auth_db is not None:
            return self.auth_db.get_mcp_server_config(clean_server_id)
        for config in self.static_configs:
            if config.server_id == clean_server_id:
                return config
        return None

    def upsert_config(
        self,
        payload: MCPServerCreateRequest | MCPServerConfig,
        *,
        updated_by: int,
    ) -> MCPServerConfig:
        if self.auth_db is not None:
            return self.auth_db.upsert_mcp_server_config(payload, updated_by=updated_by)
        data = payload.model_dump()
        data["updated_by"] = updated_by
        config = MCPServerConfig(**data)
        self.static_configs = [
            existing for existing in self.static_configs if existing.server_id != config.server_id
        ]
        self.static_configs.append(config)
        return config

    def update_config(
        self,
        server_id: str,
        payload: MCPServerUpdateRequest,
        *,
        updated_by: int,
    ) -> MCPServerConfig | None:
        current = self.get_config(server_id)
        if current is None:
            return None
        data = current.model_dump()
        for field, value in payload.model_dump(exclude_unset=True).items():
            data[field] = value
        updated_config = MCPServerConfig(**data)
        return self.upsert_config(updated_config, updated_by=updated_by)

    def delete_config(self, server_id: str) -> bool:
        clean_server_id = str(server_id or "").strip()
        if not clean_server_id:
            return False
        if self.auth_db is not None:
            return self.auth_db.delete_mcp_server_config(clean_server_id)
        before = len(self.static_configs)
        self.static_configs = [
            existing for existing in self.static_configs if existing.server_id != clean_server_id
        ]
        return len(self.static_configs) != before

    def list_catalog(
        self,
        *,
        user_settings: Mapping[str, bool] | None = None,
    ) -> list[MCPServerCatalogItem]:
        settings = dict(user_settings or {})
        return [self._catalog_item(config, settings) for config in self.list_configs()]

    def get_catalog_item(
        self,
        server_id: str,
        *,
        user_settings: Mapping[str, bool] | None = None,
    ) -> MCPServerCatalogItem | None:
        clean_server_id = str(server_id or "").strip()
        for item in self.list_catalog(user_settings=user_settings):
            if item.server_id == clean_server_id:
                return item
        return None

    def enabled_tool_descriptors(
        self,
        *,
        user_settings: Mapping[str, bool] | None = None,
    ) -> list[MCPToolDescriptor]:
        descriptors: list[MCPToolDescriptor] = []
        for item in self.list_catalog(user_settings=user_settings):
            if item.effective_enabled:
                descriptors.extend(item.tools)
        return descriptors

    def effective_enabled_tool_keys(
        self,
        *,
        user_settings: Mapping[str, bool] | None = None,
    ) -> set[str]:
        return {
            descriptor.tool_key
            for descriptor in self.enabled_tool_descriptors(user_settings=user_settings)
        }

    def configs_by_id(self) -> dict[str, MCPServerConfig]:
        return {config.server_id: config for config in self.list_configs()}

    def _catalog_item(
        self,
        config: MCPServerConfig,
        user_settings: Mapping[str, bool],
    ) -> MCPServerCatalogItem:
        enabled_for_user = bool(
            user_settings.get(config.server_id, config.enabled_by_default)
        )
        if not config.enabled:
            return MCPServerCatalogItem(
                server_id=config.server_id,
                name=config.display_label,
                description=config.description,
                transport=config.transport,
                enabled_globally=False,
                available_globally=False,
                status="disabled",
                enabled_by_default=config.enabled_by_default,
                enabled_for_user=enabled_for_user,
                effective_enabled=False,
                tools=[],
                tool_count=0,
            )
        if not enabled_for_user:
            return MCPServerCatalogItem(
                server_id=config.server_id,
                name=config.display_label,
                description=config.description,
                transport=config.transport,
                enabled_globally=True,
                available_globally=True,
                status="disabled_for_user",
                enabled_by_default=config.enabled_by_default,
                enabled_for_user=False,
                effective_enabled=False,
                tools=[],
                tool_count=0,
            )
        try:
            tools = self.provider.list_tools(config)
        except Exception as exc:
            return MCPServerCatalogItem(
                server_id=config.server_id,
                name=config.display_label,
                description=config.description,
                transport=config.transport,
                enabled_globally=True,
                available_globally=False,
                status="unavailable",
                enabled_by_default=config.enabled_by_default,
                enabled_for_user=enabled_for_user,
                effective_enabled=False,
                tools=[],
                tool_count=0,
                last_error=str(exc),
            )
        return MCPServerCatalogItem(
            server_id=config.server_id,
            name=config.display_label,
            description=config.description,
            transport=config.transport,
            enabled_globally=True,
            available_globally=True,
            status="available",
            enabled_by_default=config.enabled_by_default,
            enabled_for_user=enabled_for_user,
            effective_enabled=enabled_for_user,
            tools=tools,
            tool_count=len(tools),
        )
