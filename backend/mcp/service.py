from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from backend.core.config import Settings
from backend.data_access.crypto_service import SecretCryptoService
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
    crypto: Any | None = None
    provider: Any
    static_configs: list[MCPServerConfig] = Field(default_factory=list)

    def __init__(
        self,
        *,
        auth_db: Any | None = None,
        provider: MCPToolProvider | None = None,
        configs: Iterable[MCPServerConfig] | None = None,
        settings: Settings | None = None,
    ) -> None:
        from backend.mcp.transport import SDKMCPToolProvider

        super().__init__(
            auth_db=auth_db,
            crypto=SecretCryptoService(settings or Settings()) if auth_db is not None else None,
            provider=provider or SDKMCPToolProvider(),
            static_configs=list(configs or []),
        )

    def list_configs(self) -> list[MCPServerConfig]:
        if self.auth_db is not None:
            return [self._with_secret(config) for config in self.auth_db.list_mcp_server_configs()]
        return list(self.static_configs)

    def get_config(self, server_id: str) -> MCPServerConfig | None:
        clean_server_id = str(server_id or "").strip()
        if not clean_server_id:
            return None
        if self.auth_db is not None:
            config = self.auth_db.get_mcp_server_config(clean_server_id)
            return self._with_secret(config) if config is not None else None
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
        token = str(payload.bearer_token or "").strip() or None
        if self.auth_db is not None:
            encrypted = self.crypto.encrypt_payload({"bearer_token": token}) if token else None
            saved = self.auth_db.upsert_mcp_server_config(payload, updated_by=updated_by)
            if token is None:
                self.auth_db.clear_mcp_server_secret(saved.server_id)
            else:
                self.auth_db.set_mcp_server_secret(saved.server_id, encrypted)
            return saved.model_copy(update={"bearer_token": token})
        data = payload.model_dump()
        data["bearer_token"] = token
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
        data["bearer_token"] = current.bearer_token
        for field, value in payload.model_dump(exclude_unset=True).items():
            data[field] = value
        updated_config = MCPServerConfig(**data)
        return self.upsert_config(updated_config, updated_by=updated_by)

    def test_connection(
        self,
        server_id: str,
        payload: MCPServerUpdateRequest,
    ) -> int | None:
        current = self.get_config(server_id)
        if current is None:
            return None
        data = current.model_dump() | payload.model_dump(exclude_unset=True)
        data["bearer_token"] = (
            payload.bearer_token
            if "bearer_token" in payload.model_fields_set
            else current.bearer_token
        )
        config = MCPServerConfig(**data)
        return len(self.provider.list_tools(config))

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

    def _with_secret(self, config: MCPServerConfig) -> MCPServerConfig:
        encrypted = self.auth_db.get_mcp_server_secret_blob(config.server_id)
        if not encrypted:
            return config
        payload = self.crypto.decrypt_payload(encrypted)
        token = str(payload.get("bearer_token") or "").strip() or None
        return config.model_copy(update={"bearer_token": token})

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
            tools = [
                self._apply_binding(config, descriptor)
                for descriptor in self.provider.list_tools(config)
            ]
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

    @staticmethod
    def _apply_binding(
        config: MCPServerConfig,
        descriptor: MCPToolDescriptor,
    ) -> MCPToolDescriptor:
        binding = config.tool_bindings.get(descriptor.tool_name)
        if binding is None:
            return descriptor
        return descriptor.model_copy(
            update={
                "capability_key": binding.capability_key,
                "provider_identity": binding.provider_identity or config.server_id,
                "binding_priority": binding.priority,
                "binding_preferred": binding.preferred,
            }
        )
