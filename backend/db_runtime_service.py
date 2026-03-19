from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlencode

from backend.db_connections_service import DBConnectionsService
from backend.db_connectors import (
    CatalogSchema,
    CatalogTable,
    ResolvedDBConnection,
    build_connection_adapter,
)
from backend.redaction import sanitize_error_text


@dataclass(frozen=True)
class RuntimeDBConnectionConfig:
    connection_id: str
    user_id: int
    name: str
    db_type: str
    host: str
    port: int | None
    database: str | None
    username: str | None
    password: str | None = field(repr=False, default=None)
    options: dict[str, Any] = field(default_factory=dict)

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "name": self.name,
            "db_type": self.db_type,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "password_present": self.password is not None,
            "options": dict(self.options),
        }

    def to_driver_kwargs(self) -> dict[str, Any]:
        payload = {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "password": self.password,
            "options": dict(self.options),
        }
        return {key: value for key, value in payload.items() if value is not None}

    def build_dsn(self) -> str:
        if self.db_type == "postgresql":
            userinfo = ""
            if self.username:
                userinfo = quote(self.username, safe="")
                if self.password:
                    userinfo += f":{quote(self.password, safe='')}"
                userinfo += "@"
            host = self.host
            port = f":{self.port}" if self.port else ""
            database = f"/{quote(self.database, safe='')}" if self.database else ""
            query_params: dict[str, str] = {}
            sslmode = self.options.get("sslmode")
            if isinstance(sslmode, str) and sslmode.strip():
                query_params["sslmode"] = sslmode.strip()
            schema = self.options.get("schema")
            if isinstance(schema, str) and schema.strip():
                query_params["options"] = f"-c search_path={schema.strip()}"
            query = f"?{urlencode(query_params)}" if query_params else ""
            return f"postgresql://{userinfo}{host}{port}{database}{query}"

        if self.db_type == "clickhouse":
            secure = bool(self.options.get("secure", False))
            scheme = "https" if secure else "http"
            userinfo = ""
            if self.username:
                userinfo = quote(self.username, safe="")
                if self.password:
                    userinfo += f":{quote(self.password, safe='')}"
                userinfo += "@"
            host = self.host
            port = f":{self.port}" if self.port else ""
            effective_database = self.options.get("schema")
            if not isinstance(effective_database, str) or not effective_database.strip():
                effective_database = self.database
            database = f"/{quote(str(effective_database), safe='')}" if effective_database else ""
            query_params = {
                key: value
                for key, value in self.options.items()
                if key != "secure" and value is not None
            }
            query = f"?{urlencode(query_params)}" if query_params else ""
            return f"{scheme}://{userinfo}{host}{port}{database}{query}"

        raise RuntimeError(f"Unsupported db_type for DSN builder: {self.db_type}")


@dataclass(frozen=True)
class RuntimeCatalogSchema:
    name: str
    display_name: str


@dataclass(frozen=True)
class RuntimeCatalogTable:
    schema: str
    name: str
    table_type: str
    qualified_name: str


class DBRuntimeService:
    def __init__(self, connections_service: DBConnectionsService) -> None:
        self.connections_service = connections_service

    @staticmethod
    def _from_resolved(resolved: ResolvedDBConnection) -> RuntimeDBConnectionConfig:
        return RuntimeDBConnectionConfig(
            connection_id=resolved.connection_id,
            user_id=resolved.user_id,
            name=resolved.name,
            db_type=resolved.db_type,
            host=resolved.host,
            port=resolved.port,
            database=resolved.database,
            username=resolved.username,
            password=resolved.password,
            options=dict(resolved.options),
        )

    def get_runtime_config(
        self,
        *,
        user_id: int,
        connection_id: str,
    ) -> RuntimeDBConnectionConfig:
        resolved = self.connections_service.resolve_connection_for_runtime(
            user_id, connection_id
        )
        return self._from_resolved(resolved)

    def _build_adapter(
        self,
        *,
        user_id: int,
        connection_id: str,
    ):
        resolved = self.connections_service.resolve_connection_for_runtime(
            user_id, connection_id
        )
        adapter = build_connection_adapter(
            resolved,
            timeout_sec=self.connections_service.settings.db_connections_test_timeout_sec,
        )
        return self._from_resolved(resolved), adapter

    @staticmethod
    def _normalize_schema_items(items: list[CatalogSchema]) -> list[RuntimeCatalogSchema]:
        return [
            RuntimeCatalogSchema(
                name=item.name,
                display_name=item.display_name,
            )
            for item in items
        ]

    @staticmethod
    def _normalize_table_items(items: list[CatalogTable]) -> list[RuntimeCatalogTable]:
        return [
            RuntimeCatalogTable(
                schema=item.schema,
                name=item.name,
                table_type=item.table_type,
                qualified_name=item.qualified_name,
            )
            for item in items
        ]

    def list_schemas(
        self,
        *,
        user_id: int,
        connection_id: str,
    ) -> list[RuntimeCatalogSchema]:
        _, adapter = self._build_adapter(user_id=user_id, connection_id=connection_id)
        try:
            schemas = adapter.list_schemas()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(sanitize_error_text(str(exc))) from exc
        return self._normalize_schema_items(schemas)

    def list_tables(
        self,
        *,
        user_id: int,
        connection_id: str,
        schema: str,
    ) -> list[RuntimeCatalogTable]:
        clean_schema = str(schema or "").strip()
        if not clean_schema:
            raise ValueError("schema is required")
        _, adapter = self._build_adapter(user_id=user_id, connection_id=connection_id)
        try:
            tables = adapter.list_tables(clean_schema)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(sanitize_error_text(str(exc))) from exc
        return self._normalize_table_items(tables)

    def build_demo_tool_contract(
        self,
        *,
        user_id: int,
        connection_id: str,
    ) -> dict[str, Any]:
        runtime = self.get_runtime_config(user_id=user_id, connection_id=connection_id)
        return {
            "runtime_config": runtime,
            "driver_kwargs": runtime.to_driver_kwargs(),
            "dsn": runtime.build_dsn(),
            "schemas": self.list_schemas(user_id=user_id, connection_id=connection_id),
            "safe_log_context": runtime.to_safe_dict(),
        }
