from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ResolvedDBConnection:
    connection_id: str
    user_id: int
    name: str
    db_type: str
    host: str
    port: int | None
    database: str | None
    username: str | None
    password: str | None = field(repr=False)
    options: dict[str, Any]

    def to_runtime_payload(self) -> dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "name": self.name,
            "db_type": self.db_type,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "password": self.password,
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class CatalogSchema:
    name: str
    display_name: str


@dataclass(frozen=True)
class CatalogTable:
    schema: str
    name: str
    table_type: str
    qualified_name: str


@dataclass(frozen=True)
class CatalogColumn:
    schema: str
    table: str
    name: str
    data_type: str
    is_nullable: bool | None
    ordinal_position: int | None
    default_expression: str | None = None


class BaseConnectionAdapter(ABC):
    def __init__(self, resolved: ResolvedDBConnection, *, timeout_sec: int) -> None:
        self.resolved = resolved
        self.timeout_sec = timeout_sec

    @abstractmethod
    def test_connection(self) -> None:
        """Raise an exception if the connection cannot be established."""

    @abstractmethod
    def list_schemas(self) -> list[CatalogSchema]:
        """Return normalized schemas/databases visible to the connection."""

    @abstractmethod
    def list_tables(self, schema: str) -> list[CatalogTable]:
        """Return normalized tables/views visible in a schema/database."""

    @abstractmethod
    def describe_table(self, schema: str, table: str) -> list[CatalogColumn]:
        """Return normalized column metadata for a table/view."""


class PostgresConnectionAdapter(BaseConnectionAdapter):
    def _configured_schema(self) -> str | None:
        schema = self.resolved.options.get("schema")
        if not isinstance(schema, str):
            return None
        clean = schema.strip()
        return clean or None

    def _connect_kwargs(self) -> dict[str, Any]:
        connect_kwargs: dict[str, Any] = {
            "host": self.resolved.host,
            "port": self.resolved.port or 5432,
            "dbname": self.resolved.database,
            "user": self.resolved.username,
            "password": self.resolved.password,
            "connect_timeout": self.timeout_sec,
        }
        sslmode = self.resolved.options.get("sslmode")
        if isinstance(sslmode, str) and sslmode.strip():
            connect_kwargs["sslmode"] = sslmode.strip()
        return connect_kwargs

    def _apply_session_schema(self, conn: Any) -> None:
        schema = self._configured_schema()
        if not schema:
            return
        with conn.cursor() as cur:
            cur.execute("SELECT set_config('search_path', %s, false)", (schema,))

    def test_connection(self) -> None:
        import psycopg

        with psycopg.connect(**self._connect_kwargs()) as conn:
            self._apply_session_schema(conn)
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()

    def list_schemas(self) -> list[CatalogSchema]:
        import psycopg

        query = """
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
            ORDER BY schema_name
        """
        with psycopg.connect(**self._connect_kwargs()) as conn:
            self._apply_session_schema(conn)
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
        return [
            CatalogSchema(name=str(row[0]), display_name=str(row[0]))
            for row in rows
            if row and row[0] is not None
        ]

    def list_tables(self, schema: str) -> list[CatalogTable]:
        import psycopg

        query = """
            SELECT table_name, table_type
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_type IN ('BASE TABLE', 'VIEW', 'FOREIGN TABLE', 'LOCAL TEMPORARY')
            ORDER BY table_name
        """
        with psycopg.connect(**self._connect_kwargs()) as conn:
            self._apply_session_schema(conn)
            with conn.cursor() as cur:
                cur.execute(query, (schema,))
                rows = cur.fetchall()
        normalized: list[CatalogTable] = []
        for row in rows:
            if not row or row[0] is None:
                continue
            name = str(row[0])
            raw_type = str(row[1] or "").upper()
            table_type = "view" if "VIEW" in raw_type else "table"
            normalized.append(
                CatalogTable(
                    schema=schema,
                    name=name,
                    table_type=table_type,
                    qualified_name=f"{schema}.{name}",
                )
            )
        return normalized

    def describe_table(self, schema: str, table: str) -> list[CatalogColumn]:
        import psycopg

        query = """
            SELECT
                column_name,
                data_type,
                is_nullable,
                ordinal_position,
                column_default
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name = %s
            ORDER BY ordinal_position
        """
        with psycopg.connect(**self._connect_kwargs()) as conn:
            self._apply_session_schema(conn)
            with conn.cursor() as cur:
                cur.execute(query, (schema, table))
                rows = cur.fetchall()

        columns: list[CatalogColumn] = []
        for row in rows:
            if not row or row[0] is None:
                continue
            is_nullable_raw = str(row[2] or "").strip().upper()
            columns.append(
                CatalogColumn(
                    schema=schema,
                    table=table,
                    name=str(row[0]),
                    data_type=str(row[1] or ""),
                    is_nullable=True if is_nullable_raw == "YES" else False,
                    ordinal_position=int(row[3]) if row[3] is not None else None,
                    default_expression=str(row[4]) if row[4] is not None else None,
                )
            )
        return columns


class ClickHouseConnectionAdapter(BaseConnectionAdapter):
    def _effective_database(self) -> str | None:
        schema = self.resolved.options.get("schema")
        if isinstance(schema, str) and schema.strip():
            return schema.strip()
        database = self.resolved.database
        if isinstance(database, str) and database.strip():
            return database.strip()
        return None

    @staticmethod
    def _escape_literal(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    def _request(
        self,
        sql: str,
        *,
        database: str | None = None,
    ) -> dict[str, Any]:
        secure = bool(self.resolved.options.get("secure", False))
        scheme = "https" if secure else "http"
        port = self.resolved.port or (8443 if secure else 8123)
        params = {
            "database": database if database is not None else (self._effective_database() or ""),
            "user": self.resolved.username or "",
            "password": self.resolved.password or "",
            "default_format": "JSON",
        }
        query_string = urlencode(params)
        request = Request(
            url=f"{scheme}://{self.resolved.host}:{port}/?{query_string}",
            data=sql.encode("utf-8"),
            method="POST",
            headers={"Content-Type": "text/plain; charset=utf-8"},
        )
        with urlopen(request, timeout=self.timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict) or "data" not in payload:
            raise RuntimeError("ClickHouse returned an unexpected response.")
        return payload

    def test_connection(self) -> None:
        self._request("SELECT 1 FORMAT JSON")

    def list_schemas(self) -> list[CatalogSchema]:
        payload = self._request(
            """
            SELECT name
            FROM system.databases
            WHERE name NOT IN ('system', 'information_schema', 'INFORMATION_SCHEMA')
            ORDER BY name
            FORMAT JSON
            """
        )
        rows = payload.get("data", [])
        return [
            CatalogSchema(name=str(row["name"]), display_name=str(row["name"]))
            for row in rows
            if isinstance(row, dict) and row.get("name") is not None
        ]

    def list_tables(self, schema: str) -> list[CatalogTable]:
        escaped_schema = self._escape_literal(schema)
        payload = self._request(
            """
            SELECT
                database,
                name,
                engine
            FROM system.tables
            WHERE database = {schema:String}
            ORDER BY name
            FORMAT JSON
            """.replace("{schema:String}", f"'{escaped_schema}'")
        )
        rows = payload.get("data", [])
        normalized: list[CatalogTable] = []
        for row in rows:
            if not isinstance(row, dict) or row.get("name") is None:
                continue
            name = str(row["name"])
            engine = str(row.get("engine") or "")
            table_type = "view" if "view" in engine.lower() else "table"
            normalized.append(
                CatalogTable(
                    schema=str(row.get("database") or schema),
                    name=name,
                    table_type=table_type,
                    qualified_name=f"{str(row.get('database') or schema)}.{name}",
                )
            )
        return normalized

    def describe_table(self, schema: str, table: str) -> list[CatalogColumn]:
        escaped_schema = self._escape_literal(schema)
        escaped_table = self._escape_literal(table)
        payload = self._request(
            """
            SELECT
                database,
                table,
                name,
                type,
                default_kind,
                default_expression,
                position
            FROM system.columns
            WHERE database = {schema:String}
              AND table = {table:String}
            ORDER BY position
            FORMAT JSON
            """
            .replace("{schema:String}", f"'{escaped_schema}'")
            .replace("{table:String}", f"'{escaped_table}'")
        )
        rows = payload.get("data", [])
        columns: list[CatalogColumn] = []
        for row in rows:
            if not isinstance(row, dict) or row.get("name") is None:
                continue
            raw_type = str(row.get("type") or "")
            default_expression = row.get("default_expression")
            columns.append(
                CatalogColumn(
                    schema=str(row.get("database") or schema),
                    table=str(row.get("table") or table),
                    name=str(row["name"]),
                    data_type=raw_type,
                    is_nullable=True if raw_type.startswith("Nullable(") else False,
                    ordinal_position=int(row["position"]) if row.get("position") is not None else None,
                    default_expression=str(default_expression) if default_expression is not None else None,
                )
            )
        return columns


def build_connection_adapter(
    resolved: ResolvedDBConnection,
    *,
    timeout_sec: int,
) -> BaseConnectionAdapter:
    if resolved.db_type == "postgresql":
        return PostgresConnectionAdapter(resolved, timeout_sec=timeout_sec)
    if resolved.db_type == "clickhouse":
        return ClickHouseConnectionAdapter(resolved, timeout_sec=timeout_sec)
    raise RuntimeError(f"Unsupported db_type: {resolved.db_type}")


