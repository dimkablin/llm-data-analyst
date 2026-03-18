from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import UUID

import pandas as pd

from agent.prompts import db_tool_prompt
from agent.tools.base_tool import BaseExecTool
from backend.db_runtime_service import RuntimeDBConnectionConfig


BASE_FORBIDDEN_CODE_PATTERNS: tuple[tuple[str, str], ...] = tuple(
    BaseExecTool.model_fields["forbidden_code_patterns"].default
)
READ_ONLY_SQL_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
FORBIDDEN_SQL_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|attach|detach|optimize|system|rename)\b",
    re.IGNORECASE,
)
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
LOW_PRIORITY_TABLE_RE = re.compile(
    r"(alembic|migration|checkpoint|cron|store|ttl|user_db_connection|schema_migrations|assistant_version|run_event|^run$|^runs$)",
    re.IGNORECASE,
)
HIGH_PRIORITY_TABLE_RE = re.compile(
    r"(chat|message|order|sale|customer|invoice|payment|product|report|event)",
    re.IGNORECASE,
)


def _strip_sql(sql: str) -> str:
    return str(sql or "").strip().rstrip(";").strip()


def _assert_read_only_sql(sql: str) -> str:
    clean = _strip_sql(sql)
    if not clean:
        raise ValueError("SQL query is empty.")
    if not READ_ONLY_SQL_RE.match(clean):
        raise ValueError("Only read-only SELECT/WITH queries are allowed in db_tool.")
    if FORBIDDEN_SQL_RE.search(clean):
        raise ValueError("Non-read-only SQL keywords are not allowed in db_tool.")
    if ";" in clean:
        raise ValueError("Multiple SQL statements are not allowed in db_tool.")
    return clean


def _make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    return value


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    safe_df = df.copy()
    for column in safe_df.columns:
        safe_df[column] = safe_df[column].map(_make_json_safe)
    return safe_df


@dataclass
class DBDemoHelper:
    runtime: RuntimeDBConnectionConfig
    timeout_sec: float = 10.0

    def _postgres_connect_kwargs(self) -> dict[str, Any]:
        kwargs = self.runtime.to_driver_kwargs()
        connect_kwargs: dict[str, Any] = {
            "host": kwargs["host"],
            "port": kwargs.get("port", 5432),
            "dbname": kwargs.get("database"),
            "user": kwargs.get("username"),
            "password": kwargs.get("password"),
            "connect_timeout": max(1, int(self.timeout_sec)),
        }
        sslmode = kwargs.get("options", {}).get("sslmode")
        if isinstance(sslmode, str) and sslmode.strip():
            connect_kwargs["sslmode"] = sslmode.strip()
        return connect_kwargs

    def _quote_identifier(self, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("Identifier must not be empty.")
        if not IDENTIFIER_RE.match(clean):
            raise ValueError(f"Unsafe identifier: {clean}")
        if self.runtime.db_type == "clickhouse":
            return f"`{clean}`"
        return f'"{clean}"'

    def _default_schema(self) -> str:
        if self.runtime.db_type == "clickhouse":
            return str(self.runtime.database or "default")
        return "public"

    def _postgres_query_dataframe(
        self,
        sql: str,
        params: tuple[Any, ...] | None = None,
    ) -> pd.DataFrame:
        import psycopg

        with psycopg.connect(**self._postgres_connect_kwargs()) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or ())
                rows = cur.fetchall()
                columns = [desc.name for desc in (cur.description or [])]
        return _normalize_dataframe(pd.DataFrame(rows, columns=columns))

    def _clickhouse_request(
        self,
        sql: str,
        *,
        database: str | None = None,
    ) -> dict[str, Any]:
        kwargs = self.runtime.to_driver_kwargs()
        options = kwargs.get("options", {})
        secure = bool(options.get("secure", False))
        scheme = "https" if secure else "http"
        port = kwargs.get("port") or (8443 if secure else 8123)
        params = urlencode(
            {
                "database": database if database is not None else (kwargs.get("database") or ""),
                "user": kwargs.get("username") or "",
                "password": kwargs.get("password") or "",
                "default_format": "JSON",
            }
        )
        query = _strip_sql(sql)
        if "format json" not in query.lower():
            query = f"{query} FORMAT JSON"
        request = Request(
            url=f"{scheme}://{kwargs['host']}:{port}/?{params}",
            data=query.encode("utf-8"),
            method="POST",
            headers={"Content-Type": "text/plain; charset=utf-8"},
        )
        with urlopen(request, timeout=self.timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("ClickHouse returned an unexpected response.")
        return payload

    def _clickhouse_query_dataframe(
        self,
        sql: str,
        *,
        database: str | None = None,
    ) -> pd.DataFrame:
        payload = self._clickhouse_request(sql, database=database)
        rows = payload.get("data", [])
        if not isinstance(rows, list):
            raise RuntimeError("ClickHouse returned malformed row payload.")
        return _normalize_dataframe(pd.DataFrame(rows))

    def list_schemas(self) -> list[str]:
        if self.runtime.db_type == "postgresql":
            df = self._postgres_query_dataframe(
                """
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                ORDER BY schema_name
                """
            )
            return [str(item) for item in df["schema_name"].tolist()]

        df = self._clickhouse_query_dataframe(
            """
            SELECT name
            FROM system.databases
            WHERE name NOT IN ('system', 'information_schema', 'INFORMATION_SCHEMA')
            ORDER BY name
            """
        )
        return [str(item) for item in df["name"].tolist()]

    def list_tables(self, schema: str | None = None) -> list[str]:
        target_schema = str(schema or self._default_schema()).strip()
        if self.runtime.db_type == "postgresql":
            df = self._postgres_query_dataframe(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                  AND table_type IN ('BASE TABLE', 'VIEW', 'FOREIGN TABLE', 'LOCAL TEMPORARY')
                ORDER BY table_name
                """,
                (target_schema,),
            )
            return [str(item) for item in df["table_name"].tolist()]

        safe_schema = target_schema.replace("\\", "\\\\").replace("'", "\\'")
        df = self._clickhouse_query_dataframe(
            f"""
            SELECT name
            FROM system.tables
            WHERE database = '{safe_schema}'
            ORDER BY name
            """,
            database=target_schema,
        )
        return [str(item) for item in df["name"].tolist()]

    def pick_demo_table(self, preferred_schema: str | None = None) -> dict[str, str]:
        def _table_priority(table_name: str) -> tuple[int, str]:
            if HIGH_PRIORITY_TABLE_RE.search(table_name):
                return (0, table_name)
            if LOW_PRIORITY_TABLE_RE.search(table_name):
                return (2, table_name)
            return (1, table_name)

        schema_candidates: list[str] = []
        if preferred_schema:
            schema_candidates.append(str(preferred_schema).strip())
        default_schema = self._default_schema()
        if default_schema not in schema_candidates:
            schema_candidates.append(default_schema)
        for schema_name in self.list_schemas():
            if schema_name not in schema_candidates:
                schema_candidates.append(schema_name)

        for schema_name in schema_candidates:
            tables = sorted(self.list_tables(schema_name), key=_table_priority)
            if tables:
                return {
                    "schema": schema_name,
                    "table": tables[0],
                }
        raise ValueError("No accessible tables were found in the selected database source.")

    def preview_table(
        self,
        table: str,
        *,
        schema: str | None = None,
        limit: int = 5,
    ) -> pd.DataFrame:
        row_limit = max(1, min(int(limit), 200))
        target_schema = str(schema or self._default_schema()).strip()
        qualified = f"{self._quote_identifier(target_schema)}.{self._quote_identifier(table)}"
        return self.query_dataframe(f"SELECT * FROM {qualified} LIMIT {row_limit}")

    def preview_first_table(
        self,
        *,
        preferred_schema: str | None = None,
        limit: int = 5,
    ) -> pd.DataFrame:
        selected = self.pick_demo_table(preferred_schema)
        return self.preview_table(
            selected["table"],
            schema=selected["schema"],
            limit=limit,
        )

    def preview_table_result(
        self,
        table: str,
        *,
        schema: str | None = None,
        limit: int = 5,
        artifact_name: str | None = None,
    ) -> dict[str, Any]:
        target_schema = str(schema or self._default_schema()).strip()
        rows = self.preview_table(table, schema=target_schema, limit=limit)
        resolved_name = artifact_name or f"preview_{target_schema}_{table}"
        return {
            "schema_version": "1.0",
            "artifact_type": "table",
            "items": {resolved_name: rows},
        }

    def demo_preview_result(
        self,
        *,
        preferred_schema: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        selected = self.pick_demo_table(preferred_schema)
        return self.preview_table_result(
            selected["table"],
            schema=selected["schema"],
            limit=limit,
        )

    def query_dataframe(self, sql: str) -> pd.DataFrame:
        clean_sql = _assert_read_only_sql(sql)
        if self.runtime.db_type == "postgresql":
            return self._postgres_query_dataframe(clean_sql)
        return self._clickhouse_query_dataframe(
            clean_sql,
            database=str(self.runtime.database or self._default_schema()),
        )


@dataclass(repr=False)
class DemoDBConnectionView:
    runtime: RuntimeDBConnectionConfig

    @property
    def db_type(self) -> str:
        return self.runtime.db_type

    def build_dsn(self) -> str:
        return self.runtime.build_dsn()

    def to_driver_kwargs(self) -> dict[str, Any]:
        return self.runtime.to_driver_kwargs()

    def __repr__(self) -> str:
        return (
            "DemoDBConnectionView("
            f"db_type={self.runtime.db_type!r}, "
            f"host={self.runtime.host!r}, "
            f"port={self.runtime.port!r}, "
            f"database={self.runtime.database!r}, "
            f"username={self.runtime.username!r}, "
            "password=<redacted>)"
        )

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            "`db_connection` is a runtime config view. "
            "Do not call driver methods on it directly. Use helper `db` "
            "for demo operations, or `db_connection.build_dsn()` / "
            "`db_connection.to_driver_kwargs()` for advanced tool code."
        )


class DBTool(BaseExecTool):
    """
    Reference DB-aware tool with a built-in helper API for safe read-only access.
    """

    name: str = "db_tool"
    artifact_name: str = "table"
    human_name: str = "таблиц из базы данных"
    description: str = db_tool_prompt
    allowed_libs: set[str] = {"pandas", "json"}
    allowed_artifact_types: tuple = (pd.DataFrame, pd.Series)
    forbidden_code_patterns: tuple[tuple[str, str], ...] = BASE_FORBIDDEN_CODE_PATTERNS + (
        (
            r"\b(psycopg|psycopg2|urllib|urlopen|Request)\b",
            "В db_tool не нужно импортировать драйверы и HTTP-клиенты вручную. Используйте helper `db`.",
        ),
        (
            r"\bdb_connection\s*\.\s*cursor\s*\(",
            "`db_connection` — это config object, а не открытое соединение. Используйте helper `db`.",
        ),
    )

    def __init__(
        self,
        df: pd.DataFrame,
        execution_timeout_sec: float = 25.0,
        tool_cache_size: int = 24,
        db_runtime_config: RuntimeDBConnectionConfig | None = None,
    ) -> None:
        super().__init__(
            df,
            execution_timeout_sec=execution_timeout_sec,
            include_plotly=False,
            tool_cache_size=tool_cache_size,
            db_runtime_config=db_runtime_config,
        )

    def get_execution_scope(self) -> dict[str, Any]:
        if self._db_runtime_config is None:
            return {}
        return {
            "db_connection": DemoDBConnectionView(runtime=self._db_runtime_config),
            "db": DBDemoHelper(
                runtime=self._db_runtime_config,
                timeout_sec=min(15.0, self.execution_timeout_sec),
            ),
        }

    def _run(self, code: str) -> tuple[str, dict[str, object]]:
        if self._db_runtime_config is None:
            text = (
                "вќЊ Ошибка при создании таблиц из базы данных: "
                "db_tool доступен только когда к сессии привязан DB source."
            )
            return text, {self.artifact_name: None, "text": text}
        return super()._run(code)
