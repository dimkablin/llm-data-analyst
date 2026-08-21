from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from datetime import time as datetime_time
from decimal import Decimal
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import UUID

import pandas as pd

from backend.artifacts.artifact_meta import (
    build_db_metadata_recipe_step,
    build_sql_recipe_step,
)
from backend.data_access.db_connectors import ResolvedDBConnection, build_connection_adapter
from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig

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
SELECT_STAR_RE = re.compile(r"^\s*(with\b[\s\S]+?\)\s*select\s+\*|select\s+\*)\b", re.IGNORECASE)
TERMINAL_LIMIT_RE = re.compile(
    r"\blimit\s+(?P<limit>\d+)(?:\s+offset\s+(?P<offset>\d+))?\s*$",
    re.IGNORECASE,
)
DEFAULT_ANALYTIC_MAX_ROWS = 200
HARD_ANALYTIC_MAX_ROWS = 1000
MAX_RESULT_CELLS = 12000


def _strip_sql(sql: str) -> str:
    return str(sql or "").strip().rstrip(";").strip()


def _mask_sql_non_code(sql: str) -> str:
    """Replace quoted text and comments with spaces while preserving positions."""
    masked = list(sql)
    index = 0
    while index < len(sql):
        start = index
        delimiter = ""
        if sql.startswith("--", index):
            end = sql.find("\n", index + 2)
            end = len(sql) if end < 0 else end
        elif sql.startswith("/*", index):
            close = sql.find("*/", index + 2)
            end = len(sql) if close < 0 else close + 2
        elif sql[index] in {"'", '"', "`"}:
            delimiter = sql[index]
            end = index + 1
            while end < len(sql):
                if sql[end] == "\\":
                    end += 2
                    continue
                if sql[end] == delimiter:
                    if end + 1 < len(sql) and sql[end + 1] == delimiter:
                        end += 2
                        continue
                    end += 1
                    break
                end += 1
        elif sql[index] == "$" and (match := re.match(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$", sql[index:])):
            delimiter = match.group(0)
            close = sql.find(delimiter, index + len(delimiter))
            end = len(sql) if close < 0 else close + len(delimiter)
        else:
            index += 1
            continue

        for position in range(start, end):
            if masked[position] not in "\r\n":
                masked[position] = " "
        index = end
    return "".join(masked)


def _escape_literal_percent_for_psycopg(sql: str) -> str:
    """Double '%' that are not psycopg placeholders (%s, %b, %f, %t)."""
    if "%" not in sql:
        return sql
    return re.sub(r"%(?![sbft])", "%%", sql)


def _assert_read_only_sql(sql: str) -> str:
    raw = str(sql or "").strip()
    if not raw:
        raise ValueError("SQL query is empty.")
    masked = _mask_sql_non_code(raw)
    semicolons = [index for index, char in enumerate(masked) if char == ";"]
    if semicolons:
        if len(semicolons) != 1 or semicolons[0] != len(raw) - 1:
            raise ValueError("Multiple SQL statements are not allowed.")
        raw = raw[:-1].rstrip()
        masked = masked[:-1].rstrip()
    if not READ_ONLY_SQL_RE.match(masked):
        raise ValueError("Only read-only SELECT/WITH queries are allowed.")
    if FORBIDDEN_SQL_RE.search(masked):
        raise ValueError("Non-read-only SQL keywords are not allowed.")
    return raw


def _coerce_max_rows(
    value: int | None,
    *,
    default: int = DEFAULT_ANALYTIC_MAX_ROWS,
    hard_max: int = HARD_ANALYTIC_MAX_ROWS,
) -> int:
    if value is None:
        return default
    return max(1, min(int(value), hard_max))


def _normalize_analytic_sql(
    sql: str,
    *,
    max_rows: int,
) -> tuple[str, dict[str, Any]]:
    clean_sql = _assert_read_only_sql(sql)
    requested_limit: int | None = None
    truncated_by_guardrail = False
    warnings: list[str] = []
    fetch_limit = _coerce_max_rows(max_rows) + 1

    limit_match = TERMINAL_LIMIT_RE.search(clean_sql)
    if limit_match:
        requested_limit = int(limit_match.group("limit"))
        if requested_limit <= max_rows:
            fetch_limit = requested_limit + 1
        else:
            truncated_by_guardrail = True
            warnings.append(f"Original LIMIT {requested_limit} exceeded max_rows={max_rows} and was reduced.")
    else:
        truncated_by_guardrail = True
        warnings.append(f"Appended LIMIT {max_rows} to keep the result compact.")

    if SELECT_STAR_RE.search(clean_sql):
        warnings.append("Query uses SELECT *. Prefer explicit columns for large or production queries.")

    normalized_sql = clean_sql
    if limit_match and requested_limit is not None:
        offset = int(limit_match.group("offset") or 0)
        replacement = f"LIMIT {fetch_limit}"
        if offset:
            replacement += f" OFFSET {offset}"
        normalized_sql = TERMINAL_LIMIT_RE.sub(replacement, clean_sql)
    elif limit_match is None:
        normalized_sql = f"{clean_sql} LIMIT {fetch_limit}"

    return normalized_sql, {
        "requested_sql": clean_sql,
        "requested_limit": requested_limit,
        "max_rows": max_rows,
        "fetch_limit": fetch_limit,
        "warnings": warnings,
        "guardrail_limited": truncated_by_guardrail,
    }


def _make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date, datetime_time)):
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
        values = safe_df[column].dropna()
        if not values.empty:
            date_like = values.map(
                lambda value: (
                    isinstance(value, (datetime, date))
                    or (isinstance(value, str) and len(value.strip()) >= 10 and _is_iso_datetime(value))
                )
            ).all()
            if date_like:
                safe_df[column] = pd.to_datetime(safe_df[column], errors="coerce", utc=True).dt.tz_localize(
                    None
                )
                continue
        safe_df[column] = safe_df[column].map(_make_json_safe)
    return safe_df


def _is_iso_datetime(value: str) -> bool:
    try:
        datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


@dataclass
class DBAnalyticsHelper:
    runtime: RuntimeDBConnectionConfig
    timeout_sec: float = 10.0

    def _configured_schema(self) -> str | None:
        schema = self.runtime.options.get("schema")
        if not isinstance(schema, str):
            return None
        clean = schema.strip()
        return clean or None

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

    def _statement_timeout_ms(self) -> int:
        return max(1000, int(self.timeout_sec * 1000))

    def _apply_postgres_session(self, conn: Any) -> None:
        schema = self._configured_schema()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT set_config('statement_timeout', %s, false)",
                (str(self._statement_timeout_ms()),),
            )
            if schema:
                cur.execute("SELECT set_config('search_path', %s, false)", (schema,))

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
        configured_schema = self._configured_schema()
        if configured_schema:
            return configured_schema
        if self.runtime.db_type == "clickhouse":
            return str(self.runtime.database or "default")
        return "public"

    def _resolved_connection(self) -> ResolvedDBConnection:
        return ResolvedDBConnection(
            connection_id=self.runtime.connection_id,
            user_id=self.runtime.user_id,
            name=self.runtime.name,
            db_type=self.runtime.db_type,
            host=self.runtime.host,
            port=self.runtime.port,
            database=self.runtime.database,
            username=self.runtime.username,
            password=self.runtime.password,
            options=dict(self.runtime.options),
        )

    def _catalog_adapter(self):
        return build_connection_adapter(
            self._resolved_connection(),
            timeout_sec=max(1, int(self.timeout_sec)),
        )

    def _source_ref(self) -> dict[str, str]:
        return {
            "source_type": "db_connection",
            "source_ref_id": self.runtime.connection_id,
            "source_label": self.runtime.name,
            "source_mode": "read_only",
        }

    def _metadata_recipe(
        self,
        *,
        action: str,
        schema: str | None = None,
        table: str | None = None,
    ) -> list[dict[str, Any]]:
        summary_parts = [f"Read {action} from attached database catalog"]
        if schema:
            summary_parts.append(f"schema={schema}")
        if table:
            summary_parts.append(f"table={table}")
        return [
            build_db_metadata_recipe_step(
                action=action,
                title=action.replace("_", " ").title(),
                tool_name="sql_tool",
                summary="; ".join(summary_parts),
            )
        ]

    def _query_recipe(
        self,
        *,
        sql: str,
        purpose: str | None = None,
        max_rows: int,
        warnings: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        summary = str(purpose or "Analytical read query").strip() or "Analytical read query"
        if warnings:
            summary = f"{summary}. Warnings: {'; '.join(warnings)}"
        return [
            build_sql_recipe_step(
                sql=sql,
                title="Executed SQL",
                tool_name="sql_tool",
                summary=f"{summary}; max_rows={max_rows}",
            )
        ]

    def _table_result(
        self,
        rows: pd.DataFrame,
        *,
        artifact_name: str,
        recipe: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "artifact_type": "table",
            "items": {artifact_name: _normalize_dataframe(rows)},
            "source": self._source_ref(),
        }
        if recipe:
            payload["recipe"] = recipe
        if meta:
            payload["meta"] = meta
        return payload

    @staticmethod
    def _cap_dataframe_rows(
        rows: pd.DataFrame,
        *,
        max_rows: int,
    ) -> tuple[pd.DataFrame, bool]:
        if len(rows) <= max_rows:
            return rows, False
        return rows.head(max_rows).copy(), True

    @staticmethod
    def _shrink_for_cell_budget(
        rows: pd.DataFrame,
        *,
        cell_budget: int = MAX_RESULT_CELLS,
    ) -> tuple[pd.DataFrame, bool]:
        if rows.empty or rows.shape[1] <= 0:
            return rows, False
        max_rows = max(1, cell_budget // max(1, rows.shape[1]))
        if len(rows) <= max_rows:
            return rows, False
        return rows.head(max_rows).copy(), True

    def _package_query_result(
        self,
        rows: pd.DataFrame,
        *,
        artifact_name: str,
        executed_sql: str,
        requested_sql: str,
        purpose: str | None,
        max_rows: int,
        execution_time_ms: int,
        warnings: list[str] | None = None,
        truncated: bool = False,
        requested_limit: int | None = None,
        has_more_rows: bool = False,
    ) -> dict[str, Any]:
        safe_rows = _normalize_dataframe(rows)
        query_meta = {
            "purpose": str(purpose or "").strip() or None,
            "requested_sql": requested_sql,
            "executed_sql": executed_sql,
            "max_rows": max_rows,
            "requested_limit": requested_limit,
            "returned_rows": len(safe_rows),
            "column_count": len(safe_rows.columns),
            "truncated": bool(truncated),
            "has_more_rows": bool(has_more_rows),
            "execution_time_ms": int(execution_time_ms),
            "warnings": list(warnings or []),
        }
        meta = {
            "query": query_meta,
            "execution_stats": {
                "row_count": query_meta["returned_rows"],
                "column_count": query_meta["column_count"],
                "truncated": query_meta["truncated"],
                "has_more_rows": query_meta["has_more_rows"],
                "execution_time_ms": query_meta["execution_time_ms"],
            },
            "warnings": list(warnings or []),
        }
        return self._table_result(
            safe_rows,
            artifact_name=artifact_name,
            recipe=self._query_recipe(
                sql=executed_sql,
                purpose=purpose,
                max_rows=max_rows,
                warnings=list(warnings or []),
            ),
            meta=meta,
        )

    def _postgres_query_dataframe(
        self,
        sql: str,
        params: tuple[Any, ...] | None = None,
    ) -> pd.DataFrame:
        import psycopg

        bound_sql = sql if params else _escape_literal_percent_for_psycopg(sql)
        with psycopg.connect(**self._postgres_connect_kwargs()) as conn:
            self._apply_postgres_session(conn)
            with conn.cursor() as cur:
                cur.execute(bound_sql, params or ())
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
                "database": database
                if database is not None
                else (self._configured_schema() or kwargs.get("database") or ""),
                "user": kwargs.get("username") or "",
                "password": kwargs.get("password") or "",
                "default_format": "JSON",
                "max_execution_time": max(1, int(self.timeout_sec)),
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

    def list_schemas(self) -> list[dict[str, str]]:
        adapter = self._catalog_adapter()
        return [
            {
                "name": item.name,
                "display_name": item.display_name,
            }
            for item in adapter.list_schemas()
        ]

    def list_schemas_result(
        self,
        *,
        artifact_name: str = "db_schemas",
    ) -> dict[str, Any]:
        rows = pd.DataFrame(self.list_schemas(), columns=["name", "display_name"])
        return self._table_result(
            rows,
            artifact_name=artifact_name,
            recipe=self._metadata_recipe(action="list_schemas"),
        )

    def list_tables(self, schema: str | None = None) -> list[dict[str, str]]:
        target_schema = str(schema or self._default_schema()).strip()
        adapter = self._catalog_adapter()
        return [
            {
                "schema": item.schema,
                "table_name": item.name,
                "table_type": item.table_type,
                "qualified_name": item.qualified_name,
            }
            for item in adapter.list_tables(target_schema)
        ]

    def list_tables_with_columns(
        self,
        schema: str | None = None,
    ) -> list[dict[str, Any]]:
        """Single-query fetch of all tables + their column names.

        Returns a list of dicts compatible with TableCandidate construction:
        [{"schema", "table_name", "table_type", "qualified_name", "columns"}, ...]
        """
        target_schema = str(schema or self._default_schema()).strip()
        adapter = self._catalog_adapter()
        combined = adapter.list_tables_with_columns(target_schema)
        result: list[dict[str, Any]] = []
        for cat_table, cat_columns in combined.values():
            result.append(
                {
                    "schema": cat_table.schema,
                    "table_name": cat_table.name,
                    "table_type": cat_table.table_type,
                    "qualified_name": cat_table.qualified_name,
                    "columns": [col.name for col in cat_columns],
                    "column_types": {col.name: col.data_type for col in cat_columns},
                }
            )
        return result

    def list_all_tables_with_columns(self) -> list[dict[str, Any]]:
        """Enumerate every schema and return tables with column names (deduplicated)."""
        combined: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for schema_row in self.list_schemas():
            schema_name = str(schema_row.get("name") or "").strip()
            if not schema_name:
                continue
            if schema_name.lower() in {"information_schema", "pg_catalog", "pg_toast"}:
                continue
            for row in self.list_tables_with_columns(schema_name):
                table_name = str(row.get("table_name") or "").strip()
                if not table_name:
                    continue
                key = (str(row.get("schema") or schema_name), table_name)
                if key in seen:
                    continue
                seen.add(key)
                combined.append(row)
        return combined

    def list_effective_tables_with_columns(
        self,
        schema: str | None = None,
    ) -> list[dict[str, Any]]:
        """Tables visible to tools for this connection.

        If ``options.schema`` is set on the connection, only that schema is used
        (matches PostgreSQL ``search_path``). Otherwise all non-system schemas are
        enumerated.
        """
        if schema is not None:
            return self.list_tables_with_columns(schema)
        if self._configured_schema():
            return self.list_tables_with_columns()
        return self.list_all_tables_with_columns()

    def list_relationships(self, schema: str | None = None) -> list[dict[str, str]]:
        target_schema = str(schema or self._default_schema()).strip()
        adapter = self._catalog_adapter()
        return [
            {
                "from_schema": item.from_schema,
                "from_table": item.from_table,
                "from_column": item.from_column,
                "to_schema": item.to_schema,
                "to_table": item.to_table,
                "to_column": item.to_column,
            }
            for item in adapter.list_relationships(target_schema)
        ]

    def list_effective_relationships(self, schema: str | None = None) -> list[dict[str, str]]:
        if schema is not None:
            return self.list_relationships(schema)
        if self._configured_schema():
            return self.list_relationships()
        relationships: list[dict[str, str]] = []
        for schema_row in self.list_schemas():
            schema_name = str(schema_row.get("name") or "").strip()
            if not schema_name:
                continue
            if schema_name.lower() in {"information_schema", "pg_catalog", "pg_toast"}:
                continue
            relationships.extend(self.list_relationships(schema_name))
        return relationships

    def list_tables_result(
        self,
        schema: str | None = None,
        *,
        artifact_name: str | None = None,
    ) -> dict[str, Any]:
        if schema is not None:
            target_schema = str(schema).strip()
        elif self._configured_schema():
            target_schema = self._configured_schema() or self._default_schema()
        else:
            target_schema = ""
        rows_meta = self.list_effective_tables_with_columns(schema)
        table_rows = [
            {
                "schema": row.get("schema"),
                "table_name": row.get("table_name"),
                "table_type": row.get("table_type"),
                "qualified_name": row.get("qualified_name"),
            }
            for row in rows_meta
        ]
        if target_schema:
            resolved_name = artifact_name or f"db_tables_{target_schema}"
        else:
            resolved_name = artifact_name or "db_tables"
        rows = pd.DataFrame(
            table_rows,
            columns=["schema", "table_name", "table_type", "qualified_name"],
        )
        return self._table_result(
            rows,
            artifact_name=resolved_name,
            recipe=self._metadata_recipe(action="list_tables", schema=target_schema),
        )

    def describe_table(
        self,
        table: str,
        *,
        schema: str | None = None,
    ) -> list[dict[str, Any]]:
        target_schema = str(schema or self._default_schema()).strip()
        clean_table = str(table or "").strip()
        if not clean_table:
            raise ValueError("table must not be empty.")
        adapter = self._catalog_adapter()
        columns = adapter.describe_table(target_schema, clean_table)
        if not columns:
            raise ValueError(
                f"Table '{target_schema}.{clean_table}' was not found or has no visible columns."
            )
        return [
            {
                "schema": item.schema,
                "table_name": item.table,
                "column_name": item.name,
                "data_type": item.data_type,
                "is_nullable": item.is_nullable,
                "ordinal_position": item.ordinal_position,
                "default_expression": item.default_expression,
            }
            for item in columns
        ]

    def describe_table_result(
        self,
        table: str,
        *,
        schema: str | None = None,
        artifact_name: str | None = None,
    ) -> dict[str, Any]:
        target_schema = str(schema or self._default_schema()).strip()
        clean_table = str(table or "").strip()
        rows = pd.DataFrame(
            self.describe_table(clean_table, schema=target_schema),
            columns=[
                "schema",
                "table_name",
                "column_name",
                "data_type",
                "is_nullable",
                "ordinal_position",
                "default_expression",
            ],
        )
        resolved_name = artifact_name or f"describe_{target_schema}_{clean_table}"
        return self._table_result(
            rows,
            artifact_name=resolved_name,
            recipe=self._metadata_recipe(
                action="describe_table",
                schema=target_schema,
                table=clean_table,
            ),
        )

    def list_columns(
        self,
        table: str,
        *,
        schema: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.describe_table(table, schema=schema)

    def validate_sql(
        self,
        sql: str,
        *,
        max_rows: int | None = None,
    ) -> dict[str, Any]:
        effective_max_rows = _coerce_max_rows(max_rows)
        normalized_sql, validation = _normalize_analytic_sql(
            sql,
            max_rows=effective_max_rows,
        )
        return {
            "requested_sql": validation["requested_sql"],
            "normalized_sql": normalized_sql,
            "requested_limit": validation["requested_limit"],
            "max_rows": validation["max_rows"],
            "warnings": list(validation["warnings"]),
        }

    def execute_read_query(
        self,
        sql: str,
        *,
        purpose: str | None = None,
        max_rows: int | None = None,
        artifact_name: str = "db_query_result",
    ) -> dict[str, Any]:
        effective_max_rows = _coerce_max_rows(max_rows)
        executed_sql, validation = _normalize_analytic_sql(
            sql,
            max_rows=effective_max_rows,
        )
        started_at = time.perf_counter()
        rows = self.query_dataframe(executed_sql)
        execution_time_ms = int((time.perf_counter() - started_at) * 1000)

        warnings = list(validation["warnings"])
        truncated = False
        requested_limit = validation["requested_limit"]
        has_more_rows = bool(
            requested_limit is not None
            and requested_limit <= effective_max_rows
            and len(rows) > requested_limit
        )
        if has_more_rows:
            rows = rows.head(requested_limit).copy()
            warnings.append(f"Explicit LIMIT {requested_limit} omitted additional rows.")
        rows, capped = self._cap_dataframe_rows(rows, max_rows=effective_max_rows)
        truncated = truncated or capped
        if capped:
            warnings.append(f"Result exceeded max_rows={effective_max_rows} and was truncated.")

        rows, cell_capped = self._shrink_for_cell_budget(rows)
        truncated = truncated or cell_capped
        if cell_capped:
            warnings.append(f"Result exceeded cell budget={MAX_RESULT_CELLS} and was truncated.")

        return self._package_query_result(
            rows,
            artifact_name=artifact_name,
            executed_sql=executed_sql,
            requested_sql=validation["requested_sql"],
            purpose=purpose,
            max_rows=effective_max_rows,
            execution_time_ms=execution_time_ms,
            warnings=warnings,
            truncated=truncated,
            requested_limit=requested_limit,
            has_more_rows=has_more_rows,
        )

    def execute_analytic_query(
        self,
        sql: str,
        *,
        purpose: str | None = None,
        max_rows: int | None = None,
        artifact_name: str = "db_query_result",
    ) -> dict[str, Any]:
        return self.execute_read_query(
            sql,
            purpose=purpose,
            max_rows=max_rows,
            artifact_name=artifact_name,
        )

    def pick_demo_table(self, preferred_schema: str | None = None) -> dict[str, str]:
        def _table_priority(item: dict[str, str]) -> tuple[int, str]:
            table_name = str(item.get("table_name") or "").strip()
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
        for schema_item in self.list_schemas():
            schema_name = str(schema_item.get("name") or "").strip()
            if schema_name and schema_name not in schema_candidates:
                schema_candidates.append(schema_name)

        for schema_name in schema_candidates:
            tables = sorted(self.list_tables(schema_name), key=_table_priority)
            if tables:
                selected = tables[0]
                return {
                    "schema": str(selected.get("schema") or schema_name),
                    "table": str(selected.get("table_name") or ""),
                    "qualified_name": str(selected.get("qualified_name") or ""),
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
        rows = self.query_dataframe(f"SELECT * FROM {qualified} LIMIT {row_limit}")
        rows, _ = self._cap_dataframe_rows(rows, max_rows=row_limit)
        return rows

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
        resolved_name = artifact_name or f"preview_{target_schema}_{table}"
        qualified = (
            f"{self._quote_identifier(target_schema)}.{self._quote_identifier(str(table or '').strip())}"
        )
        return self.execute_analytic_query(
            f"SELECT * FROM {qualified}",
            purpose=f"Preview rows from {target_schema}.{str(table or '').strip()}",
            max_rows=max(1, min(int(limit), 200)),
            artifact_name=resolved_name,
        )

    def get_table_preview(
        self,
        table: str,
        *,
        schema: str | None = None,
        limit: int = 5,
        artifact_name: str | None = None,
    ) -> dict[str, Any]:
        return self.preview_table_result(
            table,
            schema=schema,
            limit=limit,
            artifact_name=artifact_name,
        )

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

    def query_dataframe(
        self,
        sql: str,
        params: tuple[Any, ...] | list[Any] | None = None,
    ) -> pd.DataFrame:
        clean_sql = _assert_read_only_sql(sql)
        bound_params: tuple[Any, ...] | None = None
        if params is not None:
            bound_params = tuple(params)
        if self.runtime.db_type == "postgresql":
            return self._postgres_query_dataframe(clean_sql, bound_params)
        return self._clickhouse_query_dataframe(
            clean_sql,
            database=str(self._configured_schema() or self.runtime.database or self._default_schema()),
        )


DBDemoHelper = DBAnalyticsHelper


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
