from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.session_source import SessionSource, SourceType, is_duckdb_source_type

logger = logging.getLogger(__name__)

class SourceInventorySource(BaseModel):
    source_id: str
    source_type: SourceType
    label: str
    alias: str | None = None


class SourceInventoryTable(BaseModel):
    source_id: str
    source_type: SourceType
    table_name: str
    qualified_name: str
    schema_name: str | None = None
    source_label: str = ""
    source_alias: str | None = None
    columns: list[str] = Field(default_factory=list)
    column_types: dict[str, str] = Field(default_factory=dict)
    row_count: int | None = None
    column_count: int | None = None


class SourceInventory(BaseModel):
    session_id: str
    sources: list[SourceInventorySource] = Field(default_factory=list)
    tables: list[SourceInventoryTable] = Field(default_factory=list)

    @property
    def has_multiple_tables(self) -> bool:
        return len(self.tables) > 1

    @property
    def has_multiple_sources(self) -> bool:
        return len({table.source_id for table in self.tables}) > 1


def build_source_inventory(
    *,
    session_id: str,
    session_source: dict[str, Any] | None,
    manifest_store: ManifestStore,
    csv_runtime: Any | None,
    db_runtime: RuntimeDBConnectionConfig | None,
) -> SourceInventory:
    sources: list[SourceInventorySource] = []
    tables: list[SourceInventoryTable] = []
    seen_sources: set[str] = set()
    seen_tables: set[tuple[str, str]] = set()

    manifest = manifest_store.load(session_id)
    csv_sources = [source for source in manifest.sources if is_duckdb_source_type(source.source_type)]
    for source in csv_sources:
        _append_source(
            sources,
            seen_sources,
            _csv_source_id(source),
            source.source_type,
            _source_label(source),
            source.alias,
        )

    _append_csv_runtime_tables(
        tables=tables,
        seen_tables=seen_tables,
        session_id=session_id,
        session_source=session_source,
        csv_runtime=csv_runtime,
        csv_sources=csv_sources,
    )
    _append_csv_manifest_tables(
        tables=tables,
        seen_tables=seen_tables,
        csv_sources=csv_sources,
    )
    _append_db_tables(
        sources=sources,
        tables=tables,
        seen_sources=seen_sources,
        db_runtime=db_runtime,
    )

    tables.sort(key=lambda item: (item.source_type, item.qualified_name))
    return SourceInventory(session_id=session_id, sources=sources, tables=tables)


def _append_source(
    sources: list[SourceInventorySource],
    seen_sources: set[str],
    source_id: str,
    source_type: SourceType,
    label: str,
    alias: str | None,
) -> None:
    if source_id in seen_sources:
        return
    seen_sources.add(source_id)
    sources.append(
        SourceInventorySource(
            source_id=source_id,
            source_type=source_type,
            label=label,
            alias=alias,
        )
    )


def _append_csv_runtime_tables(
    *,
    tables: list[SourceInventoryTable],
    seen_tables: set[tuple[str, str]],
    session_id: str,
    session_source: dict[str, Any] | None,
    csv_runtime: Any | None,
    csv_sources: list[SessionSource],
) -> None:
    if csv_runtime is None:
        return
    sid = _csv_session_id(session_id, session_source, csv_sources)
    if not sid or not bool((session_source or {}).get("csv_loaded")):
        return

    source_by_table = _csv_source_by_table(csv_sources)
    try:
        runtime_rows = csv_runtime.list_tables(sid)
    except Exception as exc:
        logger.debug("CSV source inventory runtime list failed: %s", exc)
        return

    for row in runtime_rows:
        table_name = str(row.get("table_name") or "").strip()
        if not table_name:
            continue
        source = source_by_table.get(table_name)
        columns = _runtime_csv_columns(csv_runtime, sid, table_name)
        table = _csv_inventory_table(
            table_name=table_name,
            qualified_name=str(row.get("qualified_name") or table_name).strip() or table_name,
            schema_name=str(row.get("schema") or "main").strip() or None,
            source=source,
            columns=columns,
        )
        _append_table(tables, seen_tables, table)


def _append_csv_manifest_tables(
    *,
    tables: list[SourceInventoryTable],
    seen_tables: set[tuple[str, str]],
    csv_sources: list[SessionSource],
) -> None:
    for source in csv_sources:
        columns = [str(column) for column in source.schema_hint.keys() if str(column).strip()]
        for table_name in source.csv_table_names:
            clean_table = str(table_name or "").strip()
            if not clean_table:
                continue
            table = _csv_inventory_table(
                table_name=clean_table,
                qualified_name=clean_table,
                schema_name="main",
                source=source,
                columns=columns,
            )
            _append_table(tables, seen_tables, table)


def _append_db_tables(
    *,
    sources: list[SourceInventorySource],
    tables: list[SourceInventoryTable],
    seen_sources: set[str],
    db_runtime: RuntimeDBConnectionConfig | None,
) -> None:
    if db_runtime is None:
        return
    source_id = _db_source_id(db_runtime)
    _append_source(
        sources,
        seen_sources,
        source_id,
        "db_connection",
        str(db_runtime.name or "DB source"),
        None,
    )
    try:
        from backend.tools.impl.db_helpers import DBAnalyticsHelper

        rows = DBAnalyticsHelper(runtime=db_runtime, timeout_sec=15.0).list_effective_tables_with_columns()
    except Exception as exc:
        logger.debug("DB source inventory list failed: %s", exc)
        return

    for row in rows:
        table_name = str(row.get("table_name") or "").strip()
        qualified_name = str(row.get("qualified_name") or table_name).strip()
        if not table_name or not qualified_name:
            continue
        tables.append(
            SourceInventoryTable(
                source_id=source_id,
                source_type="db_connection",
                table_name=table_name,
                qualified_name=qualified_name,
                schema_name=str(row.get("schema") or "").strip() or None,
                source_label=str(db_runtime.name or "DB source"),
                columns=[str(column) for column in row.get("columns", []) if str(column).strip()],
                column_types={
                    str(name): str(dtype)
                    for name, dtype in (row.get("column_types") or {}).items()
                },
            )
        )


def _append_table(
    tables: list[SourceInventoryTable],
    seen_tables: set[tuple[str, str]],
    table: SourceInventoryTable,
) -> None:
    key = (table.source_id, table.qualified_name)
    if key in seen_tables:
        return
    seen_tables.add(key)
    tables.append(table)


def _csv_inventory_table(
    *,
    table_name: str,
    qualified_name: str,
    schema_name: str | None,
    source: SessionSource | None,
    columns: list[str],
) -> SourceInventoryTable:
    return SourceInventoryTable(
        source_id=_csv_source_id(source),
        source_type=source.source_type if source is not None else "csv",
        table_name=table_name,
        qualified_name=qualified_name,
        schema_name=schema_name,
        source_label=_source_label(source),
        source_alias=source.alias if source is not None else None,
        columns=columns,
        row_count=source.row_count if source is not None else None,
        column_count=source.column_count if source is not None else None,
    )


def _runtime_csv_columns(csv_runtime: Any, session_id: str, table_name: str) -> list[str]:
    try:
        return [
            str(column.get("column_name") or "").strip()
            for column in csv_runtime.describe_table(session_id, table_name)
            if str(column.get("column_name") or "").strip()
        ]
    except Exception as exc:
        logger.debug("CSV source inventory describe failed for %s: %s", table_name, exc)
        return []


def _csv_source_by_table(csv_sources: list[SessionSource]) -> dict[str, SessionSource]:
    return {
        str(table_name).strip(): source
        for source in csv_sources
        for table_name in source.csv_table_names
        if str(table_name).strip()
    }


def _csv_session_id(
    session_id: str,
    session_source: dict[str, Any] | None,
    csv_sources: list[SessionSource],
) -> str:
    direct = str((session_source or {}).get("csv_session_id") or "").strip()
    if direct:
        return direct
    for source in csv_sources:
        sid = str(source.csv_session_id or "").strip()
        if sid:
            return sid
    return str(session_id or "").strip()


def _csv_source_id(source: SessionSource | None) -> str:
    if source is None:
        return "csv"
    return str(source.alias or source.file_name or source.display_name or "csv").strip()


def _db_source_id(db_runtime: RuntimeDBConnectionConfig) -> str:
    return f"db:{db_runtime.connection_id}"


def _source_label(source: SessionSource | None) -> str:
    if source is None:
        return "CSV session"
    return str(source.display_name or source.file_name or source.alias or "CSV source")


def format_source_inventory_prompt(
    inventory: SourceInventory,
    *,
    max_tables: int = 24,
    max_columns: int = 24,
) -> str:
    if not inventory.tables:
        return ""

    lines = [
        "[SOURCE INVENTORY]",
        "Use exact `qualified_name` values when generating SQL.",
    ]
    for table in inventory.tables[:max_tables]:
        columns = ", ".join(
            f"`{column}` ({dtype})" if (dtype := table.column_types.get(column, "").strip())
            else f"`{column}`"
            for column in table.columns[:max_columns]
        ) or "unknown columns"
        if len(table.columns) > max_columns:
            columns = f"{columns}, ... +{len(table.columns) - max_columns} columns"
        source_parts = [
            part
            for part in (table.source_label, table.source_alias or table.source_id)
            if str(part or "").strip()
        ]
        source = ", ".join(dict.fromkeys(source_parts))
        lines.append(
            f"- `{table.qualified_name}` [{table.source_type}; source={source}]: {columns}"
        )
    if len(inventory.tables) > max_tables:
        lines.append(f"- ... {len(inventory.tables) - max_tables} more tables")
    return "\n".join(lines)
