"""Session data-source catalog tool.

The tool exposes the typed source inventory assembled by the agent runtime. It
does not query user data directly; it only helps the LLM choose the right table
or source before using sql_tool, database_tool, pandas_tool, or plotly_tool.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from backend.data_access.source_inventory import (
    SourceInventory,
    SourceInventorySource,
    SourceInventoryTable,
    SourceType,
)
from backend.tools.instructions import tool_description

CatalogAction = Literal["list_sources", "list_tables", "describe_table", "search"]
CatalogStatus = Literal["ok", "not_found", "ambiguous", "error"]

MAX_RESULT_TABLES = 50


class DataCatalogToolArgs(BaseModel):
    action: CatalogAction = Field(
        ...,
        description=("Catalog action: list_sources, list_tables, describe_table, or search."),
    )
    table: str | None = Field(
        default=None,
        description=(
            "Table name for describe_table. Prefer exact qualified_name values, for example mart.orders."
        ),
    )
    source_id: str | None = Field(
        default=None,
        description="Optional exact source_id filter, for example orders_csv or db:conn-1.",
    )
    source_type: SourceType | None = Field(
        default=None,
        description="Optional source type filter: csv or db_connection.",
    )
    query: str | None = Field(
        default=None,
        description="Optional case-insensitive search text for table, source, or column names.",
    )


class DataCatalogToolResult(BaseModel):
    status: CatalogStatus
    action: CatalogAction
    message: str
    sources: list[SourceInventorySource] = Field(default_factory=list)
    tables: list[SourceInventoryTable] = Field(default_factory=list)
    selected_table: SourceInventoryTable | None = None
    table_count_by_source: dict[str, int] = Field(default_factory=dict)


class DataCatalogTool(BaseTool):
    """Returns structured source/table inventory for the current session."""

    name: str = "data_catalog_tool"
    description: str = tool_description("data_catalog_tool")
    args_schema: type[BaseModel] = DataCatalogToolArgs
    response_format: str = "content"
    parallel_safe: ClassVar[bool] = True

    _source_inventory: SourceInventory = PrivateAttr()

    def __init__(self, *, source_inventory: SourceInventory) -> None:
        super().__init__()
        self._source_inventory = source_inventory

    def _run(
        self,
        action: CatalogAction,
        table: str | None = None,
        source_id: str | None = None,
        source_type: SourceType | None = None,
        query: str | None = None,
    ) -> str:
        result = self._run_action(
            action=action,
            table=table,
            source_id=source_id,
            source_type=source_type,
            query=query,
        )
        if result.status == "error":
            raise ValueError(result.message)
        return result.model_dump_json(indent=2)

    def _run_action(
        self,
        *,
        action: CatalogAction,
        table: str | None,
        source_id: str | None,
        source_type: SourceType | None,
        query: str | None,
    ) -> DataCatalogToolResult:
        if action == "list_sources":
            return self._list_sources(source_id=source_id, source_type=source_type)

        tables = self._filtered_tables(source_id=source_id, source_type=source_type)
        if action == "list_tables":
            return self._list_tables(tables, query=query)
        if action == "search":
            return self._search(tables, query=query)
        if action == "describe_table":
            return self._describe_table(tables, table=table)

        return DataCatalogToolResult(
            status="error",
            action=action,
            message=f"Unknown catalog action: {action}",
        )

    def _list_sources(
        self,
        *,
        source_id: str | None,
        source_type: SourceType | None,
    ) -> DataCatalogToolResult:
        sources = [
            source
            for source in self._source_inventory.sources
            if self._source_matches(source, source_id=source_id, source_type=source_type)
        ]
        return DataCatalogToolResult(
            status="ok",
            action="list_sources",
            message=f"Found {len(sources)} source(s).",
            sources=sources,
            table_count_by_source=self._table_count_by_source(),
        )

    def _list_tables(
        self,
        tables: list[SourceInventoryTable],
        *,
        query: str | None,
    ) -> DataCatalogToolResult:
        matched = self._query_tables(tables, query)[:MAX_RESULT_TABLES]
        return DataCatalogToolResult(
            status="ok",
            action="list_tables",
            message=f"Found {len(matched)} table(s).",
            tables=matched,
            table_count_by_source=self._table_count_by_source(),
        )

    def _search(
        self,
        tables: list[SourceInventoryTable],
        *,
        query: str | None,
    ) -> DataCatalogToolResult:
        clean_query = str(query or "").strip()
        if not clean_query:
            return DataCatalogToolResult(
                status="error",
                action="search",
                message="query is required for search.",
            )
        matched = self._query_tables(tables, clean_query)[:MAX_RESULT_TABLES]
        status: CatalogStatus = "ok" if matched else "not_found"
        return DataCatalogToolResult(
            status=status,
            action="search",
            message=f"Found {len(matched)} matching table(s).",
            tables=matched,
            table_count_by_source=self._table_count_by_source(),
        )

    def _describe_table(
        self,
        tables: list[SourceInventoryTable],
        *,
        table: str | None,
    ) -> DataCatalogToolResult:
        clean_table = str(table or "").strip()
        if not clean_table:
            return DataCatalogToolResult(
                status="error",
                action="describe_table",
                message="table is required for describe_table.",
            )

        matches = self._match_tables(tables, clean_table)
        if len(matches) == 1:
            selected = matches[0]
            return DataCatalogToolResult(
                status="ok",
                action="describe_table",
                message=f"Selected table {selected.qualified_name}.",
                selected_table=selected,
                tables=[selected],
                table_count_by_source=self._table_count_by_source(),
            )
        if len(matches) > 1:
            return DataCatalogToolResult(
                status="ambiguous",
                action="describe_table",
                message=("Table name is ambiguous. Use one of the returned qualified_name values."),
                tables=matches[:MAX_RESULT_TABLES],
                table_count_by_source=self._table_count_by_source(),
            )
        return DataCatalogToolResult(
            status="not_found",
            action="describe_table",
            message=f"Table '{clean_table}' was not found in the current source inventory.",
            table_count_by_source=self._table_count_by_source(),
        )

    def _filtered_tables(
        self,
        *,
        source_id: str | None,
        source_type: SourceType | None,
    ) -> list[SourceInventoryTable]:
        clean_source_id = str(source_id or "").strip()
        return [
            table
            for table in self._source_inventory.tables
            if (not clean_source_id or table.source_id == clean_source_id)
            and (source_type is None or table.source_type == source_type)
        ]

    @staticmethod
    def _source_matches(
        source: SourceInventorySource,
        *,
        source_id: str | None,
        source_type: SourceType | None,
    ) -> bool:
        clean_source_id = str(source_id or "").strip()
        if clean_source_id and source.source_id != clean_source_id:
            return False
        return source_type is None or source.source_type == source_type

    @staticmethod
    def _query_tables(
        tables: list[SourceInventoryTable],
        query: str | None,
    ) -> list[SourceInventoryTable]:
        clean_query = str(query or "").strip().lower()
        if not clean_query:
            return tables
        return [table for table in tables if DataCatalogTool._table_contains(table, clean_query)]

    @staticmethod
    def _match_tables(
        tables: list[SourceInventoryTable],
        table_name: str,
    ) -> list[SourceInventoryTable]:
        needle = table_name.lower()
        if "." not in needle:
            bare_matches = [table for table in tables if table.table_name.lower() == needle]
            if bare_matches:
                return bare_matches
        qualified_matches = [table for table in tables if table.qualified_name.lower() == needle]
        if qualified_matches:
            return qualified_matches
        return [table for table in tables if table.table_name.lower() == needle]

    @staticmethod
    def _table_contains(table: SourceInventoryTable, needle: str) -> bool:
        haystack = [
            table.qualified_name,
            table.table_name,
            table.schema_name or "",
            table.source_id,
            table.source_label,
            table.source_alias or "",
            *table.columns,
        ]
        return any(needle in str(value).lower() for value in haystack)

    def _table_count_by_source(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for table in self._source_inventory.tables:
            counts[table.source_id] = counts.get(table.source_id, 0) + 1
        return counts
