from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.data_access.data_catalog import fuzzy_match_column

SourceKind = Literal[
    "session_dataframe",
    "duckdb_table",
    "sql_result",
    "pandas_result",
    "table_schema",
    "unknown",
]


class ColumnLineage(BaseModel):
    """Best-effort lineage for one output column."""

    output_column: str
    source_table: str | None = None
    source_column: str | None = None


class SourceTableSchema(BaseModel):
    """Schema of a DuckDB/DB table that is not necessarily a pandas variable."""

    table_name: str
    columns: list[str] = Field(default_factory=list)
    source_kind: str = "duckdb_table"

    @field_validator("columns", mode="before")
    @classmethod
    def _normalize_columns(cls, raw: object) -> list[str]:
        if raw is None:
            return []
        return [str(item) for item in raw if str(item).strip()]


class DataFrameSchemaEntry(BaseModel):
    """Current schema and origin of one pandas DataFrame in the sandbox."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    variable_name: str
    source_kind: SourceKind = "unknown"
    source_name: str | None = None
    columns: list[str] = Field(default_factory=list)
    alias_map: dict[str, str] = Field(default_factory=dict)
    source_tables: list[str] = Field(default_factory=list)
    lineage: dict[str, list[ColumnLineage]] = Field(default_factory=dict)

    @field_validator("columns", mode="before")
    @classmethod
    def _normalize_columns(cls, raw: object) -> list[str]:
        if raw is None:
            return []
        return [str(item) for item in raw if str(item).strip()]

    @field_validator("alias_map", mode="before")
    @classmethod
    def _normalize_alias_map(cls, raw: object) -> dict[str, str]:
        if not isinstance(raw, dict):
            return {}
        return {
            str(key): str(value)
            for key, value in raw.items()
            if str(key).strip() and str(value).strip()
        }

    def resolve_column(self, requested_column: str) -> str | None:
        requested = str(requested_column)
        if requested in self.columns:
            return requested
        aliased = self.alias_map.get(requested)
        if aliased in self.columns:
            return aliased
        return fuzzy_match_column(requested, self.columns)


class ColumnResolution(BaseModel):
    variable_name: str
    requested_column: str
    resolved_column: str | None = None
    reason: Literal["exists", "alias", "fuzzy", "missing"] = "missing"

    @property
    def needs_rewrite(self) -> bool:
        return (
            self.resolved_column is not None
            and self.resolved_column != self.requested_column
        )


class DataFrameSchemaRegistry(BaseModel):
    """Typed source of truth for sandbox DataFrame schemas and source tables."""

    entries: dict[str, DataFrameSchemaEntry] = Field(default_factory=dict)
    source_tables: dict[str, SourceTableSchema] = Field(default_factory=dict)

    def register(self, entry: DataFrameSchemaEntry) -> None:
        self.entries[str(entry.variable_name)] = entry

    def register_dataframe(
        self,
        *,
        variable_name: str,
        df: pd.DataFrame,
        source_kind: SourceKind = "unknown",
        source_name: str | None = None,
        alias_map: dict[str, str] | None = None,
        source_tables: list[str] | None = None,
        lineage: dict[str, list[ColumnLineage]] | None = None,
    ) -> None:
        self.register(
            DataFrameSchemaEntry(
                variable_name=variable_name,
                source_kind=source_kind,
                source_name=source_name,
                columns=[str(col) for col in df.columns],
                alias_map=dict(alias_map or {}),
                source_tables=list(source_tables or []),
                lineage=dict(lineage or {}),
            )
        )

    def register_source_table(
        self,
        table_name: str,
        columns: Iterable[str],
        *,
        source_kind: str = "duckdb_table",
    ) -> None:
        clean_name = str(table_name or "").strip()
        if not clean_name:
            return
        schema = SourceTableSchema(
            table_name=clean_name,
            columns=[str(col) for col in columns if str(col).strip()],
            source_kind=source_kind,
        )
        self.source_tables[clean_name] = schema

    def get(self, variable_name: str) -> DataFrameSchemaEntry | None:
        return self.entries.get(str(variable_name))

    def resolve_column(
        self,
        variable_name: str,
        requested_column: str,
    ) -> ColumnResolution:
        entry = self.get(variable_name)
        if entry is None:
            return ColumnResolution(
                variable_name=variable_name,
                requested_column=requested_column,
            )
        requested = str(requested_column)
        if requested in entry.columns:
            return ColumnResolution(
                variable_name=variable_name,
                requested_column=requested,
                resolved_column=requested,
                reason="exists",
            )
        aliased = entry.alias_map.get(requested)
        if aliased in entry.columns:
            return ColumnResolution(
                variable_name=variable_name,
                requested_column=requested,
                resolved_column=aliased,
                reason="alias",
            )
        fuzzy = fuzzy_match_column(requested, entry.columns)
        if fuzzy:
            return ColumnResolution(
                variable_name=variable_name,
                requested_column=requested,
                resolved_column=fuzzy,
                reason="fuzzy",
            )
        return ColumnResolution(
            variable_name=variable_name,
            requested_column=requested,
        )


_AS_ALIAS_RE = re.compile(
    r"(?P<expr>[^,\n]+?)\s+AS\s+(?P<alias>\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
_QUOTED_IDENTIFIER_RE = re.compile(r'"([^"]+)"|`([^`]+)`|\[([^\]]+)\]')
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_SQL_KEYWORDS = {
    "as",
    "case",
    "cast",
    "coalesce",
    "count",
    "else",
    "end",
    "from",
    "max",
    "min",
    "null",
    "select",
    "sum",
    "then",
    "when",
}


def _strip_identifier_quotes(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and (
        (text[0] == text[-1] == '"')
        or (text[0] == text[-1] == "`")
        or (text[0] == "[" and text[-1] == "]")
    ):
        return text[1:-1]
    return text


def infer_sql_alias_map(sql: str, output_columns: Iterable[str]) -> dict[str, str]:
    """Best-effort source-column to SQL-output alias map.

    This is intentionally conservative; it records aliases only when the alias
    is an actual output column from the executed DataFrame.
    """

    output = {str(col) for col in output_columns}
    if not output:
        return {}

    aliases: dict[str, str] = {}
    for match in _AS_ALIAS_RE.finditer(str(sql or "")):
        alias = _strip_identifier_quotes(match.group("alias"))
        if alias not in output:
            continue
        expr = match.group("expr")
        sources: list[str] = []
        for quoted in _QUOTED_IDENTIFIER_RE.findall(expr):
            source = next((part for part in quoted if part), "")
            if source:
                sources.append(source)
        if not sources:
            for identifier in _IDENTIFIER_RE.findall(expr):
                normalized = identifier.lower()
                if normalized in _SQL_KEYWORDS or identifier == alias:
                    continue
                sources.append(identifier)
        for source in sources:
            if source != alias:
                aliases.setdefault(source, alias)
    return aliases
