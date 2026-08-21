"""Domain-neutral tabular context hints for agent prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

_TIME_NAME_RE = re.compile(
    r"(date|time|month|year|week|day|period|timestamp|dt\b)",
    re.IGNORECASE,
)
_METRIC_NAME_RE = re.compile(
    r"(revenue|sales|amount|count|sum|total|avg|mean|median|price|volume|qty|quantity|"
    r"margin|profit|score|rate|traffic|conversion|orders?)",
    re.IGNORECASE,
)
_PLAN_NAME_RE = re.compile(
    r"(plan|target|budget|forecast)",
    re.IGNORECASE,
)
_ID_NAME_RE = re.compile(
    r"(^id$|_id$|uuid|guid|sku|code)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ColumnRoles:
    time: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()
    identifiers: tuple[str, ...] = ()
    plan_metrics: tuple[str, ...] = ()


def infer_column_roles(df: pd.DataFrame, *, max_columns: int = 40) -> ColumnRoles:
    """Infer generic column roles for prompt guidance.

    The result is intentionally advisory. The agent must still verify schema and
    values with tools before making claims.
    """
    time_cols: list[str] = []
    metric_cols: list[str] = []
    dimension_cols: list[str] = []
    id_cols: list[str] = []
    plan_cols: list[str] = []

    row_count = len(df)
    for col in list(df.columns)[:max_columns]:
        name = str(col)
        series = df[col]
        unique = int(series.nunique(dropna=True)) if row_count else 0

        if is_datetime64_any_dtype(series) or _TIME_NAME_RE.search(name):
            time_cols.append(name)
            continue

        if is_numeric_dtype(series):
            if _ID_NAME_RE.search(name):
                id_cols.append(name)
            elif _PLAN_NAME_RE.search(name):
                plan_cols.append(name)
            elif _METRIC_NAME_RE.search(name) or unique > min(20, max(5, row_count // 20)):
                metric_cols.append(name)
            elif unique <= 30:
                dimension_cols.append(name)
            else:
                metric_cols.append(name)
            continue

        if row_count >= 20 and unique >= max(row_count * 0.92, row_count - 2):
            id_cols.append(name)
            continue

        if _ID_NAME_RE.search(name):
            id_cols.append(name)
        else:
            dimension_cols.append(name)

    return ColumnRoles(
        time=tuple(time_cols),
        metrics=tuple(metric_cols),
        dimensions=tuple(dimension_cols),
        identifiers=tuple(id_cols),
        plan_metrics=tuple(plan_cols),
    )


def _format_roles_block(roles: ColumnRoles) -> str:
    lines = ["### Column Roles (advisory; verify with schema/tools)"]
    mapping = [
        ("Time / period", roles.time),
        ("Metrics for SUM/AVG/COUNT", roles.metrics),
        ("Dimensions for GROUP BY", roles.dimensions),
        ("Plan / target metrics", roles.plan_metrics),
        ("Identifiers / keys, do not aggregate as measures", roles.identifiers),
    ]
    for label, cols in mapping:
        if cols:
            lines.append(f"- {label}: {', '.join(cols)}")
    if len(lines) == 1:
        lines.append("- No obvious roles inferred. Inspect schema and sample values first.")
    return "\n".join(lines)


def build_universal_analytics_playbook(
    *,
    dataset_name: str = "",
    df: pd.DataFrame | None = None,
    session_source: dict[str, Any] | None = None,
    db_name: str = "",
    db_type: str = "",
    db_schema: str = "",
) -> str:
    """Build source-aware but domain-neutral prompt context for tabular analysis."""
    source = session_source or {}
    source_type = str(source.get("source_type") or "").strip().lower()
    lines: list[str] = ["## Universal Analytics Context"]

    if source_type in {"db_connection", "openproject"} or db_name:
        source_label = "OpenProject PostgreSQL" if source_type == "openproject" else "database"
        lines.append(f"### Source: {source_label}")
        if db_name:
            type_suffix = f" ({db_type})" if db_type else ""
            lines.append(f"- Connection: {db_name}{type_suffix}")
        configured_schema = str(db_schema or (source.get("db_options") or {}).get("schema") or "").strip()
        if configured_schema:
            lines.append(f"- Configured schema: `{configured_schema}`.")
        else:
            lines.append("- Schema is not configured; list schemas/tables before SQL.")
        lines.append("- Use database_tool/sql_tool for schema inspection and aggregation.")
    elif source.get("csv_loaded") or df is not None:
        tables = source.get("csv_table_names") or []
        label = dataset_name or str(source.get("source_label") or "").strip()
        lines.append("### Source: tabular file / session table")
        if label:
            lines.append(f"- Dataset label: {label}")
        if tables:
            lines.append(f"- Session tables: {', '.join(str(table) for table in tables)}")
        lines.append("- Prefer SQL for aggregations; use pandas/plotly for derived tables and charts.")

    if df is not None and not df.empty:
        lines.append(_format_roles_block(infer_column_roles(df)))

    lines.extend(
        [
            "",
            "### Generic Workflow",
            "| User task | Runtime behavior |",
            "|---|---|",
            "| Understand available data | inspect schema, table names, columns, sample values |",
            "| Trend over time | group by a verified time column and aggregate verified metrics |",
            "| Top-N or ranking | aggregate first, then order by the requested metric |",
            "| Segment comparison | group by the requested dimension and compare the same metric |",
            "| Composition / share | compute numerator and denominator explicitly from the same source |",
            "| Plan vs actual | compare verified plan/target and actual metrics; call tools when available |",
            "| Forecast | resolve the specialized forecasting capability "
            "through the current active capability catalog |",
            "| Report or summary | use generate_summary_tool/generate_report_tool when requested |",
            "",
            "### Required Rules",
            "- Use only table and column names present in schema/context.",
            "- File names are labels, not values in a column.",
            "- Preview/sample rows are not the full dataset; absence claims require full-source counts.",
            "- Percent/share metrics must state their denominator.",
            "- Numbers in the final answer must come from tool output.",
            "- Load specialized instructions only for a concrete workflow gap; "
            "the active prompt already contains the base workflow.",
            "- Final answer structure: essence, key numbers, insights, artifacts, what to verify.",
        ]
    )
    return "\n".join(lines).strip()


def build_sql_generation_hints(columns: object, *, db_schema: str = "") -> str:
    """Extra SQL generator hints that stay independent from business domains."""
    _ = db_schema
    hints = [
        "\n- use only listed table and column names",
        "\n- aggregate with GROUP BY + SUM/AVG/COUNT; avoid SELECT * on large tables",
        "\n- for share calculations, compute numerator and denominator explicitly",
        "\n- do not SUM identifiers or near-unique keys",
    ]
    column_names = [str(column) for column in columns or []]
    if column_names:
        shown = ", ".join(f"`{name}`" for name in column_names[:40])
        hidden = len(column_names) - 40
        suffix = f", ... +{hidden} columns" if hidden > 0 else ""
        hints.append(f"\n- exact column names: {shown}{suffix}")
    if any(not name.isascii() for name in column_names):
        hints.append("\n- quote non-ASCII identifiers according to the SQL dialect")
    return "".join(hints)


def build_dataset_profile_block(
    df: pd.DataFrame | None = None,
    *,
    dataset_name: str = "",
    session_source: dict[str, Any] | None = None,
    db_name: str = "",
    db_type: str = "",
    db_schema: str = "",
) -> str:
    """Return generic source context; domain behavior belongs to skills/tools."""
    source = session_source or {}
    has_tabular_source = (
        df is not None
        or bool(source.get("csv_loaded"))
        or str(source.get("source_type") or "").lower() == "db_connection"
        or str(source.get("source_type") or "").lower() == "openproject"
        or str(source.get("source_mode") or "").lower() == "postgres_sync"
        or bool(db_name)
    )
    if not has_tabular_source:
        return ""

    return build_universal_analytics_playbook(
        dataset_name=dataset_name,
        df=df,
        session_source=source,
        db_name=db_name,
        db_type=db_type,
        db_schema=db_schema,
    )
