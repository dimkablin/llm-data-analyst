from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from backend.instructions import InstructionDocument
from backend.tools.instructions import ToolInstructionError, get_default_tool_instruction_registry


@dataclass(frozen=True)
class ToolCatalogSpec:
    tool_key: str
    tool_label: str
    display_name_ru: str
    description: str
    description_ru: str
    capabilities: tuple[str, ...]
    requires_session_data: bool
    kind: str
    source_type: str | None = None
    enabled_by_default: bool = True


BUILTIN_TOOL_SPECS: tuple[ToolCatalogSpec, ...] = (
    ToolCatalogSpec(
        tool_key="planner_tool",
        tool_label="Planner",
        display_name_ru="Планировщик",
        description="On-demand analysis plan generation for complex multi-step queries.",
        description_ru="Составление плана анализа для сложных многошаговых запросов.",
        capabilities=("planning",),
        requires_session_data=False,
        kind="builtin",
    ),
    ToolCatalogSpec(
        tool_key="generate_summary_tool",
        tool_label="Generate Summary",
        display_name_ru="Generate summary",
        description=(
            "Generic summary generation from current session history, notes, and "
            "artifact summaries. Does not calculate new metrics."
        ),
        description_ru=(
            "Generic summary generation from current session history, notes, and "
            "artifact summaries. Does not calculate new metrics."
        ),
        capabilities=("summary_generation", "session_context"),
        requires_session_data=False,
        kind="builtin",
    ),
    ToolCatalogSpec(
        tool_key="generate_report_tool",
        tool_label="Generate Report",
        display_name_ru="Generate report",
        description=(
            "DOCX report export from persisted session chat history and artifacts."
        ),
        description_ru=(
            "DOCX report export from persisted session chat history and artifacts."
        ),
        capabilities=("report_export", "docx"),
        requires_session_data=False,
        kind="builtin",
    ),
    ToolCatalogSpec(
        tool_key="data_catalog_tool",
        tool_label="Data Catalog",
        display_name_ru="Data catalog",
        description=(
            "Structured session inventory: list sources, list tables, search columns, "
            "and disambiguate duplicate table names before using analytical tools."
        ),
        description_ru=(
            "Structured session inventory: sources, tables, columns, and duplicate "
            "table-name disambiguation before SQL or dataframe work."
        ),
        capabilities=("source_catalog", "schema_discovery", "table_disambiguation"),
        requires_session_data=True,
        kind="builtin",
    ),
    ToolCatalogSpec(
        tool_key="sql_tool",
        tool_label="SQL tool",
        display_name_ru="SQL по таблицам",
        description=(
            "Primary entry point for tabular data extraction from attached DB and/or CSV-in-DuckDB: "
            "pick table, generate safe SELECT, return a named table artifact. "
            "Use this first when data must be fetched from the database."
        ),
        description_ru=(
            "Основной вход для получения табличных данных из подключённой БД и/или CSV в DuckDB: "
            "выбор таблицы, безопасный SELECT, именованный табличный артефакт. "
            "Используй его первым, когда нужно получить данные из БД."
        ),
        capabilities=("read_only_sql", "table_artifact", "nl_to_sql"),
        requires_session_data=True,
        kind="builtin",
    ),
    ToolCatalogSpec(
        tool_key="database_tool",
        tool_label="Database Tool",
        display_name_ru="Структура БД",
        description="Lightweight DB catalog queries: list tables, describe columns, preview rows, list schemas. No LLM-generated SQL.",  # noqa: E501
        description_ru="Быстрый просмотр структуры БД: список таблиц, колонки, превью строк, схемы. Без генерации SQL.",  # noqa: E501
        capabilities=("db_catalog", "table_artifact", "db_preview"),
        requires_session_data=True,
        kind="builtin",
    ),
    ToolCatalogSpec(
        tool_key="pandas_tool",
        tool_label="Pandas Tool",
        display_name_ru="Табличная обработка",
        description=(
            "Tabular transformations and aggregations over dataframe variables already present in the session sandbox."  # noqa: E501
            "Does not fetch data from the database directly."
        ),
        description_ru=(
            "Табличные преобразования, группировки и вычисления по датафреймам, "
            "которые уже лежат в sandbox текущей сессии. "
            "Не получает данные из БД напрямую."
        ),
        capabilities=("dataframe_transform", "aggregation", "table_artifact"),
        requires_session_data=True,
        kind="builtin",
    ),
    ToolCatalogSpec(
        tool_key="plotly_tool",
        tool_label="Plotly Tool",
        display_name_ru="Графики",
        description=(
            "Chart creation from dataframe variables already present in the session sandbox. "
            "Use after sql_tool or pandas_tool. Does not fetch data from the database directly."
        ),
        description_ru=(
            "Построение графиков по датафреймам, которые уже лежат в sandbox текущей сессии. "
            "Используй после sql_tool или pandas_tool. Не получает данные из БД напрямую."
        ),
        capabilities=("chart", "plotly", "chart_artifact"),
        requires_session_data=True,
        kind="builtin",
    ),
)

INTEGRATION_TOOL_SPECS: tuple[ToolCatalogSpec, ...] = (
    ToolCatalogSpec(
        tool_key="search_tool",
        tool_label="Search",
        display_name_ru="Поиск",
        description="External quick search integration.",
        description_ru="Быстрый внешний поиск по теме пользователя.",
        capabilities=("search", "web_results"),
        requires_session_data=False,
        kind="integration",
        source_type="search",
    ),
    ToolCatalogSpec(
        tool_key="rag_tool",
        tool_label="RAG",
        display_name_ru="База знаний",
        description="Semantic retrieval from the configured indexed knowledge base with source references.",
        description_ru=(
            "Семантический поиск по настроенной индексированной базе знаний "
            "с ссылками на источники."
        ),
        capabilities=("knowledge_base_search", "document_answer"),
        requires_session_data=False,
        kind="integration",
        source_type="rag",
    ),
    ToolCatalogSpec(
        tool_key="forecast_tool",
        tool_label="Forecast",
        display_name_ru="Прогноз",
        description="External forecasting over compact prepared time series.",
        description_ru="Прогнозирование по компактным временным рядам из данных сессии.",
        capabilities=("forecast", "time_series_forecast"),
        requires_session_data=True,
        kind="integration",
        source_type="forecast",
    ),
    ToolCatalogSpec(
        tool_key="anomaly_planfact_tool",
        tool_label="Anomaly / Plan-fact",
        display_name_ru="План-факт и аномалии",
        description="External anomaly and plan-fact analysis over aligned plan/fact time series.",
        description_ru="Анализ отклонений, план-факт и поиск аномалий по выровненным временным рядам.",
        capabilities=("anomaly_detection", "plan_fact_analysis"),
        requires_session_data=True,
        kind="integration",
        source_type="anomaly_planfact",
    ),
)

ALL_TOOL_SPECS: tuple[ToolCatalogSpec, ...] = BUILTIN_TOOL_SPECS + INTEGRATION_TOOL_SPECS
KNOWN_TOOL_KEYS: frozenset[str] = frozenset(spec.tool_key for spec in ALL_TOOL_SPECS)


def _tool_instruction(tool_key: str) -> InstructionDocument:
    document = get_default_tool_instruction_registry().get_optional(tool_key)
    if document is None:
        raise ToolInstructionError(f"Missing TOOL.md for registered tool: {tool_key}")
    return document


def _tool_description(spec: ToolCatalogSpec) -> str:
    document = _tool_instruction(spec.tool_key)
    return document.metadata.description or spec.description


def _tool_enabled_by_default(spec: ToolCatalogSpec) -> bool:
    document = _tool_instruction(spec.tool_key)
    return bool(document.metadata.enabled_by_default)


def build_tool_catalog(
    *,
    source_descriptors: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    user_settings: Mapping[str, bool] | None = None,
) -> list[dict[str, Any]]:
    source_by_type = {
        str(item.get("source_type")): dict(item)
        for item in source_descriptors
        if isinstance(item, dict) and item.get("source_type")
    }
    raw_user_settings = dict(user_settings or {})

    catalog: list[dict[str, Any]] = []
    for spec in BUILTIN_TOOL_SPECS:
        enabled_by_default = _tool_enabled_by_default(spec)
        description = _tool_description(spec)
        enabled_for_user = bool(raw_user_settings.get(spec.tool_key, enabled_by_default))
        catalog.append(
            {
                "tool_key": spec.tool_key,
                "kind": spec.kind,
                "tool_label": spec.tool_label,
                "display_name_ru": spec.display_name_ru,
                "description": description,
                "description_ru": spec.description_ru,
                "capabilities": list(spec.capabilities),
                "requires_session_data": spec.requires_session_data,
                "source_type": None,
                "source_ref_id": None,
                "source_mode": "runtime",
                "enabled_globally": True,
                "available_globally": True,
                "status": "available",
                "enabled_by_default": enabled_by_default,
                "enabled_for_user": enabled_for_user,
                "effective_enabled": enabled_for_user,
                "timeout_hint_sec": None,
            }
        )

    for spec in INTEGRATION_TOOL_SPECS:
        source_descriptor = source_by_type.get(spec.source_type or "", {})
        enabled_globally = bool(source_descriptor.get("enabled", False))
        available_globally = bool(source_descriptor.get("available", False))
        enabled_by_default = _tool_enabled_by_default(spec)
        description = _tool_description(spec)
        enabled_for_user = bool(raw_user_settings.get(spec.tool_key, enabled_by_default))
        catalog.append(
            {
                "tool_key": spec.tool_key,
                "kind": spec.kind,
                "tool_label": str(source_descriptor.get("source_label") or spec.tool_label),
                "display_name_ru": str(
                    source_descriptor.get("display_name_ru") or spec.display_name_ru
                ),
                "description": str(source_descriptor.get("description") or description),
                "description_ru": str(
                    source_descriptor.get("description_ru") or spec.description_ru
                ),
                "capabilities": list(source_descriptor.get("capabilities") or spec.capabilities),
                "requires_session_data": bool(
                    source_descriptor.get("requires_session_data", spec.requires_session_data)
                ),
                "source_type": source_descriptor.get("source_type") or spec.source_type,
                "source_ref_id": source_descriptor.get("source_ref_id"),
                "source_mode": source_descriptor.get("source_mode"),
                "enabled_globally": enabled_globally,
                "available_globally": available_globally,
                "status": str(source_descriptor.get("status") or "disabled"),
                "enabled_by_default": enabled_by_default,
                "enabled_for_user": enabled_for_user,
                "effective_enabled": enabled_for_user and available_globally,
                "timeout_hint_sec": source_descriptor.get("timeout_hint_sec"),
            }
        )

    return catalog
