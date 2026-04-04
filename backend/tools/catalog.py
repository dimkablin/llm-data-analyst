from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


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
        tool_key="review_tool",
        tool_label="Reviewer",
        display_name_ru="Проверка ответа",
        description="Hybrid quality check for agent answers: heuristics + optional LLM review.",
        description_ru="Гибридная проверка качества: эвристики + LLM-ревью для сложных запросов.",
        capabilities=("quality_check",),
        requires_session_data=False,
        kind="builtin",
    ),
    ToolCatalogSpec(
        tool_key="sql_tool",
        tool_label="SQL tool",
        display_name_ru="SQL по таблицам",
        description="Natural-language questions over attached DB and/or CSV-in-DuckDB: table pick, safe SELECT, tabular artifact.",  # noqa: E501
        description_ru="Вопросы на естественном языке по привязанной БД и/или CSV в DuckDB: выбор таблицы, безопасный SELECT, табличный артефакт.",  # noqa: E501
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
        description="Tabular transformations and aggregations over the active dataframe session data.",
        description_ru="Табличные преобразования, группировки и вычисления по данным текущей сессии.",
        capabilities=("dataframe_transform", "aggregation", "table_artifact"),
        requires_session_data=True,
        kind="builtin",
    ),
    ToolCatalogSpec(
        tool_key="plotly_tool",
        tool_label="Plotly Tool",
        display_name_ru="Графики",
        description="Chart creation and plot artifacts from tabular data (CSV dataframe or SQL query result).",  # noqa: E501
        description_ru="Построение графиков по табличным данным (CSV датафрейм или результат SQL-запроса).",
        capabilities=("chart", "plotly", "chart_artifact"),
        requires_session_data=True,
        kind="builtin",
    ),
    ToolCatalogSpec(
        tool_key="value_tool",
        tool_label="Value Tool",
        display_name_ru="Метрики",
        description="Scalar metrics and compact numeric/text outputs from session data.",
        description_ru="Быстрые одиночные метрики и компактные числовые результаты по данным сессии.",
        capabilities=("scalar_metric", "value_artifact"),
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
        description="External knowledge-base retrieval and answer generation via RAG.",
        description_ru="Поиск и ответ по внешней базе знаний через RAG.",
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
        enabled_for_user = bool(raw_user_settings.get(spec.tool_key, True))
        catalog.append(
            {
                "tool_key": spec.tool_key,
                "kind": spec.kind,
                "tool_label": spec.tool_label,
                "display_name_ru": spec.display_name_ru,
                "description": spec.description,
                "description_ru": spec.description_ru,
                "capabilities": list(spec.capabilities),
                "requires_session_data": spec.requires_session_data,
                "source_type": None,
                "source_ref_id": None,
                "source_mode": "runtime",
                "enabled_globally": True,
                "available_globally": True,
                "status": "available",
                "enabled_for_user": enabled_for_user,
                "effective_enabled": enabled_for_user,
                "timeout_hint_sec": None,
            }
        )

    for spec in INTEGRATION_TOOL_SPECS:
        source_descriptor = source_by_type.get(spec.source_type or "", {})
        enabled_globally = bool(source_descriptor.get("enabled", False))
        available_globally = bool(source_descriptor.get("available", False))
        enabled_for_user = bool(raw_user_settings.get(spec.tool_key, True))
        catalog.append(
            {
                "tool_key": spec.tool_key,
                "kind": spec.kind,
                "tool_label": str(source_descriptor.get("source_label") or spec.tool_label),
                "display_name_ru": str(
                    source_descriptor.get("display_name_ru") or spec.display_name_ru
                ),
                "description": str(source_descriptor.get("description") or spec.description),
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
                "enabled_for_user": enabled_for_user,
                "effective_enabled": enabled_for_user and available_globally,
                "timeout_hint_sec": source_descriptor.get("timeout_hint_sec"),
            }
        )

    return catalog


