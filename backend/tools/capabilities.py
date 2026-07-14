from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field


class RuntimeTableDescriptor(BaseModel):
    table_name: str
    qualified_name: str | None = None
    columns: list[str] = Field(default_factory=list)
    file_name: str | None = None
    display_name: str | None = None
    source_alias: str | None = None
    schema_hint: dict[str, str] = Field(default_factory=dict)
    preprocessing_summary: dict[str, Any] = Field(default_factory=dict)
    row_count: int | None = None
    column_count: int | None = None

    @property
    def display_columns(self) -> list[str]:
        if self.columns:
            return [str(item) for item in self.columns if str(item).strip()]
        return [str(item) for item in self.schema_hint.keys() if str(item).strip()]


class RuntimeTableDescriptorPromptOptions(BaseModel):
    header: str
    table_template: str
    hidden_tables_template: str
    unknown_columns_label: str
    source_text_template: str = " ({raw_sources})"
    rows_label: str = "rows"
    columns_label: str = "columns"
    column_overflow_template: str = "... +{hidden_columns} columns"
    max_tables: int = Field(default=12, ge=1)
    max_columns: int = Field(default=12, ge=1)


_CAPABILITY_TABLE_PROMPT_OPTIONS = RuntimeTableDescriptorPromptOptions(
    header="- Uploaded DuckDB table descriptors:",
    table_template="  - `{table_name}`{source_text}: {columns}{stats_text}",
    hidden_tables_template="  - ... {hidden_tables} more table descriptors omitted",
    unknown_columns_label="unknown columns",
    rows_label="rows",
    columns_label="columns",
    column_overflow_template="... +{hidden_columns} columns",
    max_tables=12,
    max_columns=12,
)

_CAPABILITY_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "table_analysis",
        "label": "табличный анализ",
        "requires_any_tools": ("pandas_tool",),
        "requires_data_source": "dataset",
    },
    {
        "key": "db_query",
        "label": "SQL-анализ БД",
        "requires_any_tools": ("sql_tool", "database_tool"),
        "requires_data_source": "db",
    },
    {
        "key": "charting",
        "label": "построение графиков",
        "requires_any_tools": ("plotly_tool",),
        "requires_data_source": "any_data",
    },
    {
        "key": "forecasting",
        "label": "прогнозирование",
        "requires_any_tools": ("forecast_tool",),
        "requires_data_source": "any_data",
    },
    {
        "key": "anomaly_planfact",
        "label": "анализ аномалий и план-факт",
        "requires_any_tools": ("anomaly_planfact_tool",),
        "requires_data_source": "any_data",
    },
    {
        "key": "external_search",
        "label": "внешний поиск",
        "requires_any_tools": ("search_tool",),
        "requires_data_source": None,
    },
    {
        "key": "knowledge_base_search",
        "label": "поиск по базе знаний",
        "requires_any_tools": ("rag_tool",),
        "requires_data_source": None,
    },
)


def _tool_list(items: Iterable[str]) -> list[str]:
    return sorted({str(item).strip() for item in items if str(item).strip()})


def coerce_runtime_table_descriptors(items: Iterable[Any] | None) -> list[RuntimeTableDescriptor]:
    descriptors: list[RuntimeTableDescriptor] = []
    for item in items or []:
        try:
            if isinstance(item, RuntimeTableDescriptor):
                descriptor = item
            else:
                descriptor = RuntimeTableDescriptor.model_validate(item)
        except Exception:
            continue
        if descriptor.table_name.strip():
            descriptors.append(descriptor)
    return descriptors


def _deduplicated_descriptor_sources(descriptor: RuntimeTableDescriptor) -> list[str]:
    sources: list[str] = []
    for value in (
        descriptor.display_name,
        descriptor.file_name,
        descriptor.source_alias,
    ):
        clean = str(value or "").strip()
        if clean and clean not in sources:
            sources.append(clean)
    return sources


def format_runtime_table_descriptors(
    descriptors: list[RuntimeTableDescriptor],
    options: RuntimeTableDescriptorPromptOptions,
) -> list[str]:
    if not descriptors:
        return []

    lines = [options.header]
    for descriptor in descriptors[: options.max_tables]:
        table_name = descriptor.qualified_name or descriptor.table_name
        source_bits = _deduplicated_descriptor_sources(descriptor)
        sources = ", ".join(f"`{source}`" for source in source_bits)
        raw_sources = ", ".join(source_bits)
        source_text = (
            options.source_text_template.format(
                sources=sources,
                raw_sources=raw_sources,
            )
            if source_bits
            else ""
        )
        columns = descriptor.display_columns
        shown_columns = columns[: options.max_columns]
        columns_text = (
            ", ".join(f"`{column}`" for column in shown_columns)
            if shown_columns
            else options.unknown_columns_label
        )
        hidden_columns = len(columns) - len(shown_columns)
        if hidden_columns > 0:
            columns_text = (
                f"{columns_text}, "
                f"{options.column_overflow_template.format(hidden_columns=hidden_columns)}"
            )
        stats: list[str] = []
        if descriptor.row_count is not None:
            stats.append(f"{descriptor.row_count} {options.rows_label}")
        if descriptor.column_count is not None:
            stats.append(f"{descriptor.column_count} {options.columns_label}")
        stats_text = f"; {', '.join(stats)}" if stats else ""
        lines.append(
            options.table_template.format(
                table_name=table_name,
                sources=sources,
                source_text=source_text,
                stats=", ".join(stats),
                stats_text=stats_text,
                columns=columns_text,
            )
        )
    hidden_tables = len(descriptors) - options.max_tables
    if hidden_tables > 0:
        lines.append(options.hidden_tables_template.format(hidden_tables=hidden_tables))
    return lines


def _has_required_data_source(
    requirement: str | None,
    *,
    has_dataframe: bool,
    has_db_source: bool,
    has_knowledge_base: bool,
) -> bool:
    if requirement is None:
        return True
    if requirement == "dataset":
        return has_dataframe
    if requirement == "db":
        return has_db_source
    if requirement == "knowledge_base":
        return has_knowledge_base
    if requirement == "any_data":
        return has_dataframe or has_db_source
    return False


def build_runtime_capability_context(
    *,
    available_tool_keys: Iterable[str],
    has_dataframe: bool,
    has_db_source: bool,
    has_knowledge_base: bool = False,
    csv_table_names: list[str] | None = None,
    csv_table_descriptors: list[RuntimeTableDescriptor | dict[str, Any]] | None = None,
    source_table_count: int = 0,
    source_count: int = 0,
) -> dict[str, Any]:
    tool_keys = _tool_list(available_tool_keys)
    tool_key_set = set(tool_keys)

    capabilities: list[dict[str, Any]] = []
    available_capability_keys: list[str] = []
    unavailable_capability_keys: list[str] = []
    available_labels: list[str] = []
    unavailable_labels: list[str] = []

    for spec in _CAPABILITY_SPECS:
        available = _has_required_data_source(
            spec["requires_data_source"],
            has_dataframe=has_dataframe,
            has_db_source=has_db_source,
            has_knowledge_base=has_knowledge_base,
        ) and any(tool in tool_key_set for tool in spec["requires_any_tools"])
        payload = {
            "key": spec["key"],
            "label": spec["label"],
            "available": available,
            "requires_any_tools": list(spec["requires_any_tools"]),
        }
        capabilities.append(payload)
        if available:
            available_capability_keys.append(spec["key"])
            available_labels.append(spec["label"])
        else:
            unavailable_capability_keys.append(spec["key"])
            unavailable_labels.append(spec["label"])

    source_mode = (
        "db"
        if has_db_source
        else "dataset"
        if has_dataframe
        else "knowledge_base"
        if has_knowledge_base
        else "none"
    )
    available_tools_text = ", ".join(f"`{item}`" for item in tool_keys) if tool_keys else "нет"
    available_caps_text = ", ".join(available_labels) if available_labels else "нет"
    unavailable_caps_text = ", ".join(unavailable_labels) if unavailable_labels else "нет"

    lines = [
        "[ROLE: CAPABILITIES]",
        f"- Активный режим данных: `{source_mode}`",
        f"- Доступные tools: {available_tools_text}",
        f"- Доступные capabilities: {available_caps_text}",
        f"- Недоступные capabilities: {unavailable_caps_text}",
    ]
    unavailable_capabilities = [
        capability for capability in capabilities if not bool(capability.get("available"))
    ]
    if unavailable_capabilities:
        lines.append(
            "- If the user asks for an unavailable capability, do not approximate it "
            "manually and do not use substitute tools. Answer directly that the "
            "capability cannot be performed because required tool(s) are disabled "
            "or unavailable. Use the capability label and required tool names from "
            "this capability list."
        )
        lines.append("- Unavailable capability details:")
        for capability in unavailable_capabilities:
            required_tools = ", ".join(
                f"`{tool}`" for tool in capability.get("requires_any_tools", [])
            )
            lines.append(
                f"  - {capability['label']} (`{capability['key']}`); "
                f"required tools: {required_tools}"
            )
    if csv_table_names:
        tables_str = ", ".join(f"`{t}`" for t in csv_table_names)
        lines.append(f"- Таблицы в DuckDB: {tables_str}")
    table_descriptors = coerce_runtime_table_descriptors(csv_table_descriptors)
    lines.extend(format_runtime_table_descriptors(table_descriptors, _CAPABILITY_TABLE_PROMPT_OPTIONS))
    if (
        "data_catalog_tool" in tool_key_set
        and (source_table_count > 1 or source_count > 1)
    ):
        lines.append(
            "- CATALOG-FIRST: multiple sources or tables are available. "
            "Call `data_catalog_tool(action=\"list_tables\")` or "
            "`data_catalog_tool(action=\"describe_table\", table=\"<qualified_name>\")` "
            "before SQL/dataframe work when table choice is not explicit. "
            "Use exact `qualified_name` values; do not guess from bare table names."
        )
    lines.append(
        "- Нельзя обещать действия, требующие недоступного capability. "
        "Если нужный capability недоступен, честно сообщи об ограничении и предложи ближайшую доступную альтернативу."  # noqa: E501
    )
    prompt_block = "\n".join(lines)

    return {
        "source_mode": source_mode,
        "available_tool_keys": tool_keys,
        "capabilities": capabilities,
        "available_capability_keys": available_capability_keys,
        "unavailable_capability_keys": unavailable_capability_keys,
        "prompt_block": prompt_block,
    }
