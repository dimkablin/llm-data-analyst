from __future__ import annotations

from typing import Any, Iterable


_CAPABILITY_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "table_analysis",
        "label": "табличный анализ",
        "requires_any_tools": ("pandas_tool", "value_tool"),
        "requires_data_source": "dataset",
    },
    {
        "key": "db_query",
        "label": "SQL-анализ БД",
        "requires_any_tools": ("db_tool",),
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
        "key": "deep_research",
        "label": "глубокое исследование",
        "requires_any_tools": ("deep_research_tool",),
        "requires_data_source": None,
    },
)


def _tool_list(items: Iterable[str]) -> list[str]:
    return sorted({str(item).strip() for item in items if str(item).strip()})


def _has_required_data_source(requirement: str | None, *, has_dataframe: bool, has_db_source: bool) -> bool:
    if requirement is None:
        return True
    if requirement == "dataset":
        return has_dataframe
    if requirement == "db":
        return has_db_source
    if requirement == "any_data":
        return has_dataframe or has_db_source
    return False


def build_runtime_capability_context(
    *,
    available_tool_keys: Iterable[str],
    has_dataframe: bool,
    has_db_source: bool,
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

    source_mode = "db" if has_db_source else "dataset" if has_dataframe else "none"
    available_tools_text = ", ".join(f"`{item}`" for item in tool_keys) if tool_keys else "нет"
    available_caps_text = ", ".join(available_labels) if available_labels else "нет"
    unavailable_caps_text = ", ".join(unavailable_labels) if unavailable_labels else "нет"

    prompt_block = "\n".join(
        [
            "[ROLE: CAPABILITIES]",
            f"- Активный режим данных: `{source_mode}`",
            f"- Доступные tools: {available_tools_text}",
            f"- Доступные capabilities: {available_caps_text}",
            f"- Недоступные capabilities: {unavailable_caps_text}",
            (
                "- Нельзя обещать действия, требующие недоступного capability. "
                "Если нужный capability недоступен, честно сообщи об ограничении и предложи ближайшую доступную альтернативу."
            ),
        ]
    )

    return {
        "source_mode": source_mode,
        "available_tool_keys": tool_keys,
        "capabilities": capabilities,
        "available_capability_keys": available_capability_keys,
        "unavailable_capability_keys": unavailable_capability_keys,
        "prompt_block": prompt_block,
    }
