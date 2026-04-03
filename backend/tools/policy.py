from __future__ import annotations

from typing import Any, Iterable

DATAFRAME_BASE_TOOL_KEYS: frozenset[str] = frozenset(
    {"pandas_tool", "value_tool", "plotly_tool"}
)
DB_BASE_TOOL_KEYS: frozenset[str] = frozenset({"sql_tool", "database_tool"})
ARTIFACT_OPTIONAL_TOOL_KEYS: frozenset[str] = frozenset({"search_tool", "memory"})


def normalize_allowed_tool_keys(
    allowed_tool_keys: Iterable[str] | None,
) -> set[str] | None:
    if allowed_tool_keys is None:
        return None
    return {str(item).strip() for item in allowed_tool_keys if str(item).strip()}


def is_tool_allowed(tool_key: str, allowed_tool_keys: set[str] | None) -> bool:
    if allowed_tool_keys is None:
        return True
    return str(tool_key or "").strip() in allowed_tool_keys


def normalize_tool_names(tool_names: Iterable[str] | None) -> set[str]:
    if tool_names is None:
        return set()
    return {str(item).strip() for item in tool_names if str(item).strip()}


def supports_artifact_optional_output(tool_names: Iterable[str] | None) -> bool:
    normalized = normalize_tool_names(tool_names)
    return bool(normalized) and normalized.issubset(ARTIFACT_OPTIONAL_TOOL_KEYS)


def effective_enabled_tool_keys(catalog_rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(item["tool_key"])
        for item in catalog_rows
        if bool(item.get("effective_enabled"))
    }


def detect_data_access_mode(
    *,
    has_dataframe: bool,
    session_source: dict[str, Any] | None,
) -> str | None:
    source_type = ""
    if isinstance(session_source, dict):
        source_type = str(session_source.get("source_type", "")).strip().lower()
    if source_type == "db_connection":
        return "db"
    if has_dataframe:
        return "dataset"
    return None


def required_data_tool_keys(
    *,
    has_dataframe: bool,
    session_source: dict[str, Any] | None,
) -> frozenset[str]:
    mode = detect_data_access_mode(
        has_dataframe=has_dataframe,
        session_source=session_source,
    )
    if mode == "db":
        return DB_BASE_TOOL_KEYS
    if mode == "dataset":
        return DATAFRAME_BASE_TOOL_KEYS
    return frozenset()


def has_enabled_data_tools(
    *,
    has_dataframe: bool,
    session_source: dict[str, Any] | None,
    allowed_tool_keys: set[str] | None,
) -> bool:
    required = required_data_tool_keys(
        has_dataframe=has_dataframe,
        session_source=session_source,
    )
    if not required:
        return True
    return any(is_tool_allowed(tool_key, allowed_tool_keys) for tool_key in required)


