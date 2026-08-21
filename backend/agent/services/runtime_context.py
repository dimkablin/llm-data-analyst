from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backend.agent.dataset_profiles import build_dataset_profile_block
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.session_source import is_duckdb_source_type
from backend.tools.capabilities import (
    RuntimeTableDescriptorPromptOptions,
    coerce_runtime_table_descriptors,
    format_runtime_table_descriptors,
)


def is_rag_session_source(session_source: dict[str, Any] | None) -> bool:
    if not isinstance(session_source, dict):
        return False
    source_type = str(session_source.get("source_type") or "").strip().lower()
    source_mode = str(session_source.get("source_mode") or "").strip().lower()
    return source_type == "rag" or source_mode == "lightrag"


def build_rag_session_prompt_block(session_source: dict[str, Any] | None) -> str:
    if not is_rag_session_source(session_source):
        return ""
    source_label = str((session_source or {}).get("source_label") or "").strip()
    label = source_label or "Knowledge base"
    return (
        "[KNOWLEDGE_BASE_CONTEXT]\n"
        f"Knowledge source: {label}.\n"
        "For interpretive requests, use `rag_tool`. Do not infer facts from CSV, SQL or pandas. "
        "Use RAG only as the factual source.\n"
        "If `rag_tool` returns no relevant context, do not invent a definition: "
        "ask a clarifying question or request a more precise term/query."
    )

def is_planfact_session_source(session_source: dict[str, Any] | None) -> bool:
    if not isinstance(session_source, dict):
        return False
    return str(session_source.get("source_type") or "").strip().lower() == "planfact"


def build_planfact_session_prompt_block(session_source: dict[str, Any] | None) -> str:
    if not is_planfact_session_source(session_source):
        return ""
    return (
        "[PLANFACT_SOURCE]\n"
        "Источник `planfact` активен.\n"
        "Для обзора по ЦФО используй `planfact_by_cfo_period`.\n"
        "Для детализации по статьям используй `planfact_by_cfo_article_period`.\n"
        "Для детализации причин используй описательные поля `service_content`, `plan_counterparty`, "
        "`fact_counterparty`, `fact_contract`; не используй их как ключи соединения.\n"
        "Отклонение = `fact_amount - plan_amount`.\n"
        "Положительное отклонение значит факт выше плана.\n"
        "Топ отклонений сортируй по `ABS(variance_amount)`.\n"
        "Факт без плана: `plan_amount = 0 AND fact_amount <> 0`.\n"
        "План без факта: `fact_amount = 0 AND plan_amount <> 0`."
    )


def build_planfact_period_prompt_block() -> str:
    return (
        "[PLANFACT_PERIOD_RULE]\n"
        "For every Plan-Fact comparison, by default include only periods that are actually present "
        "in fact data: `period IN (SELECT DISTINCT period FROM planfact_fact_monthly)`. "
        "Apply this same period filter to both plan and fact before aggregating; never include plan-only "
        "months in a Plan-Fact total. If the user asks to analyze only the plan (without comparing it "
        "to fact), all available plan periods may be used. If a requested Plan-Fact period has no fact "
        "data, state that fact is unavailable instead of treating it as a valid comparison."
    )


CHAT_TABLE_PROMPT_OPTIONS = RuntimeTableDescriptorPromptOptions(
    header=(
        "Загруженные файлы доступны как отдельные таблицы DuckDB. "
        "Описывай каждый файл отдельно; не объединяй схемы в один общий датасет без запроса."
    ),
    table_template="- `{table_name}`{source_text}{stats_text}; колонки: {columns}.",
    hidden_tables_template="- Еще таблиц скрыто из краткого контекста: {hidden_tables}.",
    unknown_columns_label="колонки не указаны",
    source_text_template="; источник: {sources}",
    rows_label="строк",
    columns_label="столбцов",
    column_overflow_template="... +{hidden_columns} колонок",
    max_tables=12,
    max_columns=18,
)


def build_chat_data_context(df: pd.DataFrame | None, session_source: dict) -> str:
    """Compact data-context suffix for the chat LLM so it knows what sources are loaded."""
    parts: list[str] = []
    raw_table_descriptors = session_source.get("csv_table_descriptors")
    table_descriptors = coerce_runtime_table_descriptors(
        raw_table_descriptors if isinstance(raw_table_descriptors, list) else None
    )
    table_names = session_source.get("csv_table_names") or []
    has_csv_context = bool(
        session_source.get("csv_loaded")
        or table_descriptors
        or table_names
    )
    if has_csv_context:
        if table_descriptors:
            parts.append(
                "\n".join(
                    format_runtime_table_descriptors(
                        table_descriptors,
                        CHAT_TABLE_PROMPT_OPTIONS,
                    )
                )
            )
        else:
            tables_str = ", ".join(table_names) if table_names else "неизвестно"
            parts.append(f"Загружен CSV/XLSX в DuckDB. Таблицы: {tables_str}.")
        profile_block = build_dataset_profile_block(
            None,
            dataset_name=str(session_source.get("source_label") or "").strip(),
            session_source=session_source,
        )
        if profile_block:
            parts.append(profile_block)
    elif df is not None:
        parts.append(f"Загружен датасет: {df.shape[0]} строк, {df.shape[1]} столбцов.")
        profile_block = build_dataset_profile_block(
            df,
            dataset_name=str(session_source.get("source_label") or "").strip(),
            session_source=session_source,
        )
        if profile_block:
            parts.append(profile_block)
    source_label = str(session_source.get("source_label") or "").strip()
    source_type = str(session_source.get("source_type") or "").strip().lower()
    if source_type == "db_connection" and source_label:
        parts.append(f"Подключена база данных: {source_label}.")
    rag_block = build_rag_session_prompt_block(session_source)
    if rag_block:
        parts.append(rag_block)
    planfact_block = build_planfact_session_prompt_block(session_source)
    if planfact_block:
        parts.append(planfact_block)
        parts.append(build_planfact_period_prompt_block())
    if not parts:
        return ""
    return "[КОНТЕКСТ ДАННЫХ]\n" + "\n".join(parts)


def build_runtime_metadata(trace_context: dict[str, Any] | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if not trace_context:
        return metadata

    session_id = trace_context.get("session_id")
    if isinstance(session_id, str) and session_id:
        metadata["session_id"] = session_id
        metadata["thread_id"] = session_id
        metadata["conversation_id"] = session_id

    user_id = trace_context.get("user_id")
    if user_id is not None:
        metadata["user_id"] = str(user_id)

    username = trace_context.get("username")
    if isinstance(username, str) and username:
        metadata["username"] = username

    request_kind = trace_context.get("request_kind")
    if isinstance(request_kind, str) and request_kind:
        metadata["request_kind"] = request_kind

    for key in ("db_connection_id", "connection_id"):
        value = trace_context.get(key)
        if isinstance(value, str) and value.strip():
            metadata["db_connection_id"] = value.strip()
            metadata["data_source"] = "db_connection"
            break

    csv_session_id = trace_context.get("csv_session_id")
    if isinstance(csv_session_id, str) and csv_session_id.strip():
        metadata["csv_session_id"] = csv_session_id.strip()

    if bool(trace_context.get("csv_duckdb_loaded")) and "data_source" not in metadata:
        metadata["data_source"] = "csv_duckdb"

    return metadata


def extract_db_connection_id(
    session_source: dict[str, Any] | None,
    trace_context: dict[str, Any] | None,
) -> str | None:
    if isinstance(session_source, dict):
        source_type = str(session_source.get("source_type", "")).strip().lower()
        source_ref_id = session_source.get("source_ref_id")
        if source_type == "db_connection" and isinstance(source_ref_id, str) and source_ref_id.strip():
            return source_ref_id.strip()

    if not trace_context:
        return None

    for key in ("db_connection_id", "connection_id"):
        value = trace_context.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("db_connection", "db_source", "source", "data_source"):
        nested = trace_context.get(key)
        if not isinstance(nested, dict):
            continue
        for nested_key in ("db_connection_id", "connection_id"):
            value = nested.get(nested_key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def resolve_tool_db_runtime_config(
    *,
    db_runtime_service: Any | None,
    session_source: dict[str, Any] | None,
    trace_context: dict[str, Any] | None,
) -> Any | None:
    connection_id = extract_db_connection_id(session_source, trace_context)
    if not connection_id:
        return None
    if db_runtime_service is None:
        raise RuntimeError("DB runtime service is not configured.")

    user_id_raw = (trace_context or {}).get("user_id")
    try:
        user_id = int(user_id_raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "trace_context.user_id is required for DB tool runtime."
        ) from exc

    return db_runtime_service.get_runtime_config(
        user_id=user_id,
        connection_id=connection_id,
    )


def resolve_csv_runtime_state(
    session_source: dict[str, Any] | None,
    trace_context: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    if isinstance(session_source, dict):
        direct_loaded = bool(session_source.get("csv_loaded"))
        direct_sid = session_source.get("csv_session_id")
        if direct_loaded and isinstance(direct_sid, str) and direct_sid.strip():
            return True, direct_sid.strip()

        source_type = str(session_source.get("source_type", "")).strip().lower()
        if is_duckdb_source_type(source_type) and isinstance(direct_sid, str) and direct_sid.strip():
            return True, direct_sid.strip()

    if trace_context and bool(trace_context.get("csv_duckdb_loaded")):
        sid = trace_context.get("csv_session_id") or trace_context.get("session_id")
        if isinstance(sid, str) and sid.strip():
            return True, sid.strip()

    return False, None


def csv_table_descriptors_from_manifest(
    *,
    storage_dir: str | Path,
    csv_session_id: str | None,
    session_id: str | None,
    session_source: dict[str, Any] | None,
    manifest_store: ManifestStore | None = None,
) -> list[dict[str, Any]]:
    direct = (session_source or {}).get("csv_table_descriptors")
    if isinstance(direct, list):
        return [item for item in direct if isinstance(item, dict)]

    manifest_session_id = str(csv_session_id or session_id or "").strip()
    if not manifest_session_id:
        return []

    try:
        manifest = (manifest_store or ManifestStore(storage_dir)).load(manifest_session_id)
    except Exception:
        return []

    wanted = {
        str(item).strip()
        for item in (session_source or {}).get("csv_table_names", [])
        if str(item).strip()
    }
    descriptors: list[dict[str, Any]] = []
    for source in manifest.sources:
        if not is_duckdb_source_type(source.source_type):
            continue
        for table_name in source.csv_table_names:
            clean_table_name = str(table_name or "").strip()
            if not clean_table_name or (wanted and clean_table_name not in wanted):
                continue
            descriptors.append(
                {
                    "table_name": clean_table_name,
                    "qualified_name": clean_table_name,
                    "columns": list(source.schema_hint.keys()),
                    "file_name": source.file_name,
                    "display_name": source.display_name,
                    "source_alias": source.alias,
                    "schema_hint": dict(source.schema_hint or {}),
                    "preprocessing_summary": dict(source.preprocessing_summary or {}),
                    "row_count": source.row_count,
                    "column_count": source.column_count,
                }
            )
    return descriptors


def normalize_session_source_for_sql_mode(
    session_source: dict[str, Any] | None,
    trace_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(session_source, dict):
        session_source = {}

    normalized = dict(session_source)

    csv_loaded, csv_session_id = resolve_csv_runtime_state(
        session_source,
        trace_context,
    )
    if csv_loaded and csv_session_id:
        normalized["source_type"] = "db_connection"
        normalized["source_ref_id"] = csv_session_id
        normalized["source_label"] = str(
            normalized.get("source_label") or f"CSV DuckDB session {csv_session_id}"
        )
        normalized["source_mode"] = str(normalized.get("source_mode") or "read_only")
        normalized["csv_loaded"] = True
        normalized["csv_session_id"] = csv_session_id

    return normalized
