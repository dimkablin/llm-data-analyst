from __future__ import annotations

import ast
import errno
import hashlib
import json
import logging
import uuid
from typing import Any

import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict, Field

from backend.agent.callbacks import LLMTextCollector, ToolCollector
from backend.agent.constants import LLM_UNAVAILABLE_USER_TEXT
from backend.agent.context_window import (
    build_context_usage_snapshot,
    estimate_message_tokens,
    reserved_response_tokens_for_settings,
    trim_context_messages,
)
from backend.agent.models import AgentOutcome, AgentResponse, ErrorCategory
from backend.agent.runtime_llm import build_runtime_llm
from backend.agent.services.events import emit_context_usage_event
from backend.agent.services.runtime_context import build_runtime_metadata as _build_runtime_metadata
from backend.agent.working_memory import AnalysisWorkingMemory, ArtifactHandle
from backend.artifacts.execution import is_tabular_artifact_type
from backend.core.config import Settings
from backend.core.redaction import compact_error_text
from backend.mcp.models import MCPToolError
from backend.observability.phoenix import record_llm_usage_on_active_span
from backend.tools.impl.base_tool import BaseExecTool

logger = logging.getLogger(__name__)
_CONTEXT_SAFETY_MARGIN_TOKENS = 512
_PARTIAL_RESULT_SUMMARY_INSTRUCTION = """\
[ROLE: PARTIAL_RESULT_SUMMARY]
Execution stopped before a normal final answer because the tool loop stopped
making progress or exhausted its allowed steps. Produce one complete,
user-friendly final answer in the user's language using only the preceding
messages. Lead with all reliable findings already obtained, including supported
values, periods, and segments. Then briefly state what remains incomplete and
why. If nothing useful was obtained, say so plainly. Do not expose a raw
traceback, local paths, credentials, secrets, generated code, or internal
implementation details. Do not claim the task was completed. Return only the
user-facing answer. Do not call tools.
"""


def _bound_tool_schema_tokens(llm: Any, bound_llm: Any) -> int:
    bound_kwargs = getattr(bound_llm, "kwargs", None)
    tool_schemas = bound_kwargs.get("tools") if isinstance(bound_kwargs, dict) else None
    if not tool_schemas:
        return 0

    payload = json.dumps(
        tool_schemas,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    conservative_tokens = (len(payload.encode("utf-8")) + 2) // 3
    try:
        return max(conservative_tokens, int(llm.get_num_tokens(payload)))
    except Exception:
        # ponytail: conservative fallback until providers expose schema-token usage.
        return conservative_tokens


def _parse_textual_tool_call(content: str, tool_names: set[str]) -> dict[str, Any] | None:
    text = str(content or "").strip()
    try:
        node = ast.parse(text, mode="eval").body
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id not in tool_names
            or node.args
            or any(keyword.arg is None for keyword in node.keywords)
        ):
            return None
        arguments = {str(keyword.arg): ast.literal_eval(keyword.value) for keyword in node.keywords}
    except (SyntaxError, ValueError):
        return None
    return {
        "name": node.func.id,
        "args": arguments,
        "id": f"textual-{uuid.uuid4().hex}",
        "type": "tool_call",
    }


def _compact_tool_error_message(message: str, *, limit: int = 900) -> str:
    return compact_error_text(str(message or ""), limit=limit)


def _failure_key(
    tool_name: str,
    message: str,
    arguments: dict[str, Any],
    artifact: dict[str, Any] | None = None,
) -> str:
    artifact = artifact or {}
    error_type = str(artifact.get("error_type") or "ToolError").strip().casefold()
    missing_symbol = str(artifact.get("missing_symbol") or "").strip().casefold()
    error_text = str(artifact.get("error") or message)
    detail = "" if missing_symbol else " ".join(error_text.split()).casefold()
    return json.dumps(
        {
            "tool": tool_name,
            "args": arguments,
            "exception": error_type,
            "missing_symbol": missing_symbol or None,
            "detail": detail,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


class ToolFailureSummary(BaseModel):
    tool_name: str
    message: str

    @classmethod
    def from_tool_message(
        cls,
        *,
        output: ToolMessage,
        message: str,
    ) -> ToolFailureSummary | None:
        artifact = getattr(output, "artifact", None)
        message_status = str(getattr(output, "status", "") or "").strip().casefold()
        artifact_status = (
            str(artifact.get("status") or "").strip().casefold() if isinstance(artifact, dict) else ""
        )
        if message_status != "error" and artifact_status != "error":
            return None
        return cls(
            tool_name=str(getattr(output, "name", "") or "unknown").strip() or "unknown",
            message=_compact_tool_error_message(message),
        )

    def to_llm_unavailable_text(self) -> str:
        return (
            f"До того как LLM стала недоступна, инструмент `{self.tool_name}` вернул ошибку: {self.message}"
        )


def _is_llm_transport_failure(exc: BaseException) -> bool:
    """True for timeouts and TCP/DNS failures (not 4xx/5xx API errors)."""
    visited: set[int] = set()

    def walk(err: BaseException | None) -> bool:
        if err is None:
            return False
        eid = id(err)
        if eid in visited:
            return False
        visited.add(eid)

        if isinstance(err, TimeoutError | ConnectionError | BrokenPipeError):
            return True
        if isinstance(err, OSError) and err.errno is not None:
            _transport_errno = {
                errno.ECONNREFUSED,
                errno.ENETUNREACH,
                errno.EHOSTUNREACH,
                errno.ENETDOWN,
                errno.EPIPE,
                errno.ECONNRESET,
                errno.ETIMEDOUT,
            }
            if hasattr(errno, "WSAENETUNREACH"):
                _transport_errno.add(int(errno.WSAENETUNREACH))
            if hasattr(errno, "WSAETIMEDOUT"):
                _transport_errno.add(int(errno.WSAETIMEDOUT))
            if err.errno in _transport_errno:
                return True

        try:
            import httpx

            if isinstance(
                err,
                httpx.ConnectError
                | httpx.ConnectTimeout
                | httpx.ReadTimeout
                | httpx.WriteTimeout
                | httpx.PoolTimeout,
            ):
                return True
        except ImportError:
            pass

        try:
            from openai import APIConnectionError, APITimeoutError

            if isinstance(err, APIConnectionError | APITimeoutError):
                return True
        except ImportError:
            pass

        if walk(err.__cause__):
            return True
        if err.__context__ is not err.__cause__ and walk(err.__context__):
            return True
        return False

    return walk(exc)


def _log_llm_invoke_failure(where: str, exc: BaseException, settings: Settings) -> None:
    if _is_llm_transport_failure(exc):
        logger.warning(
            "%s: LLM endpoint unreachable or timed out (%s). base_url=%s model=%s",
            where,
            exc,
            settings.llm_base_url,
            settings.llm_model,
        )
    else:
        logger.exception("%s failed", where)


def _build_tool_message_text(
    result: object,
) -> tuple[str, ArtifactHandle | list[ArtifactHandle] | None]:
    def _short(obj: object, limit: int = 1600) -> str:
        try:
            text = json.dumps(obj, ensure_ascii=False, default=str, indent=2)
        except Exception:
            text = str(obj)
        return text[:limit]

    def _preview_rows(table_obj: object, max_rows: int = 15) -> list[dict]:
        try:
            if isinstance(table_obj, pd.Series):
                table_obj = table_obj.to_frame()

            if isinstance(table_obj, pd.DataFrame):
                return table_obj.head(max_rows).to_dict(orient="records")
        except Exception:
            pass

        if isinstance(table_obj, list):
            return table_obj[:max_rows]

        if isinstance(table_obj, dict):
            # sometimes already row-like or serialized table payload
            if "rows" in table_obj and isinstance(table_obj["rows"], list):
                return table_obj["rows"][:max_rows]
            return [table_obj]

        return []

    def _low_cardinality_values(table_obj: object) -> dict[str, list[str]]:
        try:
            if isinstance(table_obj, pd.Series):
                table_obj = table_obj.to_frame()
            if not isinstance(table_obj, pd.DataFrame):
                return {}

            values_by_column: dict[str, list[str]] = {}
            for column in table_obj.columns:
                series = table_obj[column]
                if not (
                    pd.api.types.is_string_dtype(series.dtype)
                    or pd.api.types.is_bool_dtype(series.dtype)
                    or isinstance(series.dtype, pd.CategoricalDtype)
                ):
                    continue
                values = series.dropna().astype(str).drop_duplicates()
                if 0 < len(values) <= 20:
                    values_by_column[str(column)] = values.tolist()
                if len(values_by_column) >= 8:
                    break
            return values_by_column
        except Exception:
            return {}

    def _column_value_profile(table_obj: object) -> dict[str, object]:
        try:
            if isinstance(table_obj, pd.Series):
                table_obj = table_obj.to_frame()
            if not isinstance(table_obj, pd.DataFrame) or table_obj.empty:
                return {}

            all_null: list[str] = []
            constant: dict[str, str] = {}
            for column in table_obj.columns[:80]:
                non_null = table_obj[column].dropna()
                if non_null.empty:
                    if len(all_null) < 8:
                        all_null.append(str(column))
                    continue
                try:
                    is_constant = non_null.nunique(dropna=True) == 1
                except Exception:
                    continue
                if is_constant and len(constant) < 8:
                    constant[str(column)] = str(non_null.iloc[0])[:120]

            profile: dict[str, object] = {}
            if all_null:
                profile["all_null_columns"] = all_null
            if constant:
                profile["constant_columns"] = constant
            return profile
        except Exception:
            return {}

    def _table_schema(obj: object) -> dict[str, str]:
        try:
            if isinstance(obj, pd.Series):
                obj = obj.to_frame()

            if isinstance(obj, pd.DataFrame):
                return {str(col): str(dtype) for col, dtype in obj.dtypes.items()}
        except Exception:
            pass
        return {}

    def _row_count(obj: object) -> int | None:
        try:
            if isinstance(obj, pd.Series):
                return len(obj)

            if isinstance(obj, pd.DataFrame):
                return len(obj)

            if isinstance(obj, list):
                return len(obj)

            if isinstance(obj, dict) and isinstance(obj.get("rows"), list):
                return len(obj["rows"])
        except Exception:
            pass
        return None

    def _normalize_artifact_payload(artifact: dict[str, Any]) -> tuple[str | None, dict | None]:
        """
        Supports both artifact formats:

        1) SQLTool / direct envelope:
           {"artifact_type": "table", "items": {...}}

        2) BaseExecTool envelope:
           {"text": "...", "code": "...", "table": {...}, "plot": {...}}
        """
        artifact_type = artifact.get("artifact_type")
        items = artifact.get("items")

        if isinstance(artifact_type, str) and isinstance(items, dict):
            return artifact_type, items

        if isinstance(artifact.get("table"), dict):
            return "table", artifact["table"]

        if isinstance(artifact.get("value"), dict):
            return "value", artifact["value"]

        if isinstance(artifact.get("json"), dict):
            return "json", artifact["json"]

        if isinstance(artifact.get("plot"), dict):
            return "plot", artifact["plot"]

        return None, None

    def _tool_meta_context(meta: dict[str, Any]) -> list[dict[str, Any]]:
        context_fields = ("summary", "warnings", "metrics", "params", "info")
        contexts: list[dict[str, Any]] = []
        for name, payload in meta.items():
            if not isinstance(payload, dict):
                continue

            context = {
                field: payload.get(field)
                for field in context_fields
                if payload.get(field) not in (None, "", [], {})
            }
            if context:
                contexts.append({"source": str(name), **context})
        return contexts

    content_text = ""
    artifact = None

    if hasattr(result, "content") and hasattr(result, "artifact"):
        content_text = str(getattr(result, "content", "") or "")
        artifact = getattr(result, "artifact", None)
    elif isinstance(result, tuple):
        content_text = str(result[0] or "")
        artifact = result[1] if len(result) > 1 else None
    else:
        content_text = str(result)

    parts: list[str] = []
    if content_text.strip():
        parts.append(content_text.strip())

    truncated_result = False
    bounded_result = False
    if isinstance(artifact, dict):
        if artifact.get("status") == "error":
            error_context: dict[str, object] = {}
            for field in ("error", "error_type", "missing_symbol", "source", "tool", "query"):
                value = artifact.get(field)
                if value not in (None, ""):
                    error_context[field] = value
            if "error" not in error_context and artifact:
                raw_error = _compact_tool_error_message(content_text or str(artifact.get("error") or ""))
                if raw_error:
                    error_context["error"] = raw_error
            if artifact.get("meta") is not None:
                error_context["meta"] = artifact.get("meta")

            if error_context:
                parts.append("TOOL_ERROR_CONTEXT_FOR_LLM:\n" + _short(error_context, limit=2500))

            return "\n\n".join(p for p in parts if p).strip(), None

        meta = artifact.get("meta")
        if isinstance(meta, dict):
            truncated_result = any(
                isinstance(payload, dict) and payload.get("truncated") is True for payload in meta.values()
            )
            bounded_result = any(
                isinstance(payload, dict) and payload.get("has_more_rows") is True
                for payload in meta.values()
            )
            tool_context = _tool_meta_context(meta)
            if tool_context:
                parts.append("TOOL_RESULT_CONTEXT_FOR_LLM:\n" + _short(tool_context, limit=3000))

        artifact_type, items = _normalize_artifact_payload(artifact)

        if artifact_type == "table" and isinstance(items, dict):
            previews = []
            has_empty_table = False

            PREVIEW_ROWS = 12
            PREVIEW_LIMIT = 8000

            for name, payload in items.items():
                schema = _table_schema(payload)
                total_rows = _row_count(payload)
                rows = _preview_rows(payload, max_rows=PREVIEW_ROWS)
                if total_rows == 0:
                    has_empty_table = True

                header = str(name)
                if total_rows is not None:
                    header += f" — {total_rows} rows × {len(schema)} cols"

                preview: dict[str, object] = {
                    "table": header,
                    "schema": schema,
                }
                categorical_values = _low_cardinality_values(payload)
                if categorical_values:
                    preview["low_cardinality_values_in_result"] = categorical_values
                value_profile = _column_value_profile(payload)
                if value_profile:
                    preview["column_value_profile"] = value_profile
                    preview["profile_guidance"] = (
                        "When the task needs multiple groups, an all-null or constant column "
                        "cannot satisfy the requested grouping dimension; do not relabel or "
                        "repeat it as that dimension. Inspect another categorical source "
                        "column, or reshape matching peer numeric columns from wide form into "
                        "dimension/value rows."
                    )
                preview[f"sample_{min(PREVIEW_ROWS, len(rows))}_of_{total_rows or '?'}_rows"] = rows
                previews.append(preview)

            if previews:
                parts.append("TABLE_RESULT:\n" + _short(previews, limit=PREVIEW_LIMIT))
            if has_empty_table:
                parts.append(
                    "EMPTY_RESULT: at least one table artifact has 0 rows and provides "
                    "no analytical evidence. Do not repeat an equivalent query. Inspect "
                    "exact values and types of filtered columns with SELECT DISTINCT, or "
                    "change the source, filters, or grain before rerunning. If the requested "
                    "period or measure is absent in this source, continue with the next planned "
                    "source or replan once using this observation."
                )

            if truncated_result:
                parts.append(
                    "TRUNCATED_RESULT: this table is only a capped preview, not the "
                    "complete analysis dataset. Do not analyze this preview or raise "
                    "LIMIT. If the query is not yet at final requested grain, aggregate "
                    "to the final requested grain in SQL. Changing only "
                    "the date range, LIMIT, ORDER BY, or artifact_name does not change "
                    "grain. If the requested period is coarser than the selected time "
                    "column, use an explicit time bucket in SELECT and GROUP BY. If the "
                    "query is already at final grain, fetch complete non-overlapping partitions "
                    "with exhaustive predicates, store them under distinct names, and "
                    "concatenate only complete partitions once before calculation or "
                    "visualization. Do not repeat an equivalent query."
                )

            if bounded_result:
                parts.append(
                    "BOUNDED_RESULT: the explicit LIMIT omitted additional rows. Use "
                    "this artifact only if the exact top-N is the intended final output "
                    "after complete aggregation. Otherwise remove LIMIT and aggregate "
                    "to the final requested grain; do not increase LIMIT incrementally."
                )

        elif artifact_type == "value" and isinstance(items, dict):
            parts.append("VALUE_RESULT:\n" + _short(items))

        elif artifact_type == "json" and isinstance(items, dict):
            parts.append("JSON_RESULT:\n" + _short(items))

        elif artifact_type == "plot" and isinstance(items, dict):
            plot_names = list(items.keys())[:5]
            parts.append("PLOT_RESULT:\n" + _short({"plot_names": plot_names}))

        else:
            parts.append("ARTIFACT_RESULT:\n" + _short(artifact))

    text = "\n\n".join(p for p in parts if p).strip()

    # Build ArtifactHandle objects from every typed section in a bundled result.
    handles: list[ArtifactHandle] = []
    if isinstance(artifact, dict):
        primary_type, primary_items = _normalize_artifact_payload(artifact)
        payloads = (
            [(primary_type, primary_items)]
            if primary_type in ("table", "value", "plot", "json") and isinstance(primary_items, dict)
            else []
        )
        payloads.extend(
            (kind, artifact[kind])
            for kind in ("table", "value", "plot", "json")
            if kind != primary_type and isinstance(artifact.get(kind), dict)
        )

        for artifact_type, items in payloads:
            if artifact_type == "table" and len(items) > 1:
                logger.debug(
                    "_build_tool_message_text: multi-table result (%d tables);"
                    " handle created for first only: %s",
                    len(items),
                    list(items.keys()),
                )

            artifact_name = next(iter(items), "")
            payload = items.get(artifact_name)

            handle_schema: dict[str, str] | None = None
            row_count: int | None = None
            summary: str | None = None

            if artifact_type == "table" and payload is not None:
                try:
                    if isinstance(payload, pd.Series):
                        payload = payload.to_frame()

                    if isinstance(payload, pd.DataFrame):
                        handle_schema = {str(col): str(dtype) for col, dtype in payload.dtypes.items()}
                        row_count = len(payload)
                        summary = f"{artifact_name}, {row_count}×{len(handle_schema)}"
                        if truncated_result:
                            summary += "; TRUNCATED preview"
                        elif bounded_result:
                            summary += "; BOUNDED result"
                except Exception:
                    pass

            elif artifact_type == "value":
                summary = str(payload)[:80] if payload is not None else None

            elif artifact_type == "plot":
                summary = str(artifact_name)

            if artifact_name:
                handles.append(
                    ArtifactHandle(
                        id=str(uuid.uuid4()),
                        name=str(artifact_name),
                        type=str(artifact_type),
                        tool_name="",  # filled in by caller
                        step_index=0,  # filled in by caller
                        schema=handle_schema,
                        row_count=row_count,
                        summary=summary,
                    )
                )

    if len(handles) > 1:
        parts.append(
            "AVAILABLE_ARTIFACT_HANDLES:\n"
            + _short([{"name": handle.name, "type": handle.type} for handle in handles])
        )
        text = "\n\n".join(p for p in parts if p).strip()
    return text, handles[0] if len(handles) == 1 else handles or None


# Observation masking — conservative policy (feature-flagged)
_MASK_KEEP_LAST_N = 3  # keep last N tool results at full content
_MASK_MIN_STEPS = 4  # do not mask if step_index < 4
_MASK_MIN_TOOLS = 3  # do not mask if tool_call_count < 3
_MASK_KEEP_TABLE_ROWS = 5  # compact evidence tables must remain visible for final synthesis


def _apply_observation_masking(
    messages: list,
    tc_id_to_handle: dict,
    tc_id_to_step: dict,
    current_step: int,
    masked_tc_ids: set,
) -> None:
    """Apply observation masking in-place on the messages list.

    Replaces old ToolMessage content with compact masked_ref strings.
    Modifies messages and masked_tc_ids in-place.
    """
    handles_by_tool_call = {
        tool_call_id: raw_handles
        if isinstance(raw_handles, list)
        else [raw_handles]
        if raw_handles is not None
        else []
        for tool_call_id, raw_handles in tc_id_to_handle.items()
    }
    latest_table_step = max(
        (
            tc_id_to_step[tool_call_id]
            for tool_call_id, handles in handles_by_tool_call.items()
            if tool_call_id in tc_id_to_step
            and any(handle.type == "table" for handle in handles)
            and current_step - tc_id_to_step[tool_call_id] >= _MASK_KEEP_LAST_N
        ),
        default=None,
    )
    for _mi, _msg in enumerate(messages):
        if not isinstance(_msg, ToolMessage):
            continue
        if getattr(_msg, "name", None) in {
            "planner_tool",
            "get_tool_instructions",
        }:
            continue
        _msg_id = getattr(_msg, "tool_call_id", "")
        if _msg_id in masked_tc_ids:
            continue  # already masked
        _handles = handles_by_tool_call.get(_msg_id, [])
        if not _handles:
            continue
        _step = tc_id_to_step.get(_msg_id)
        if _step is None:
            continue
        steps_ago = current_step - _step
        if steps_ago < _MASK_KEEP_LAST_N:
            continue
        if any(
            handle.type == "table"
            and handle.row_count is not None
            and handle.row_count <= _MASK_KEEP_TABLE_ROWS
            for handle in _handles
        ):
            continue
        if any(handle.type == "error" for handle in _handles):
            continue
        if _step == latest_table_step and any(handle.type == "table" for handle in _handles):
            continue
        masked_content = "\n".join(handle.masked_ref for handle in _handles)
        messages[_mi] = ToolMessage(content=masked_content, tool_call_id=_msg_id)
        masked_tc_ids.add(_msg_id)


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content or "")


def reset_text_collectors(callbacks: list[Any]) -> None:
    for callback in callbacks:
        if isinstance(callback, LLMTextCollector):
            callback.messages.clear()


def collect_tool_stats(callbacks: list[Any]) -> tuple[list[Any], int, list[str]]:
    tool_collector = next(
        (callback for callback in callbacks if isinstance(callback, ToolCollector)),
        None,
    )
    if tool_collector is None:
        return [], 0, []

    artifacts = list(tool_collector.artifacts)
    if not any(is_tabular_artifact_type(getattr(artifact, "artifact_type", "")) for artifact in artifacts):
        artifacts.extend(
            stored
            for stored in tool_collector.execution_store.all()
            if is_tabular_artifact_type(getattr(stored, "artifact_type", ""))
        )

    tool_names: list[str] = []
    seen: set[str] = set()
    for item in tool_collector.tool_names:
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        tool_names.append(normalized)
    return artifacts, int(tool_collector.tool_calls), tool_names


def ingest_tool_invoke_result(
    tool_collector: ToolCollector | None,
    tool_name: str,
    result: object,
    tool_call_id: str,
) -> None:
    """Ensure ToolCollector sees artifacts when LangChain callbacks miss them."""
    if tool_collector is None:
        return
    before = len(tool_collector.artifacts)
    if isinstance(result, ToolMessage):
        tool_collector.absorb_tool_message(tool_name, result)
    elif isinstance(result, tuple) and len(result) > 1 and isinstance(result[1], dict):
        tool_collector.absorb_tool_message(
            tool_name,
            ToolMessage(
                content=str(result[0] or ""),
                artifact=dict(result[1]),
                tool_call_id=tool_call_id,
            ),
        )
    if len(tool_collector.artifacts) <= before:
        return


def artifact_recovery_text(artifacts: list[Any]) -> str:
    if not artifacts:
        return ""
    from backend.agent.constants import RECOVERY_TEXT_PREFIX

    summary = artifact_summary_text(artifacts)
    if not summary:
        return ""
    return summary.replace("Артефакты построены", RECOVERY_TEXT_PREFIX, 1)


def llm_unavailable_final_text(
    artifacts: list[Any],
    tool_failure: ToolFailureSummary | None,
) -> str:
    recovery_text = artifact_recovery_text(artifacts)
    parts: list[str] = []
    if recovery_text:
        parts.append(recovery_text)
        parts.append("LLM стала недоступна до завершения финального ответа.")
    else:
        parts.append(LLM_UNAVAILABLE_USER_TEXT)

    if tool_failure is not None:
        parts.append(tool_failure.to_llm_unavailable_text())

    return "\n\n".join(part for part in parts if part).strip()


def artifact_summary_text(artifacts: list[Any]) -> str:
    """Neutral summary when the loop ended with artifacts but no final text."""
    if not artifacts:
        return ""

    counts: dict[str, int] = {}
    labels: list[str] = []
    for artifact in artifacts[:8]:
        artifact_type = str(getattr(artifact, "artifact_type", "artifact")).strip() or "artifact"
        counts[artifact_type] = counts.get(artifact_type, 0) + 1
        label = str(getattr(artifact, "text", "")).strip()
        if label:
            labels.append(label)

    typed_counts = ", ".join(
        f"{name}: {value}" for name, value in sorted(counts.items(), key=lambda item: item[0])
    )
    if labels:
        labels_preview = ", ".join(labels[:4])
        if len(labels) > 4:
            labels_preview += ", ..."
        return f"Артефакты построены ({typed_counts}). Доступные артефакты: {labels_preview}."

    return f"Артефакты построены ({typed_counts})."


class ToolLoopRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    settings: Settings
    include_reasoning: bool
    tools: list[Any] = Field(default_factory=list)
    callbacks: list[Any] = Field(default_factory=list)
    max_iterations: int
    trace_context: dict[str, Any] | None = None
    working_memory: AnalysisWorkingMemory | None = None
    messages: list[BaseMessage]
    cancel_event: Any | None = None


def _is_cancelled(cancel_event: Any | None) -> bool:
    is_set = getattr(cancel_event, "is_set", None)
    return bool(callable(is_set) and is_set())


def _cancelled_agent_response(
    *,
    callbacks: list[Any],
    all_tool_names: list[str],
    total_tool_calls: int,
    reasoning_steps: list[str],
) -> AgentResponse:
    artifacts, collector_tool_calls, collector_tool_names = collect_tool_stats(callbacks)
    return AgentResponse(
        final_text=artifact_summary_text(artifacts),
        reasoning="Cancelled by client.",
        reasoning_steps=reasoning_steps,
        artifacts=artifacts,
        route="analysis",
        tool_calls=max(total_tool_calls, collector_tool_calls),
        tool_names=_merge_tool_names(all_tool_names, collector_tool_names),
        outcome=AgentOutcome.cancelled(),
    )


def _max_tools_per_cycle_text(limit: int) -> str:
    return (
        f"Tool call skipped: MAX_TOOLS_PER_CYCLE={limit}. "
        "Request this tool again in the next cycle if it is still needed."
    )


def _as_tool_messages(output: object) -> list[ToolMessage]:
    if isinstance(output, list):
        return [item for item in output if isinstance(item, ToolMessage)]
    if isinstance(output, dict):
        messages = output.get("messages")
        if isinstance(messages, list):
            return [item for item in messages if isinstance(item, ToolMessage)]
    if isinstance(output, ToolMessage):
        return [output]
    return []


def _build_tool_node(
    tools: list[Any],
    tool_collector: ToolCollector | None,
) -> ToolNode:
    def _wrap_tool_call(request: ToolCallRequest, execute: Any) -> ToolMessage:
        tool_name = str(request.tool_call.get("name", "")).strip()
        tool_call_id = str(request.tool_call.get("id", "")).strip()
        if request.tool is None:
            return ToolMessage(
                content=f"Unknown tool: {tool_name}",
                name=tool_name,
                tool_call_id=tool_call_id,
                status="error",
            )

        artifact_count_before = len(tool_collector.artifacts) if tool_collector is not None else 0
        try:
            result = execute(request)
        except Exception as exc:
            artifact = None
            if isinstance(exc, MCPToolError):
                artifact = {
                    "status": "error",
                    "error": exc.details.model_dump(mode="json"),
                }
            return ToolMessage(
                content=f"Tool error: {exc}",
                name=tool_name,
                tool_call_id=tool_call_id,
                status="error",
                artifact=artifact,
            )

        if (
            tool_collector is not None
            and isinstance(result, ToolMessage)
            and len(tool_collector.artifacts) == artifact_count_before
        ):
            ingest_tool_invoke_result(tool_collector, tool_name, result, tool_call_id)
        return result

    return ToolNode(
        tools,
        handle_tool_errors=False,
        wrap_tool_call=_wrap_tool_call,
    )


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or "").strip()


def _tools_by_name(tools: list[Any]) -> dict[str, Any]:
    return {name: tool for tool in tools if (name := _tool_name(tool))}


def _can_parallelize_tool(tool: Any) -> bool:
    if isinstance(tool, BaseExecTool):
        return False
    return bool(getattr(tool.__class__, "parallel_safe", False))


def _tool_batch_concurrency(
    tool_calls: list[dict[str, Any]],
    tools_by_name: dict[str, Any],
    limit: int,
) -> int:
    if len(tool_calls) < 2:
        return 1

    tool_names = [str(call.get("name", "")).strip() for call in tool_calls]
    tools = [tools_by_name.get(name) for name in tool_names]
    if any(tool is None for tool in tools):
        return 1

    plan_call_count = tool_names.count("update_plan")
    if plan_call_count > 1:
        return 1
    if len(tool_calls) == 2 and plan_call_count == 1:
        return min(max(1, int(limit)), 2)

    if any(not _can_parallelize_tool(tool) for tool in tools):
        return 1
    return min(max(1, int(limit)), len(tool_calls))


def direct_tool_loop(request: ToolLoopRequest) -> AgentResponse:
    """Run the generic tool-calling loop for the agent graph."""
    settings = request.settings
    include_reasoning = request.include_reasoning
    tools = request.tools
    callbacks = request.callbacks
    max_iterations = request.max_iterations
    trace_context = request.trace_context
    working_memory = request.working_memory
    cancel_event = request.cancel_event
    reset_text_collectors(callbacks)
    llm = build_runtime_llm(
        settings,
        role="tool",
        include_reasoning=include_reasoning,
        timeout_sec=min(
            settings.agent_step_timeout_sec,
            settings.backend_query_timeout_sec,
        ),
    )
    bound_llm = llm.bind_tools(tools)
    forced_plan_llm = (
        llm.bind_tools(tools, tool_choice="update_plan")
        if settings.always_use_analysis_plan and any(_tool_name(tool) == "update_plan" for tool in tools)
        else None
    )

    messages = list(request.messages)
    count_message_tokens = getattr(llm, "get_num_tokens_from_messages", None)
    tool_schema_tokens = _bound_tool_schema_tokens(llm, bound_llm)
    max_context_tokens = settings.llm_num_ctx if settings.llm_num_ctx > 0 else None
    reserved_response_tokens = reserved_response_tokens_for_settings(
        settings,
        include_reasoning=include_reasoning,
    )
    max_message_tokens = (
        max(
            1,
            max_context_tokens
            - reserved_response_tokens
            - tool_schema_tokens
            - _CONTEXT_SAFETY_MARGIN_TOKENS,
        )
        if max_context_tokens is not None
        else 0
    )

    def count_input_tokens(seen_messages: list[BaseMessage]) -> int:
        return (
            estimate_message_tokens(
                seen_messages,
                count_message_tokens=count_message_tokens,
            )
            + tool_schema_tokens
        )

    all_tool_names: list[str] = []
    total_tool_calls = 0
    final_text = ""
    reasoning = None
    reasoning_steps: list[str] = []
    last_tool_failure: ToolFailureSummary | None = None
    failure_counts: dict[str, int] = {}
    failure_counts_by_tool: dict[str, int] = {}
    error_fingerprints: list[str] = []
    retry_count = 0
    terminal_status = "success"
    success_counts: dict[str, int] = {}
    recovery_summary_reason: str | None = None
    # Maps tool_call_id → ArtifactHandle (for masking)
    tool_call_id_to_handle: dict[str, ArtifactHandle] = {}
    # Maps tool_call_id → step_index when it was executed (for masking non-artifact tools)
    tool_call_id_to_step: dict[str, int] = {}
    # Tracks which tool_call_ids have already been masked (prevents double-wrap)
    masked_tool_call_ids: set[str] = set()

    tool_collector = next((cb for cb in callbacks if isinstance(cb, ToolCollector)), None)
    tool_node = _build_tool_node(tools, tool_collector)
    tool_lookup = _tools_by_name(tools)
    max_tools_per_cycle = max(2, int(settings.max_tools_per_cycle))

    runtime_config: dict[str, Any] = {"callbacks": callbacks}
    metadata = _build_runtime_metadata(trace_context)
    if metadata:
        runtime_config["metadata"] = metadata

    for _iteration in range(max(1, max_iterations)):
        if _is_cancelled(cancel_event):
            request.messages = messages
            return _cancelled_agent_response(
                callbacks=callbacks,
                all_tool_names=all_tool_names,
                total_tool_calls=total_tool_calls,
                reasoning_steps=reasoning_steps,
            )
        try:
            messages = trim_context_messages(
                messages,
                max_input_tokens=max_message_tokens,
                count_message_tokens=count_message_tokens,
            )
            invoke_messages = messages
            emit_context_usage_event(
                callbacks,
                build_context_usage_snapshot(
                    invoke_messages,
                    max_context_tokens=max_context_tokens,
                    reserved_response_tokens=reserved_response_tokens,
                    context_window_source="settings" if max_context_tokens else "unavailable",
                    count_message_tokens=count_input_tokens,
                ),
            )
            active_llm = forced_plan_llm if _iteration == 0 and forced_plan_llm is not None else bound_llm
            response = active_llm.invoke(invoke_messages, config=runtime_config)
            record_llm_usage_on_active_span(
                response,
                fallback_model=settings.llm_model,
                fallback_provider=settings.llm_provider,
            )
            if _is_cancelled(cancel_event):
                request.messages = messages
                return _cancelled_agent_response(
                    callbacks=callbacks,
                    all_tool_names=all_tool_names,
                    total_tool_calls=total_tool_calls,
                    reasoning_steps=reasoning_steps,
                )
        except Exception as exc:
            if _is_llm_transport_failure(exc):
                _log_llm_invoke_failure("direct_tool_loop LLM invoke", exc, settings)
                artifacts, collector_tool_calls, collector_tool_names = collect_tool_stats(callbacks)
                request.messages = messages
                return AgentResponse(
                    final_text=llm_unavailable_final_text(
                        artifacts,
                        last_tool_failure,
                    ),
                    reasoning=str(exc),
                    reasoning_steps=[],
                    artifacts=artifacts,
                    route="analysis",
                    tool_calls=max(total_tool_calls, collector_tool_calls),
                    tool_names=_merge_tool_names(all_tool_names, collector_tool_names),
                    llm_unreachable=True,
                    outcome=(
                        AgentOutcome.partial(ErrorCategory.MODEL)
                        if artifacts
                        else AgentOutcome.unavailable(ErrorCategory.MODEL)
                    ),
                )
            raise

        step_r = response.additional_kwargs.get("reasoning") or None
        if step_r:
            reasoning_steps.append(step_r)
            if reasoning is None:
                reasoning = step_r  # backward compat: первый шаг → reasoning поле

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            final_text = content_to_text(getattr(response, "content", ""))
            textual_tool_call = _parse_textual_tool_call(final_text, set(tool_lookup))
            if textual_tool_call:
                response.tool_calls = [textual_tool_call]
                tool_calls = response.tool_calls
                final_text = ""
            else:
                break

        messages.append(response)

        stop_after_repeated_observation = False
        executable_calls = list(tool_calls[:max_tools_per_cycle])
        executable_calls_by_id = {str(tc.get("id", "")).strip(): tc for tc in executable_calls}
        skipped_calls = list(tool_calls[max_tools_per_cycle:])

        for tc in executable_calls:
            tool_name = str(tc.get("name", "")).strip()
            total_tool_calls += 1
            if tool_name and tool_name not in all_tool_names:
                all_tool_names.append(tool_name)

        tool_outputs: list[ToolMessage] = []
        if executable_calls:
            if _is_cancelled(cancel_event):
                request.messages = messages
                return _cancelled_agent_response(
                    callbacks=callbacks,
                    all_tool_names=all_tool_names,
                    total_tool_calls=total_tool_calls,
                    reasoning_steps=reasoning_steps,
                )
            node_config = {
                **runtime_config,
                "max_concurrency": _tool_batch_concurrency(
                    executable_calls,
                    tool_lookup,
                    max_tools_per_cycle,
                ),
            }
            tool_outputs = _as_tool_messages(
                tool_node.invoke(executable_calls, config=node_config, runtime=Runtime())
            )

        for output in tool_outputs:
            tool_name = str(getattr(output, "name", "") or "").strip()
            tool_call_id = str(getattr(output, "tool_call_id", "") or "").strip()
            tool_message_text, built_handles = _build_tool_message_text(output)
            handles = (
                built_handles
                if isinstance(built_handles, list)
                else [built_handles]
                if built_handles is not None
                else []
            )
            primary_handle = handles[0] if handles else None

            if working_memory is not None:
                if handles:
                    for handle in handles:
                        handle.tool_name = tool_name
                        handle.step_index = working_memory.step_index
                    working_memory.artifact_handles.extend(handles)
                    working_memory.last_tool_result_summary = ", ".join(
                        handle.summary or handle.masked_ref for handle in handles
                    )
                    action_line = f"{tool_name} → {', '.join(handle.name for handle in handles)}"
                else:
                    working_memory.last_tool_result_summary = tool_message_text[:120]
                    action_line = f"{tool_name} → {tool_message_text[:60]}"
                working_memory.completed_actions.append(action_line)
                working_memory.step_index += 1
                working_memory.tool_call_count += 1
                tool_call_id_to_step[tool_call_id] = (
                    primary_handle.step_index
                    if primary_handle is not None
                    else (working_memory.step_index - 1)
                )
                if primary_handle is not None:
                    tool_call_id_to_handle[tool_call_id] = handles
            elif tool_call_id:
                tool_call_id_to_step[tool_call_id] = 0

            tool_failure = ToolFailureSummary.from_tool_message(
                output=output,
                message=tool_message_text,
            )
            last_tool_failure = tool_failure
            if tool_failure is not None and not tool_message_text.lower().startswith(
                ("pandas_tool failed", "plotly_tool failed")
            ):
                observation_text = tool_failure.message
            else:
                observation_text = tool_message_text
            output.content = observation_text
            messages.append(output)
            if tool_failure is not None:
                tool_call = executable_calls_by_id.get(tool_call_id) or {}
                artifact = getattr(output, "artifact", None)
                failure_key = _failure_key(
                    tool_name,
                    tool_failure.message,
                    tool_call.get("args") or {},
                    artifact if isinstance(artifact, dict) else None,
                )
                failure_counts[failure_key] = failure_counts.get(failure_key, 0) + 1
                fingerprint = hashlib.sha256(failure_key.encode("utf-8", errors="replace")).hexdigest()
                if fingerprint not in error_fingerprints:
                    error_fingerprints.append(fingerprint)
                failure_counts_by_tool[tool_name] = failure_counts_by_tool.get(tool_name, 0) + 1
                if failure_counts_by_tool[tool_name] > 1:
                    retry_count += 1
                if failure_counts[failure_key] >= 3:
                    final_text = (
                        "Не удалось завершить анализ: один и тот же вызов "
                        "инструмента трижды завершился одинаковой ошибкой."
                    )
                    reasoning = reasoning or final_text
                    terminal_status = "partial"
                    recovery_summary_reason = (
                        "The same tool call with identical arguments returned the same error three times."
                    )
                    stop_after_repeated_observation = True
                    break
            else:
                tool_call = executable_calls_by_id.get(tool_call_id)
                if tool_call is not None:
                    success_key = json.dumps(
                        {
                            "tool": tool_name,
                            "args": tool_call.get("args") or {},
                            "result": " ".join(tool_message_text.split()),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                    success_fingerprint = hashlib.sha256(
                        success_key.encode("utf-8", errors="replace")
                    ).hexdigest()
                    success_counts[success_fingerprint] = success_counts.get(success_fingerprint, 0) + 1
                    if success_counts[success_fingerprint] >= 3:
                        final_text = (
                            f"Не могу продолжить: tool `{tool_name}` трижды повторил "
                            "одинаковый успешный вызов без нового результата."
                        )
                        reasoning = reasoning or final_text
                        terminal_status = "failed"
                        recovery_summary_reason = (
                            "The same successful tool call returned no new result three times."
                        )
                        stop_after_repeated_observation = True
                        break
        for tc in skipped_calls:
            tool_name = str(tc.get("name", "")).strip()
            tool_call_id = str(tc.get("id", "")).strip()
            messages.append(
                ToolMessage(
                    content=_max_tools_per_cycle_text(max_tools_per_cycle),
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    status="error",
                )
            )

        if stop_after_repeated_observation:
            break

        # Observation masking pass
        if (
            settings.observation_mask_enabled
            and working_memory is not None
            and working_memory.step_index >= _MASK_MIN_STEPS
            and working_memory.tool_call_count >= _MASK_MIN_TOOLS
        ):
            _apply_observation_masking(
                messages,
                tool_call_id_to_handle,
                tool_call_id_to_step,
                current_step=working_memory.step_index,
                masked_tc_ids=masked_tool_call_ids,
            )
    else:
        # Max iterations reached — summarize evidence without treating narration as a final answer.
        terminal_status = "failed"
        recovery_summary_reason = (
            "The tool loop exhausted its allowed steps without producing a final answer."
        )
        final_text = "Анализ остановлен после исчерпания доступных шагов без итогового ответа."

    artifacts, collector_tool_calls, collector_tool_names = collect_tool_stats(callbacks)
    total_tool_calls = max(total_tool_calls, collector_tool_calls)
    all_tool_names = _merge_tool_names(all_tool_names, collector_tool_names)
    if recovery_summary_reason is not None:
        try:
            summary_response = llm.invoke(
                [
                    *messages,
                    HumanMessage(
                        content=(
                            f"{_PARTIAL_RESULT_SUMMARY_INSTRUCTION}\nStop reason: {recovery_summary_reason}"
                        )
                    ),
                ],
                config={**runtime_config, "callbacks": []},
            )
            record_llm_usage_on_active_span(
                summary_response,
                fallback_model=settings.llm_model,
                fallback_provider=settings.llm_provider,
            )
            summary_text = content_to_text(summary_response.content).strip()
            if summary_text:
                final_text = compact_error_text(
                    summary_text,
                    limit=max(1200, len(summary_text)),
                )
        except Exception as exc:
            logger.warning("Partial result summary failed: %s", compact_error_text(str(exc)))

    if not final_text and artifacts:
        final_text = artifact_summary_text(artifacts)
    if terminal_status in {"unavailable", "failed"} and artifacts:
        terminal_status = "partial"
    request.messages = messages
    if terminal_status == "success":
        outcome = AgentOutcome.success()
    elif terminal_status == "partial":
        outcome = AgentOutcome.partial(ErrorCategory.TOOL)
    elif terminal_status == "unavailable":
        outcome = AgentOutcome.unavailable(ErrorCategory.TOOL)
    else:
        outcome = AgentOutcome.failed(ErrorCategory.TOOL)
    return AgentResponse(
        final_text=final_text.strip(),
        reasoning=reasoning,
        reasoning_steps=reasoning_steps,
        artifacts=artifacts,
        route="analysis",
        tool_calls=total_tool_calls,
        tool_names=all_tool_names,
        outcome=outcome,
        error_fingerprints=error_fingerprints,
        retry_count=retry_count,
        tool_error_count=sum(failure_counts.values()),
    )


def _merge_tool_names(primary: list[str], secondary: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for name in [*primary, *secondary]:
        normalized = str(name).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return merged
