from __future__ import annotations

import errno
import json
import logging
import uuid
from typing import Any

import pandas as pd
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict, Field

from backend.agent.callbacks import LLMTextCollector, ToolCollector
from backend.agent.constants import LLM_UNAVAILABLE_USER_TEXT
from backend.agent.context_window import (
    build_context_usage_snapshot,
    reserved_response_tokens_for_settings,
    trim_context_messages,
)
from backend.agent.models import AgentResponse
from backend.agent.runtime_llm import build_runtime_llm
from backend.agent.services.events import emit_context_usage_event
from backend.agent.services.runtime_context import build_runtime_metadata as _build_runtime_metadata
from backend.agent.working_memory import AnalysisWorkingMemory, ArtifactHandle
from backend.artifacts.execution import is_tabular_artifact_type
from backend.core.config import Settings
from backend.core.redaction import compact_error_text
from backend.data_access.dataframe_utils import numeric_summary_rows
from backend.observability.phoenix import record_llm_usage_on_active_span
from backend.tools.impl.base_tool import BaseExecTool

logger = logging.getLogger(__name__)
_CONTEXT_SAFETY_MARGIN_TOKENS = 512


def _is_tool_error_observation(message: str) -> bool:
    text = str(message or "").strip()
    if not text:
        return False
    lowered = text.lower()
    return (
        lowered.startswith("tool error:")
        or lowered.startswith("unknown tool:")
        or lowered.startswith("pandas_tool failed")
        or lowered.startswith("plotly_tool failed")
        or lowered.startswith("❌")
        or "ошибка при" in lowered
    )


def _compact_tool_error_message(message: str, *, limit: int = 900) -> str:
    return compact_error_text(str(message or ""), limit=limit)


def _is_source_unavailable_observation(message: str) -> bool:
    text = str(message or "").lower()
    return any(
        marker in text
        for marker in (
            "network is unreachable",
            "connection refused",
            "connection reset",
            "connecterror",
            "timed out",
            "server closed the connection",
            "could not connect",
        )
    )


class ToolFailureSummary(BaseModel):
    tool_name: str
    message: str

    @classmethod
    def from_observation(
        cls,
        *,
        tool_name: str,
        message: str,
    ) -> ToolFailureSummary | None:
        if not _is_tool_error_observation(message):
            return None
        return cls(
            tool_name=str(tool_name or "unknown").strip() or "unknown",
            message=_compact_tool_error_message(message),
        )

    def to_llm_unavailable_text(self) -> str:
        return (
            f"До того как LLM стала недоступна, инструмент `{self.tool_name}` "
            f"вернул ошибку: {self.message}"
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
                httpx.ConnectError | httpx.ConnectTimeout | httpx.ReadTimeout | httpx.WriteTimeout | httpx.PoolTimeout,  # noqa: E501
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


def _build_tool_message_text(result: object) -> tuple[str, ArtifactHandle | None]:
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

    def _numeric_summary_rows(obj: object) -> list[dict[str, object]]:
        try:
            if isinstance(obj, pd.Series):
                obj = obj.to_frame()
            if isinstance(obj, pd.DataFrame):
                return numeric_summary_rows(obj)
        except Exception:
            pass
        return []

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

    if isinstance(artifact, dict):
        if artifact.get("status") == "error":
            return "\n\n".join(parts).strip(), None

        meta = artifact.get("meta")
        if isinstance(meta, dict):
            tool_context = _tool_meta_context(meta)
            if tool_context:
                parts.append("TOOL_RESULT_CONTEXT_FOR_LLM:\n" + _short(tool_context, limit=3000))

        artifact_type, items = _normalize_artifact_payload(artifact)

        if artifact_type == "table" and isinstance(items, dict):
            previews = []

            PREVIEW_ROWS = 30
            PREVIEW_LIMIT = 8000

            for name, payload in items.items():
                schema = _table_schema(payload)
                total_rows = _row_count(payload)
                rows = _preview_rows(payload, max_rows=PREVIEW_ROWS)
                summary_rows = _numeric_summary_rows(payload)

                header = str(name)
                if total_rows is not None:
                    header += f" — {total_rows} rows × {len(schema)} cols"

                preview = {
                    "table": header,
                    "schema": schema,
                    f"sample_{min(PREVIEW_ROWS, len(rows))}_of_{total_rows or '?'}_rows": rows,
                }
                if summary_rows:
                    preview["numeric_summary_rows_appended"] = summary_rows
                previews.append(preview)

            if previews:
                parts.append("TABLE_RESULT:\n" + _short(previews, limit=PREVIEW_LIMIT))

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

    # Build ArtifactHandle from artifact metadata
    handle: ArtifactHandle | None = None
    if isinstance(artifact, dict):
        artifact_type, items = _normalize_artifact_payload(artifact)

        if artifact_type in ("table", "value", "plot", "json") and isinstance(items, dict):
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
                        handle_schema = {
                            str(col): str(dtype)
                            for col, dtype in payload.dtypes.items()
                        }
                        row_count = len(payload)
                        summary = f"{artifact_name}, {row_count}×{len(handle_schema)}"
                except Exception:
                    pass

            elif artifact_type == "value":
                summary = str(payload)[:80] if payload is not None else None

            elif artifact_type == "plot":
                summary = str(artifact_name)

            if artifact_name:
                handle = ArtifactHandle(
                    id=str(uuid.uuid4()),
                    name=str(artifact_name),
                    type=str(artifact_type),
                    tool_name="",  # filled in by caller
                    step_index=0,  # filled in by caller
                    schema=handle_schema,
                    row_count=row_count,
                    summary=summary,
                )

    return text, handle


# Observation masking — conservative policy (feature-flagged)
_MASK_KEEP_LAST_N = 3     # keep last N tool results at full content
_MASK_MIN_STEPS = 4       # do not mask if step_index < 4
_MASK_MIN_TOOLS = 3       # do not mask if tool_call_count < 3


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
    for _mi, _msg in enumerate(messages):
        if not isinstance(_msg, ToolMessage):
            continue
        _msg_id = getattr(_msg, "tool_call_id", "")
        if _msg_id in masked_tc_ids:
            continue  # already masked
        _h = tc_id_to_handle.get(_msg_id)
        _step = tc_id_to_step.get(_msg_id)
        if _step is None:
            continue
        steps_ago = current_step - _step
        if steps_ago < _MASK_KEEP_LAST_N:
            continue
        if _h is not None and _h.type == "error":
            continue
        if _h is not None:
            masked_content = _h.masked_ref
        else:
            original = str(_msg.content)
            masked_content = (
                f"[step {_step}: {original[:80]}...]"
                if len(original) > 80
                else f"[step {_step}: {original}]"
            )
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
    if not any(
        is_tabular_artifact_type(getattr(artifact, "artifact_type", ""))
        for artifact in artifacts
    ):
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

        artifact_count_before = (
            len(tool_collector.artifacts) if tool_collector is not None else 0
        )
        try:
            result = execute(request)
        except Exception as exc:
            return ToolMessage(
                content=f"Tool error: {exc}",
                name=tool_name,
                tool_call_id=tool_call_id,
                status="error",
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

    for call in tool_calls:
        tool_name = str(call.get("name", "")).strip()
        tool = tools_by_name.get(tool_name)
        if tool is None or not _can_parallelize_tool(tool):
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

    messages = list(request.messages)
    count_message_tokens = getattr(llm, "get_num_tokens_from_messages", None)
    max_context_tokens = settings.llm_num_ctx if settings.llm_num_ctx > 0 else None
    reserved_response_tokens = reserved_response_tokens_for_settings(
        settings,
        include_reasoning=include_reasoning,
    )
    max_input_tokens = (
        max(1, max_context_tokens - reserved_response_tokens - _CONTEXT_SAFETY_MARGIN_TOKENS)
        if max_context_tokens is not None
        else 0
    )

    all_tool_names: list[str] = []
    total_tool_calls = 0
    final_text = ""
    reasoning = None
    reasoning_steps: list[str] = []
    last_tool_failure: ToolFailureSummary | None = None

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
                max_input_tokens=max_input_tokens,
                count_message_tokens=count_message_tokens,
            )
            emit_context_usage_event(
                callbacks,
                build_context_usage_snapshot(
                    messages,
                    max_context_tokens=max_context_tokens,
                    reserved_response_tokens=reserved_response_tokens,
                    context_window_source="settings" if max_context_tokens else "unavailable",
                    count_message_tokens=count_message_tokens,
                ),
            )
            response = bound_llm.invoke(messages, config=runtime_config)
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
            break

        messages.append(response)

        stop_after_repeated_error = False
        executable_calls = list(tool_calls[:max_tools_per_cycle])
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
            tool_message_text, _handle = _build_tool_message_text(output)

            if working_memory is not None:
                if _handle is not None:
                    _handle.tool_name = tool_name
                    _handle.step_index = working_memory.step_index
                    working_memory.artifact_handles.append(_handle)
                    working_memory.last_tool_result_summary = (
                        _handle.summary or _handle.masked_ref
                    )
                    action_line = f"{tool_name} → {_handle.name}"
                else:
                    working_memory.last_tool_result_summary = tool_message_text[:120]
                    action_line = f"{tool_name} → {tool_message_text[:60]}"
                working_memory.completed_actions.append(action_line)
                working_memory.step_index += 1
                working_memory.tool_call_count += 1
                tool_call_id_to_step[tool_call_id] = (
                    _handle.step_index if _handle is not None
                    else (working_memory.step_index - 1)
                )
                if _handle is not None:
                    tool_call_id_to_handle[tool_call_id] = _handle
            elif tool_call_id:
                tool_call_id_to_step[tool_call_id] = 0

            tool_failure = ToolFailureSummary.from_observation(
                tool_name=tool_name,
                message=tool_message_text,
            )
            last_tool_failure = tool_failure
            if (
                tool_failure is not None
                and not tool_message_text.lower().startswith(("pandas_tool failed", "plotly_tool failed"))
            ):
                observation_text = tool_failure.message
            else:
                observation_text = tool_message_text
            output.content = observation_text
            messages.append(output)
            if tool_failure is not None and _is_source_unavailable_observation(tool_message_text):
                final_text = (
                    f"Не могу выполнить запрос: источник данных или tool `{tool_name}` "
                    "недоступен. Проверьте подключение или включите нужный источник."
                )
                reasoning = reasoning or final_text
                stop_after_repeated_error = True
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

        if stop_after_repeated_error:
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
        # Max iterations reached — try to recover text from collector.
        text_collector = next(
            (cb for cb in callbacks if isinstance(cb, LLMTextCollector)), None
        )
        if text_collector and text_collector.messages:
            final_text = text_collector.messages[-1].get("text", "")

    artifacts, collector_tool_calls, collector_tool_names = collect_tool_stats(callbacks)
    total_tool_calls = max(total_tool_calls, collector_tool_calls)
    all_tool_names = _merge_tool_names(all_tool_names, collector_tool_names)

    if not final_text and artifacts:
        final_text = artifact_summary_text(artifacts)

    request.messages = messages
    return AgentResponse(
        final_text=final_text.strip(),
        reasoning=reasoning,
        reasoning_steps=reasoning_steps,
        artifacts=artifacts,
        route="analysis",
        tool_calls=total_tool_calls,
        tool_names=all_tool_names,
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
