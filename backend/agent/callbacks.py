from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage

from backend.artifacts.artifact_meta import build_artifact_meta, extract_artifact_hints
from backend.artifacts.execution import (
    ExecArtifactType,
    ExecutionArtifact,
    ExecutionStore,
)

THINKING_RE = re.compile(r"<think>[\s\S]*?<\/think>", re.IGNORECASE)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_OPEN_LEN = len(_THINK_OPEN)
_THINK_CLOSE_LEN = len(_THINK_CLOSE)


class ThinkingOutputParser:
    """Stateful incremental parser that separates visible text from ``<think>`` blocks.

    Safe to feed one token at a time (streaming) or the full text at once.
    Handles:
    - Closed ``<think>...</think>`` blocks
    - Unclosed ``<think>`` (discards from open tag to end — no leak)
    - Multiple ``<think>`` blocks
    - Tags split across consecutive ``feed()`` calls
    - Case-insensitive tags
    """

    def __init__(self) -> None:
        self._buf: str = ""
        self._inside: bool = False
        self._visible: list[str] = []
        self._reasoning: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, text: str) -> tuple[str, str]:
        """Process *text*, return ``(visible, reasoning)`` extracted from this chunk."""
        self._buf += text
        vis: list[str] = []
        rsn: list[str] = []

        while self._buf:
            lower = self._buf.lower()
            if self._inside:
                idx = lower.find(_THINK_CLOSE)
                if idx == -1:
                    # Keep a tail that could be a partial closing tag.
                    keep = _THINK_CLOSE_LEN - 1
                    if len(self._buf) > keep:
                        rsn.append(self._buf[:-keep])
                        self._buf = self._buf[-keep:]
                    break
                if idx > 0:
                    rsn.append(self._buf[:idx])
                self._buf = self._buf[idx + _THINK_CLOSE_LEN:]
                self._inside = False
            else:
                idx_open = lower.find(_THINK_OPEN)
                idx_close = lower.find(_THINK_CLOSE)

                if idx_open == -1 and idx_close == -1:
                    # Neither tag present — keep a tail long enough to span the longest tag.
                    keep = max(_THINK_OPEN_LEN, _THINK_CLOSE_LEN) - 1
                    if len(self._buf) > keep:
                        vis.append(self._buf[:-keep])
                        self._buf = self._buf[-keep:]
                    break

                # Orphaned </think> without preceding <think>: treat content before it as
                # reasoning (vLLM strips the opening tag server-side), discard the tag itself.
                if idx_close != -1 and (idx_open == -1 or idx_close < idx_open):
                    if idx_close > 0:
                        rsn.append(self._buf[:idx_close])
                    self._buf = self._buf[idx_close + _THINK_CLOSE_LEN:]
                    # _inside stays False — continue processing what follows
                    continue

                # Normal <think> found first.
                if idx_open > 0:
                    vis.append(self._buf[:idx_open])
                self._buf = self._buf[idx_open + _THINK_OPEN_LEN:]
                self._inside = True

        v, r = "".join(vis), "".join(rsn)
        self._visible.append(v)
        self._reasoning.append(r)
        return v, r

    def flush(self) -> tuple[str, str]:
        """Finalise the stream.

        - If inside an unclosed ``<think>``: discard the buffer (no leak).
        - Otherwise: emit remaining buffer as visible text.

        Returns ``(visible, reasoning)`` for any remaining content.
        """
        v, r = "", ""
        if self._buf:
            if self._inside:
                # Unclosed block — discard to prevent reasoning leaking downstream.
                r = self._buf
                self._reasoning.append(r)
            else:
                v = self._buf
                self._visible.append(v)
            self._buf = ""
        self._inside = False
        return v, r

    def visible(self) -> str:
        """All visible text collected so far, stripped."""
        return "".join(self._visible).strip()

    def reasoning(self) -> str:
        """All reasoning text collected so far, stripped."""
        return "".join(self._reasoning).strip()


def strip_thinking(text: str) -> str:
    parser = ThinkingOutputParser()
    parser.feed(str(text or ""))
    parser.flush()
    return parser.visible()


def extract_thinking(text: str) -> str:
    parser = ThinkingOutputParser()
    parser.feed(str(text or ""))
    parser.flush()
    return parser.reasoning()


class LLMTextCollector(BaseCallbackHandler):
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def on_llm_end(self, response: object, **kwargs: Any) -> None:
        text = ""
        generations = getattr(response, "generations", None)
        if isinstance(generations, list):
            for group in generations:
                if not isinstance(group, list):
                    continue
                for candidate in group:
                    candidate_text = getattr(candidate, "text", "")
                    if isinstance(candidate_text, str) and candidate_text.strip():
                        text = candidate_text
                        break
                    message = getattr(candidate, "message", None)
                    message_content = getattr(message, "content", "")
                    if isinstance(message_content, str) and message_content.strip():
                        text = message_content
                        break
                if text:
                    break
        if not text:
            return

        filtered = strip_thinking(text)
        reasoning = extract_thinking(text)
        if filtered or reasoning:
            self.messages.append({"text": filtered, "reasoning": reasoning})


class ToolCollector(BaseCallbackHandler):
    def __init__(
        self,
        source_context: dict[str, Any] | None = None,
        queue: asyncio.Queue | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        execution_store: ExecutionStore | None = None,
    ) -> None:
        self.artifacts: list[ExecutionArtifact] = []
        self.tool_calls: int = 0
        self.tool_names: list[str] = []
        self.events: list[dict[str, Any]] = []
        self._last_tool_name: str | None = None
        self._source_context = dict(source_context or {})
        self._queue = queue
        self._loop = loop
        self.execution_store: ExecutionStore = execution_store or ExecutionStore(session_id="")
        self.graph_tracker: Any | None = None  # Set externally for graph visualization
        self._step_index: int = 0
        self._phase_collector_ref: Any | None = None  # For graph version bumps
        self.token_callback: Any | None = None  # Set externally to TokenStreamingCallback

    def _push_event(self, event_type: str, data: Any) -> None:
        """Push event directly to SSE queue if available."""
        if self._queue is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, (event_type, data)
            )

    @staticmethod
    def _resolve_tool_name(
        *,
        tool: object | None = None,
        serialized: dict[str, Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> str | None:
        local_kwargs = kwargs or {}
        if isinstance(tool, str) and tool.strip():
            return tool.strip()
        if tool is not None:
            attr_name = getattr(tool, "name", None)
            if isinstance(attr_name, str) and attr_name.strip():
                return attr_name.strip()
            text = str(tool).strip()
            if text:
                return text
        if serialized and isinstance(serialized.get("name"), str):
            name = str(serialized["name"]).strip()
            if name:
                return name
        for key in ("name", "tool_name"):
            candidate = local_kwargs.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None

    @staticmethod
    def _extract_input_code(raw: str) -> str | None:
        """Extract the code/query string from a tool input JSON preview."""
        text = raw.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        if isinstance(parsed, dict):
            for key in ("code", "query", "input", "command"):
                candidate = parsed.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
        return None

    @staticmethod
    def _build_input_summary(tool_name: str, raw: str) -> str:
        """Build a short 1-line summary of what the tool input does."""
        text = raw.strip()
        if not text:
            return ""
        try:
            parsed = json.loads(text)
        except Exception:
            return text[:80]
        if isinstance(parsed, dict):
            for key in ("code", "query", "command"):
                candidate = parsed.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    first_line = candidate.strip().split("\n")[0][:80]
                    return first_line
            for key in ("question", "answer"):
                candidate = parsed.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()[:80]
            for key in ("path", "file_path", "dataset", "alias", "pattern", "tool_name"):
                candidate = parsed.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()[:80]
        return text[:80]

    @staticmethod
    def _build_result_summary(
        payload: object, event_payload: dict[str, Any],
    ) -> str:
        """Build a short human-readable result summary for UI display."""
        if not isinstance(payload, dict):
            return ""
        parts: list[str] = []
        status = event_payload.get("status", "ok")
        if status == "error":
            error_text = str(event_payload.get("error", "")).strip()
            if error_text:
                last_line = error_text.splitlines()[-1][:120]
                return f"Error: {last_line}"
            return "Error"
        # Summarize artifact types produced
        if "table" in payload and isinstance(payload["table"], dict):
            for name, table_data in payload["table"].items():
                if table_data is None:
                    continue
                try:
                    import pandas as _pd
                    if isinstance(table_data, _pd.DataFrame):
                        parts.append(f"{name}: {len(table_data)} rows, {len(table_data.columns)} cols")
                    elif isinstance(table_data, dict):
                        parts.append(f"{name}: table")
                    elif isinstance(table_data, list):
                        parts.append(f"{name}: {len(table_data)} rows")
                except Exception:
                    parts.append(f"{name}: table")
        if "plot" in payload and isinstance(payload["plot"], dict):
            plot_names = [n for n in payload["plot"] if payload["plot"][n] is not None]
            if plot_names:
                parts.append(f"chart: {', '.join(plot_names[:3])}")
        if "value" in payload and isinstance(payload["value"], dict):
            value_items = {
                k: v for k, v in payload["value"].items() if v is not None
            }
            if value_items:
                formatted = []
                for k, v in list(value_items.items())[:3]:
                    if isinstance(v, float):
                        formatted.append(f"{k}={v:.2f}")
                    else:
                        formatted.append(f"{k}={v}")
                parts.append(", ".join(formatted))
        return "; ".join(parts) if parts else ""

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        tool_name = self._resolve_tool_name(
            serialized=serialized,
            kwargs=kwargs,
        )
        if tool_name:
            self._last_tool_name = tool_name
            self.tool_names.append(tool_name)
        input_summary = self._build_input_summary(tool_name or "unknown", input_str[:2000])
        input_code = self._extract_input_code(input_str[:2000])
        pre_reasoning = (
            self.token_callback.take_pending_thinking()
            if self.token_callback is not None
            else ""
        )
        event: dict[str, Any] = {
            "phase": "start",
            "tool_name": tool_name or "unknown",
            "input_preview": input_str[:360],
            "input_summary": input_summary,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if pre_reasoning:
            event["pre_reasoning"] = pre_reasoning
        self.events.append(event)
        push_data: dict[str, Any] = {
            "tool_name": event["tool_name"],
            "input_preview": event["input_preview"],
            "input_summary": input_summary,
        }
        if input_code:
            push_data["input_code"] = input_code[:1200]
        self._push_event("tool_start", push_data)
        if self.graph_tracker is not None:
            self.graph_tracker.tool_start(event["tool_name"], self._step_index)
            if self._phase_collector_ref is not None:
                self._phase_collector_ref._graph_version += 1  # noqa: SLF001

    def on_tool_error(self, error: BaseException | str, tool=None, **kwargs: Any) -> None:
        self.tool_calls += 1
        tool_name = self._resolve_tool_name(tool=tool, kwargs=kwargs) or self._last_tool_name
        if tool_name:
            self.tool_names.append(tool_name)

        error_text = str(error)
        event_payload: dict[str, Any] = {
            "phase": "end",
            "tool_name": tool_name or "unknown",
            "status": "error",
            "error": error_text,
            "artifact_keys": [],
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self.events.append(event_payload)
        self._push_event("tool_end", {
            "tool_name": tool_name or "unknown",
            "status": "error",
            "artifact_keys": [],
            "result_summary": error_text[:300],
            "output_preview": error_text[:800],
        })

    def on_tool_end(self, output: object, tool=None, **kwargs: Any) -> None:
        self.tool_calls += 1
        payload = self._normalize_output(output)
        tool_name = self._resolve_tool_name(tool=tool, kwargs=kwargs) or self._last_tool_name
        if tool_name:
            self.tool_names.append(tool_name)
        if not isinstance(payload, dict):
            self.events.append(
                {
                    "phase": "end",
                    "tool_name": tool_name or "unknown",
                    "status": "empty_output",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
            return
        if "artifact" in payload and isinstance(payload["artifact"], dict):
            payload = payload["artifact"]
        tool_code = payload.get("code") if isinstance(payload.get("code"), str) else None
        artifact_hints = extract_artifact_hints(payload)
        artifact_meta = build_artifact_meta(
            tool_name=tool_name,
            tool_code=tool_code,
            source_context=self._source_context,
            artifact_hints=artifact_hints,
        )
        event_payload: dict[str, Any] = {
            "phase": "end",
            "tool_name": str(tool_name or "unknown"),
            "timestamp": datetime.now(UTC).isoformat(),
            "status": "ok",
            "artifact_keys": [],
        }
        if tool_code:
            event_payload["code_preview"] = tool_code[:1200]
        source_meta = artifact_meta.get("source")
        if isinstance(source_meta, dict):
            source_type = str(source_meta.get("source_type", "")).strip()
            if source_type:
                event_payload["source_type"] = source_type
        if isinstance(payload.get("text"), str) and payload.get("text", "").startswith("❌"):
            event_payload["status"] = "error"
            event_payload["error"] = str(payload.get("text"))

        producer = tool_name or "unknown"

        if "plot" in payload and isinstance(payload["plot"], dict):
            event_payload["artifact_keys"].append("plot")
            for name, fig in payload["plot"].items():
                if fig is None:
                    continue
                ea = self.execution_store.put(ExecutionArtifact(
                    artifact_type=ExecArtifactType.PLOT,
                    producer_tool=producer,
                    data=fig,
                    name=name,
                    meta=dict(artifact_meta),
                ))
                self.artifacts.append(ea)

        if "table" in payload and isinstance(payload["table"], dict):
            event_payload["artifact_keys"].append("table")
            for name, table in payload["table"].items():
                if table is None:
                    continue
                ea = self.execution_store.put(ExecutionArtifact(
                    artifact_type=ExecArtifactType.DATAFRAME,
                    producer_tool=producer,
                    data=table,
                    name=name,
                    meta=dict(artifact_meta),
                ))
                self.artifacts.append(ea)

        if "value" in payload and isinstance(payload["value"], dict):
            event_payload["artifact_keys"].append("value")
            clean_value_payload = {
                key: value
                for key, value in payload["value"].items()
                if value is not None
            }
            if not clean_value_payload:
                self.events.append(event_payload)
                return
            ea = self.execution_store.put(ExecutionArtifact(
                artifact_type=ExecArtifactType.SCALAR,
                producer_tool=producer,
                data=clean_value_payload,
                name="values",
                meta=dict(artifact_meta),
            ))
            self.artifacts.append(ea)

        if "json" in payload and isinstance(payload["json"], dict):
            event_payload["artifact_keys"].append("json")
            for name, json_data in payload["json"].items():
                if json_data is None:
                    continue
                ea = self.execution_store.put(ExecutionArtifact(
                    artifact_type=ExecArtifactType.JSON,
                    producer_tool=producer,
                    data=json_data,
                    name=name,
                    meta=dict(artifact_meta),
                ))
                self.artifacts.append(ea)

        self.events.append(event_payload)

        # Build a concise result summary for UI display
        result_summary = self._build_result_summary(payload, event_payload)

        push_payload: dict[str, Any] = {
            "tool_name": event_payload["tool_name"],
            "status": event_payload["status"],
            "artifact_keys": event_payload.get("artifact_keys", []),
        }
        if result_summary:
            push_payload["result_summary"] = result_summary
        raw_text = payload.get("text") if isinstance(payload, dict) else None
        if isinstance(raw_text, str) and raw_text.strip():
            push_payload["output_preview"] = raw_text.strip()[:800]
        self._push_event("tool_end", push_payload)

    def to_persisted_activities(self) -> list[dict[str, Any]]:
        """Merge start/end event pairs into PersistedToolCall records for storage.

        Returns one dict per tool invocation with a stable schema independent
        of the live streaming DTO (StreamToolCall).
        """
        activities: list[dict[str, Any]] = []
        for event in self.events:
            phase = event.get("phase")
            tool_name = str(event.get("tool_name") or "unknown")
            if phase == "start":
                activity: dict[str, Any] = {
                    "tool_name": tool_name,
                    "status": "done",
                    "input_summary": event.get("input_summary") or "",
                    "input_preview": event.get("input_preview") or "",
                    "artifact_keys": [],
                    "started_at": event.get("timestamp"),
                    "finished_at": None,
                }
                if event.get("pre_reasoning"):
                    activity["pre_reasoning"] = event["pre_reasoning"]
                activities.append(activity)
            elif phase == "end":
                # Pair with the last unfinished start for this tool name
                for activity in reversed(activities):
                    if activity["tool_name"] == tool_name and activity["finished_at"] is None:
                        activity["status"] = "error" if event.get("status") == "error" else "done"
                        activity["artifact_keys"] = list(event.get("artifact_keys") or [])
                        activity["finished_at"] = event.get("timestamp")
                        if event.get("error"):
                            activity["error"] = str(event["error"])[:300]
                        break
        # Strip unpaired/empty entries and clean None finished_at
        result = []
        for a in activities:
            if a.get("finished_at") is None:
                a["finished_at"] = a["started_at"]
            result.append(a)
        return result

    def absorb_tool_message(self, tool_name: str, result: ToolMessage) -> None:
        """Extract and store artifacts from a ToolMessage returned by tool.invoke().

        LangChain 1.0.0 passes a ToolMessage to on_tool_end, so _normalize_output
        already captures artifacts via ToolMessage.artifact.  This method is a
        safety net for callers that have a ToolMessage but did not go through the
        standard on_tool_end callback path (e.g. manual tool invocation loops).

        NOTE: do NOT call this for tools already processed by on_tool_end — it
        would create duplicate ExecutionArtifact entries.
        """
        artifact = getattr(result, "artifact", None)
        if not isinstance(artifact, dict):
            return

        # Normalize items-envelope format produced by BaseExecTool:
        # {"schema_version": "1.0", "artifact_type": "plot", "items": {...}}
        # → merge items under the artifact_type key so downstream checks work.
        artifact_type = artifact.get("artifact_type", "")
        items = artifact.get("items")
        if isinstance(items, dict) and artifact_type:
            artifact = {**artifact, artifact_type: items}

        tool_code = artifact.get("code") if isinstance(artifact.get("code"), str) else None
        artifact_hints = extract_artifact_hints(artifact)
        artifact_meta = build_artifact_meta(
            tool_name=tool_name,
            tool_code=tool_code,
            source_context=self._source_context,
            artifact_hints=artifact_hints,
        )
        producer = tool_name or "unknown"
        artifact_keys: list[str] = []

        if "plot" in artifact and isinstance(artifact["plot"], dict):
            artifact_keys.append("plot")
            for name, fig in artifact["plot"].items():
                if fig is None:
                    continue
                ea = self.execution_store.put(ExecutionArtifact(
                    artifact_type=ExecArtifactType.PLOT,
                    producer_tool=producer,
                    data=fig,
                    name=name,
                    meta=dict(artifact_meta),
                ))
                self.artifacts.append(ea)
        if "table" in artifact and isinstance(artifact["table"], dict):
            artifact_keys.append("table")
            for name, table in artifact["table"].items():
                if table is None:
                    continue
                ea = self.execution_store.put(ExecutionArtifact(
                    artifact_type=ExecArtifactType.DATAFRAME,
                    producer_tool=producer,
                    data=table,
                    name=name,
                    meta=dict(artifact_meta),
                ))
                self.artifacts.append(ea)
        if "value" in artifact and isinstance(artifact["value"], dict):
            clean = {k: v for k, v in artifact["value"].items() if v is not None}
            if clean:
                artifact_keys.append("value")
                ea = self.execution_store.put(ExecutionArtifact(
                    artifact_type=ExecArtifactType.SCALAR,
                    producer_tool=producer,
                    data=clean,
                    name="values",
                    meta=dict(artifact_meta),
                ))
                self.artifacts.append(ea)
        if "json" in artifact and isinstance(artifact["json"], dict):
            artifact_keys.append("json")
            for name, json_data in artifact["json"].items():
                if json_data is None:
                    continue
                ea = self.execution_store.put(ExecutionArtifact(
                    artifact_type=ExecArtifactType.JSON,
                    producer_tool=producer,
                    data=json_data,
                    name=name,
                    meta=dict(artifact_meta),
                ))
                self.artifacts.append(ea)
        if self.graph_tracker is not None:
            self.graph_tracker.tool_end(
                tool_name,
                self._step_index,
                status="done",
                artifact_keys=artifact_keys,
            )
            if self._phase_collector_ref is not None:
                self._phase_collector_ref._graph_version += 1  # noqa: SLF001

    @staticmethod
    def _normalize_output(output: object) -> object:
        def _from_text(raw: str) -> object:
            value = raw.strip()
            if not value:
                return None
            try:
                parsed = json.loads(value)
            except Exception:
                return {"text": value}
            return parsed

        if isinstance(output, tuple) and len(output) >= 2:
            possible_artifact = output[1]
            if isinstance(possible_artifact, dict):
                return possible_artifact
        if isinstance(output, ToolMessage):
            artifact = getattr(output, "artifact", None)
            if isinstance(artifact, dict):
                payload = dict(artifact)
                # Unwrap ToolResultEnvelope: {artifact_type, items} → {artifact_type: items, ...}
                artifact_type = payload.get("artifact_type", "")
                items = payload.get("items")
                if artifact_type and isinstance(items, dict):
                    payload[artifact_type] = items
                return payload
            content = getattr(output, "content", "")
            if isinstance(content, str):
                return _from_text(content)
            return None
        if isinstance(output, str):
            return _from_text(output)
        return output


class AgentProgressCollector(BaseCallbackHandler):
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def add_event(
        self,
        *,
        phase: str,
        title: str,
        details: str = "",
        step_index: int | None = None,
        max_steps: int | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "phase": str(phase or "").strip() or "info",
            "title": str(title or "").strip() or "Обновление",
            "details": str(details or "").strip(),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if isinstance(step_index, int):
            payload["step_index"] = step_index
        if isinstance(max_steps, int):
            payload["max_steps"] = max_steps
        self.events.append(payload)


class PhaseCollector(BaseCallbackHandler):
    """Collects structured ReAct phase events (think / act / evaluate / finalize)."""

    ignore_llm = True
    ignore_chain = True
    ignore_agent = True
    ignore_retriever = True
    ignore_chat_model = True

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []
        self.graph_tracker: Any | None = None
        self._graph_version: int = 0

    def add_phase(
        self,
        *,
        phase: str,
        title: str,
        content: str = "",
        step_index: int | None = None,
        max_steps: int | None = None,
        status: str | None = None,
    ) -> None:
        sid = step_index if isinstance(step_index, int) else 0
        phase_id = f"{phase}-{sid}"
        payload: dict[str, Any] = {
            "id": phase_id,
            "phase": str(phase or "").strip() or "info",
            "title": str(title or "").strip(),
            "content": str(content or "").strip(),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if isinstance(step_index, int):
            payload["step_index"] = step_index
        if isinstance(max_steps, int):
            payload["max_steps"] = max_steps
        if status is not None:
            payload["status"] = str(status).strip()
        self.events.append(payload)


class TokenStreamCallbackHandler(BaseCallbackHandler):
    """Streams LLM tokens to the SSE queue.

    Splits the raw token stream into two channels:
    - ``token`` events  — visible text outside ``<think>`` tags (final answer)
    - ``reasoning_token`` events — text inside ``<think>`` tags (live thinking display)

    Additionally emits block-level signals:
    - ``thinking_start`` — first reasoning token of each LLM call (no data)
    - ``thinking_end``   — complete thinking text when the LLM call ends
    """

    def __init__(
        self,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        *,
        show_think: bool = True,
    ) -> None:
        self.queue = queue
        self.loop = loop
        self._show_think = show_think
        # Per-call parser (reset after each on_llm_end).
        self._stream_parser: ThinkingOutputParser = ThinkingOutputParser()
        self.reasoning_tokens_emitted = 0
        # Per-call reasoning chunks (reset after each on_llm_end → thinking_end emission)
        self.reasoning_chunks: list[str] = []
        # Cumulative across all LLM calls (for collected_reasoning())
        self._all_reasoning: list[str] = []
        self._thinking_started_this_call: bool = False
        # Last completed thinking block — consumed by ToolCollector on tool_start
        self._pending_thinking: str = ""
        # Per-step reasoning: one entry per on_llm_end that had thinking
        self._per_step_reasoning: list[str] = []

    def _emit_reasoning(self, text: str) -> None:
        if not text:
            return
        # Always accumulate so collected_reasoning() stays accurate.
        self.reasoning_chunks.append(text)
        self._all_reasoning.append(text)
        if not self._show_think:
            # Thinking suppressed — parsed and stripped but never forwarded to the SSE queue.
            return
        if not self._thinking_started_this_call:
            self._thinking_started_this_call = True
            self.loop.call_soon_threadsafe(
                self.queue.put_nowait, ("thinking_start", None)
            )
        self.reasoning_tokens_emitted += 1
        self.loop.call_soon_threadsafe(self.queue.put_nowait, ("reasoning_token", text))

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # Ollama streams reasoning in ChatGenerationChunk.message.additional_kwargs["reasoning"].
        chunk = kwargs.get("chunk")
        if chunk is not None:
            msg = getattr(chunk, "message", None)
            chunk_reasoning = (getattr(msg, "additional_kwargs", {}) or {}).get("reasoning", "")
            if chunk_reasoning:
                self._emit_reasoning(chunk_reasoning)
        if not token:
            return
        visible, reasoning = self._stream_parser.feed(token)
        if visible:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, ("token", visible))
        if reasoning:
            self._emit_reasoning(reasoning)

    def on_llm_end(self, response: object, **kwargs: Any) -> None:
        # Flush any buffered partial tag text; unclosed <think> is discarded (no leak).
        visible, reasoning = self._stream_parser.flush()
        if reasoning:
            self._emit_reasoning(reasoning)
        if visible:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, ("token", visible))

        # Reset per-call parser so the next LLM call starts clean.
        self._stream_parser = ThinkingOutputParser()

        # Ollama + Qwen3 returns thinking in the `reasoning` field of the message
        # (OpenAI-compatible API), not as <think> tags in the token stream.
        # If no reasoning was captured via tag parsing, extract it from the response.
        if not self.reasoning_chunks:
            try:
                gen = response.generations[0][0]  # type: ignore[union-attr]
                ak = getattr(gen.message, "additional_kwargs", {})
                ollama_reasoning = (ak or {}).get("reasoning", "")
                if ollama_reasoning:
                    self._emit_reasoning(ollama_reasoning)
            except (AttributeError, IndexError, TypeError):
                pass

        # Emit thinking_end with the complete thinking block for this LLM call.
        if self.reasoning_chunks:
            complete_thinking = "".join(self.reasoning_chunks)
            if self._show_think:
                self.loop.call_soon_threadsafe(
                    self.queue.put_nowait, ("thinking_end", complete_thinking)
                )
            self._pending_thinking = complete_thinking
            self._per_step_reasoning.append(complete_thinking)
            self.reasoning_chunks = []
            self.reasoning_tokens_emitted = 0
        self._thinking_started_this_call = False

    def collected_reasoning(self) -> str | None:
        """Return all reasoning collected across every LLM call."""
        merged = "".join(self._all_reasoning).strip()
        return merged or None

    def all_reasoning_steps(self) -> list[str]:
        """Raw reasoning text per LLM call (one entry per on_llm_end with thinking)."""
        return list(self._per_step_reasoning)

    def take_pending_thinking(self) -> str:
        """Return the last completed thinking block and clear it.

        Called by ToolCollector on tool_start to associate per-tool pre-reasoning.
        """
        thinking, self._pending_thinking = self._pending_thinking, ""
        return thinking
