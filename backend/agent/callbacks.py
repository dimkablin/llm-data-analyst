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


def strip_thinking(text: str) -> str:
    return THINKING_RE.sub("", text).strip()


def extract_thinking(text: str) -> str:
    parts = THINKING_RE.findall(text or "")
    if not parts:
        return ""
    merged = "\n".join(parts)
    return re.sub(r"</?think>", "", merged, flags=re.IGNORECASE).strip()


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
        event = {
            "phase": "start",
            "tool_name": tool_name or "unknown",
            "input_preview": input_str[:360],
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self.events.append(event)
        input_code = self._extract_input_code(input_str[:2000])
        input_summary = self._build_input_summary(tool_name or "unknown", input_str[:2000])
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

    def absorb_tool_message(self, tool_name: str, result: ToolMessage) -> None:
        """Extract and store artifacts from a ToolMessage returned by tool.invoke().

        LangChain 1.x calls on_tool_end(content_string) for content_and_artifact tools,
        so the artifact payload in ToolMessage.artifact is never seen by on_tool_end.
        This method recovers it without double-counting tool_calls or events.
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
                return None
            return parsed

        if isinstance(output, tuple) and len(output) >= 2:
            possible_artifact = output[1]
            if isinstance(possible_artifact, dict):
                return possible_artifact
        if isinstance(output, ToolMessage):
            artifact = getattr(output, "artifact", None)
            if isinstance(artifact, dict):
                return artifact
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

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        self.queue = queue
        self.loop = loop
        self._inside_think = False
        self._buffer = ""
        self.reasoning_tokens_emitted = 0
        # Per-call chunks (reset after each on_llm_end → thinking_end emission)
        self.reasoning_chunks: list[str] = []
        # Cumulative across all LLM calls (for collected_reasoning())
        self._all_reasoning: list[str] = []
        self._thinking_started_this_call: bool = False

    def _extract_stream_parts(self, token: str) -> tuple[str, str]:
        self._buffer += token
        visible_parts: list[str] = []
        reasoning_parts: list[str] = []
        think_open = "<think>"
        think_close = "</think>"

        while self._buffer:
            lower_buffer = self._buffer.lower()
            if self._inside_think:
                close_idx = lower_buffer.find(think_close)
                if close_idx == -1:
                    tail_len = len(think_close) - 1
                    if len(self._buffer) <= tail_len:
                        break
                    reasoning_parts.append(self._buffer[:-tail_len])
                    self._buffer = self._buffer[-tail_len:]
                    break
                if close_idx > 0:
                    reasoning_parts.append(self._buffer[:close_idx])
                self._buffer = self._buffer[close_idx + len(think_close) :]
                self._inside_think = False
                continue

            open_idx = lower_buffer.find(think_open)
            if open_idx == -1:
                tail_len = len(think_open) - 1
                if len(self._buffer) <= tail_len:
                    break
                visible_parts.append(self._buffer[:-tail_len])
                self._buffer = self._buffer[-tail_len:]
                break

            if open_idx > 0:
                visible_parts.append(self._buffer[:open_idx])
            self._buffer = self._buffer[open_idx + len(think_open) :]
            self._inside_think = True

        return "".join(visible_parts), "".join(reasoning_parts)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if not token:
            return
        visible, reasoning = self._extract_stream_parts(token)
        if visible:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, ("token", visible))
        if reasoning:
            # Emit thinking_start on the first reasoning token of this LLM call
            if not self._thinking_started_this_call:
                self._thinking_started_this_call = True
                self.loop.call_soon_threadsafe(
                    self.queue.put_nowait, ("thinking_start", None)
                )
            self.reasoning_chunks.append(reasoning)
            self._all_reasoning.append(reasoning)
            self.reasoning_tokens_emitted += 1
            self.loop.call_soon_threadsafe(
                self.queue.put_nowait,
                ("reasoning_token", reasoning),
            )

    def on_llm_end(self, response: object, **kwargs: Any) -> None:
        # Flush any buffered partial tag text
        if self._buffer:
            trailing = self._buffer
            self._buffer = ""
            if self._inside_think:
                if not self._thinking_started_this_call:
                    self._thinking_started_this_call = True
                    self.loop.call_soon_threadsafe(
                        self.queue.put_nowait, ("thinking_start", None)
                    )
                self.reasoning_chunks.append(trailing)
                self._all_reasoning.append(trailing)
                self.reasoning_tokens_emitted += 1
                self.loop.call_soon_threadsafe(
                    self.queue.put_nowait,
                    ("reasoning_token", trailing),
                )
            else:
                self.loop.call_soon_threadsafe(self.queue.put_nowait, ("token", trailing))
        self._inside_think = False

        # Emit thinking_end with the complete thinking block for this LLM call
        if self.reasoning_chunks:
            complete_thinking = "".join(self.reasoning_chunks)
            self.loop.call_soon_threadsafe(
                self.queue.put_nowait, ("thinking_end", complete_thinking)
            )
            # Reset per-call state
            self.reasoning_chunks = []
            self.reasoning_tokens_emitted = 0
        self._thinking_started_this_call = False

    def collected_reasoning(self) -> str | None:
        """Return all reasoning collected across every LLM call."""
        merged = "".join(self._all_reasoning).strip()
        return merged or None


