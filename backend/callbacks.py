from __future__ import annotations

import asyncio
import json
import re
from typing import Any
from datetime import datetime, timezone

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage

from backend.internal_models import ArtifactRecord


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
    def __init__(self) -> None:
        self.artifacts: list[ArtifactRecord] = []
        self.tool_calls: int = 0
        self.tool_names: list[str] = []
        self.events: list[dict[str, Any]] = []
        self._last_tool_name: str | None = None

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
        self.events.append(
            {
                "phase": "start",
                "tool_name": tool_name or "unknown",
                "input_preview": input_str[:360],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

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
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            return
        if "artifact" in payload and isinstance(payload["artifact"], dict):
            payload = payload["artifact"]
        tool_code = payload.get("code") if isinstance(payload.get("code"), str) else None
        artifact_meta: dict[str, Any] = {}
        if tool_name:
            artifact_meta["tool_name"] = str(tool_name)
        if tool_code:
            artifact_meta["code"] = tool_code
        event_payload: dict[str, Any] = {
            "phase": "end",
            "tool_name": str(tool_name or "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "ok",
            "artifact_keys": [],
        }
        if tool_code:
            event_payload["code_preview"] = tool_code[:1200]
        if isinstance(payload.get("text"), str) and payload.get("text", "").startswith("❌"):
            event_payload["status"] = "error"
            event_payload["error"] = str(payload.get("text"))

        if "plot" in payload and isinstance(payload["plot"], dict):
            event_payload["artifact_keys"].append("plot")
            for name, fig in payload["plot"].items():
                if fig is None:
                    continue
                self.artifacts.append(
                    ArtifactRecord(
                        artifact_type="plot",
                        data=fig,
                        text=name,
                        meta=dict(artifact_meta),
                    )
                )
        if "table" in payload and isinstance(payload["table"], dict):
            event_payload["artifact_keys"].append("table")
            for name, table in payload["table"].items():
                if table is None:
                    continue
                self.artifacts.append(
                    ArtifactRecord(
                        artifact_type="table",
                        data=table,
                        text=name,
                        meta=dict(artifact_meta),
                    )
                )
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
            self.artifacts.append(
                ArtifactRecord(
                    artifact_type="value",
                    data=clean_value_payload,
                    text="values",
                    meta=dict(artifact_meta),
                )
            )
        self.events.append(event_payload)

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
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if isinstance(step_index, int):
            payload["step_index"] = step_index
        if isinstance(max_steps, int):
            payload["max_steps"] = max_steps
        if status is not None:
            payload["status"] = str(status).strip()
        self.events.append(payload)


class PhaseTokenStreamHandler(BaseCallbackHandler):
    """Streams LLM tokens to the activity panel as ``phase_token`` SSE events.

    Strips ``<think>``/``</think>`` tags but preserves all content so the
    user can observe the full chain-of-thought in real time.
    """

    THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        self.queue = queue
        self.loop = loop

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if not token:
            return
        clean = self.THINK_TAG_RE.sub("", token)
        if clean:
            self.loop.call_soon_threadsafe(
                self.queue.put_nowait, ("phase_token", clean)
            )


class TokenStreamCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        self.queue = queue
        self.loop = loop
        self._inside_think = False
        self._buffer = ""
        self.reasoning_tokens_emitted = 0

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
            self.reasoning_tokens_emitted += 1
            self.loop.call_soon_threadsafe(
                self.queue.put_nowait,
                ("reasoning_token", reasoning),
            )

    def on_llm_end(self, response: object, **kwargs: Any) -> None:
        if self._buffer:
            trailing = self._buffer
            self._buffer = ""
            if self._inside_think:
                self.reasoning_tokens_emitted += 1
                self.loop.call_soon_threadsafe(
                    self.queue.put_nowait,
                    ("reasoning_token", trailing),
                )
            else:
                self.loop.call_soon_threadsafe(self.queue.put_nowait, ("token", trailing))
        self._inside_think = False
