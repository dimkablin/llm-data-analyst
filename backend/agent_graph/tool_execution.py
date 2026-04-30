from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

import pandas as pd
from langchain_core.messages import ToolMessage

from backend.agent_graph.message_codec import message_to_state
from backend.agent_graph.state import ArtifactRefState, MessageState, ToolCallState


@dataclass(slots=True)
class ToolExecutionSummary:
    messages: list[MessageState]
    tool_names: list[str]
    tool_call_count: int
    completed_actions: list[str]
    artifact_refs: list[ArtifactRefState]


@dataclass(slots=True)
class ToolCallExecutor:
    """Executes model-requested tools and converts outputs to ToolMessages."""

    tools: list[Any]
    callbacks: list[Any]
    metadata: dict[str, Any]

    def execute(self, tool_calls: list[ToolCallState]) -> ToolExecutionSummary:
        tool_map = {
            str(getattr(tool, "name", "")).strip(): tool
            for tool in self.tools
            if str(getattr(tool, "name", "")).strip()
        }
        runtime_config: dict[str, Any] = {"callbacks": self.callbacks}
        if self.metadata:
            runtime_config["metadata"] = self.metadata

        messages: list[MessageState] = []
        tool_names: list[str] = []
        completed_actions: list[str] = []
        artifact_refs: list[ArtifactRefState] = []

        for call in tool_calls:
            tool_name = str(call.get("name") or "").strip()
            tool_call_id = str(call.get("id") or "")
            tool_names.append(tool_name)
            tool = tool_map.get(tool_name)
            if tool is None:
                text = f"Unknown tool: {tool_name}"
            else:
                try:
                    result = tool.invoke(
                        {
                            "name": tool_name,
                            "args": dict(call.get("args") or {}),
                            "id": tool_call_id,
                            "type": "tool_call",
                        },
                        config=runtime_config,
                    )
                    mapped = ToolResultMapper(tool_name=tool_name).map(result)
                    text = mapped.text
                    artifact_refs.extend(mapped.artifact_refs)
                except Exception as exc:
                    text = f"Tool error: {exc}"

            completed_actions.append(f"{tool_name} -> {text[:80]}")
            messages.append(
                message_to_state(
                    ToolMessage(content=text, tool_call_id=tool_call_id),
                ),
            )

        return ToolExecutionSummary(
            messages=messages,
            tool_names=tool_names,
            tool_call_count=len(tool_calls),
            completed_actions=completed_actions,
            artifact_refs=artifact_refs,
        )


@dataclass(slots=True)
class MappedToolResult:
    text: str
    artifact_refs: list[ArtifactRefState]


@dataclass(slots=True)
class ToolResultMapper:
    """Converts LangChain tool outputs into LLM-safe text and artifact refs."""

    tool_name: str

    def map(self, result: object) -> MappedToolResult:
        text = self._tool_message_text(result)
        artifact = self._extract_artifact(result)
        artifact_refs = self._artifact_refs(artifact)
        if artifact:
            artifact_preview = self._artifact_preview(artifact)
            if artifact_preview:
                text = "\n\n".join(part for part in (text, artifact_preview) if part).strip()
        return MappedToolResult(text=text, artifact_refs=artifact_refs)

    @staticmethod
    def _tool_message_text(result: object) -> str:
        if isinstance(result, ToolMessage):
            return str(result.content or "")
        if hasattr(result, "content"):
            content = getattr(result, "content", "")
            if content:
                return str(content)
        if isinstance(result, tuple) and result:
            return str(result[0] or "")
        return str(result or "")

    @staticmethod
    def _extract_artifact(result: object) -> dict[str, Any]:
        if isinstance(result, ToolMessage):
            artifact = getattr(result, "artifact", None)
            return dict(artifact) if isinstance(artifact, dict) else {}
        if hasattr(result, "artifact"):
            artifact = getattr(result, "artifact", None)
            return dict(artifact) if isinstance(artifact, dict) else {}
        if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], dict):
            return dict(result[1])
        return {}

    def _artifact_refs(self, artifact: dict[str, Any]) -> list[ArtifactRefState]:
        artifact_type = str(artifact.get("artifact_type") or "").strip()
        items = artifact.get("items")
        if not artifact_type and isinstance(artifact, dict):
            for key in ("table", "value", "plot", "json"):
                if isinstance(artifact.get(key), dict):
                    artifact_type = key
                    items = artifact[key]
                    break
        if not artifact_type or not isinstance(items, dict):
            return []

        refs: list[ArtifactRefState] = []
        for name, payload in items.items():
            ref: ArtifactRefState = {
                "id": uuid.uuid4().hex,
                "name": str(name),
                "type": artifact_type,
                "tool_name": self.tool_name,
                "step_index": 0,
                "schema": self._schema(payload),
                "row_count": self._row_count(payload),
                "summary": self._summary(name, artifact_type, payload),
            }
            refs.append(ref)
        return refs

    @staticmethod
    def _schema(payload: object) -> dict[str, str] | None:
        if isinstance(payload, pd.DataFrame):
            return {str(col): str(dtype) for col, dtype in payload.dtypes.items()}
        return None

    @staticmethod
    def _row_count(payload: object) -> int | None:
        if isinstance(payload, pd.DataFrame):
            return len(payload)
        if isinstance(payload, list):
            return len(payload)
        return None

    @staticmethod
    def _summary(name: object, artifact_type: str, payload: object) -> str | None:
        if isinstance(payload, pd.DataFrame):
            return f"{name}, {len(payload)}x{len(payload.columns)}"
        if artifact_type == "value":
            return str(payload)[:120]
        return str(name)[:120]

    @staticmethod
    def _artifact_preview(artifact: dict[str, Any]) -> str:
        artifact_type = artifact.get("artifact_type")
        items = artifact.get("items")
        if not artifact_type or not isinstance(items, dict):
            return ""
        preview = {
            "artifact_type": artifact_type,
            "items": {
                str(name): ToolResultMapper._jsonable_preview(payload)
                for name, payload in list(items.items())[:5]
            },
        }
        return "ARTIFACT_RESULT:\n" + json.dumps(preview, ensure_ascii=False, default=str)[:2000]

    @staticmethod
    def _jsonable_preview(payload: object) -> object:
        if isinstance(payload, pd.DataFrame):
            return {
                "rows": len(payload),
                "columns": list(map(str, payload.columns[:20])),
                "sample": payload.head(5).to_dict(orient="records"),
            }
        return payload
