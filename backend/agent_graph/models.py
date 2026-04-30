from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.agent_graph.state import AgentGraphState, ArtifactRefState


@dataclass(slots=True)
class AgentGraphRequest:
    """Public request contract for the LangGraph agent runtime."""

    prompt: str
    history: list[dict[str, Any]] = field(default_factory=list)
    use_history: bool = True
    include_reasoning: bool = False
    trace_context: dict[str, Any] = field(default_factory=dict)
    session_source: dict[str, Any] = field(default_factory=dict)
    selected_skill_ids: list[str] = field(default_factory=list)
    max_steps: int = 1

    def to_state(self, *, runtime_context_key: str = "") -> AgentGraphState:
        return {
            "prompt": self.prompt,
            "history": list(self.history),
            "use_history": self.use_history,
            "include_reasoning": self.include_reasoning,
            "trace_context": dict(self.trace_context),
            "session_source": dict(self.session_source),
            "selected_skill_ids": list(self.selected_skill_ids),
            "runtime_context_key": runtime_context_key,
            "max_steps": max(1, int(self.max_steps)),
        }


@dataclass(slots=True)
class AgentGraphResult:
    """Public result contract for the LangGraph agent runtime."""

    final_text: str
    reasoning: str | None = None
    route: str = "analysis"
    stop_reason: str = ""
    tool_calls: int = 0
    tool_names: list[str] = field(default_factory=list)
    artifact_refs: list[ArtifactRefState] = field(default_factory=list)
    reasoning_steps: list[str] = field(default_factory=list)
    raw_state: AgentGraphState = field(default_factory=dict)

    @classmethod
    def from_state(cls, state: AgentGraphState) -> AgentGraphResult:
        return cls(
            final_text=str(state.get("final_text") or ""),
            reasoning=state.get("reasoning"),
            route=str(state.get("route") or "analysis"),
            stop_reason=str(state.get("stop_reason") or ""),
            tool_calls=int(state.get("tool_call_count") or 0),
            tool_names=list(state.get("tool_names") or []),
            artifact_refs=list(state.get("artifact_refs") or []),
            reasoning_steps=list(state.get("reasoning_steps") or []),
            raw_state=dict(state),
        )
