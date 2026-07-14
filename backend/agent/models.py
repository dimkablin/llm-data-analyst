from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.sessions.session_memory import StructuredSessionMemory


class AgentRuntimeEffects(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_memory_notes: tuple[str, ...] = ()
    session_memory_notes: tuple[str, ...] = ()
    session_memory: StructuredSessionMemory | None = None


class AgentResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_text: str
    reasoning: str | None
    artifacts: list
    route: Literal["analysis", "summary"] = "analysis"
    tool_calls: int = 0
    tool_names: list[str] = Field(default_factory=list)
    llm_unreachable: bool = False
    reasoning_steps: list[str] = Field(default_factory=list)
    runtime_effects: AgentRuntimeEffects = Field(default_factory=AgentRuntimeEffects)


class QueryCacheEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    created_at: float
    response: AgentResponse
