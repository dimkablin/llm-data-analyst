from __future__ import annotations

import threading
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from backend.agent.models import AgentResponse


class AgentRunRequest(BaseModel):
    """Public input contract for the generic LangGraph runtime."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    df: pd.DataFrame | None = None
    prompt: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    use_history: bool = False
    include_reasoning: bool = False
    callbacks: list[Any] = Field(default_factory=list)
    trace_context: dict[str, Any] = Field(default_factory=dict)
    session_source: dict[str, Any] = Field(default_factory=dict)
    selected_skill_ids: list[str] = Field(default_factory=list)
    requested_tool_key: str | None = None
    cancel_event: threading.Event | None = None


class AgentRunResult(BaseModel):
    """Public output contract for a completed generic runtime turn."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    response: AgentResponse
