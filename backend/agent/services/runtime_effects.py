from __future__ import annotations

import copy

from pydantic import BaseModel, ConfigDict, Field

from backend.agent.models import AgentRuntimeEffects
from backend.sessions.session_memory import SessionMemory


class RuntimeEffectsRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_memory: SessionMemory
    user_memory_notes: list[str] | tuple[str, ...] = Field(default_factory=list)
    session_memory_notes: list[str] | tuple[str, ...] = Field(default_factory=list)


class RuntimeEffectsBuilder(BaseModel):
    """Build public runtime effects from per-request tool side-effect buffers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def build(self, request: RuntimeEffectsRequest) -> AgentRuntimeEffects:
        session_notes = self._clean_notes(request.session_memory_notes)
        user_notes = self._clean_notes(request.user_memory_notes)
        session_memory = copy.deepcopy(request.session_memory)
        if session_notes:
            existing_notes = str(session_memory.notes or "").strip()
            merged_notes = "\n".join(
                [*([existing_notes] if existing_notes else []), *session_notes]
            )
            session_memory.notes = merged_notes.strip()
        return AgentRuntimeEffects(
            user_memory_notes=user_notes,
            session_memory_notes=session_notes,
            session_memory=session_memory,
        )

    @staticmethod
    def _clean_notes(notes: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        return tuple(note.strip() for note in notes if str(note or "").strip())
