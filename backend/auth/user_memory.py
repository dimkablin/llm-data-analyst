"""User memory service.

Maintains two memory types per user stored in SQLite (`user_memories` table):

  profile  – static facts about the user that the user controls manually
             (name, role, expertise, preferred language, etc.)

  notes    – short agent-written memos that accumulate across sessions;
             after each query cycle the agent may append new observations,
             and an async LLM "consolidation" run merges them so the notes
             block stays compact and useful.

Memory is injected into the agent's system prompt as a frozen snapshot at
the start of each request (see `UserMemoryService.build_memory_block`).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.auth.auth_db import AuthDB

logger = logging.getLogger(__name__)

MEM_PROFILE = "profile"
MEM_NOTES   = "notes"

_CONSOLIDATE_SYSTEM = """\
You are a memory curator for a data analytics AI assistant.

Your task: merge the EXISTING notes and the NEW notes into a single, \
compact, high-quality memory block about the USER (not the current session).

Rules:
- Keep ONLY facts about the user: name, role, expertise, domain, \
preferences, recurring patterns, communication style.
- DISCARD any session-specific context: dataset descriptions, \
current analysis findings, intermediate results, data schemas.
- Remove duplicates, outdated info, vague remarks.
- Prefer bullet-point markdown (- …).
- Maximum 30 concise bullets.
- Write in the language of the notes (keep it consistent).
- Do NOT include any preamble or commentary — output ONLY the merged bullets.
"""


@dataclass
class UserMemory:
    profile: str
    notes: str

    def is_empty(self) -> bool:
        return not self.profile.strip() and not self.notes.strip()

    def build_block(self) -> str:
        """Return markdown block to inject into system prompt."""
        parts: list[str] = []
        if self.profile.strip():
            parts.append(f"### User profile\n{self.profile.strip()}")
        if self.notes.strip():
            parts.append(f"### Agent notes (from past sessions)\n{self.notes.strip()}")
        if not parts:
            return ""
        return "## User memory\n" + "\n\n".join(parts)


class UserMemoryService:
    """Thin wrapper around `AuthDB` for user memory read/write."""

    def __init__(self, db: "AuthDB") -> None:
        self._db = db

    # ── Read ────────────────────────────────────────────────────────────────

    def load(self, user_id: int) -> UserMemory:
        return UserMemory(
            profile=self._db.get_user_memory(user_id, MEM_PROFILE),
            notes=self._db.get_user_memory(user_id, MEM_NOTES),
        )

    # ── Write ───────────────────────────────────────────────────────────────

    def set_profile(self, user_id: int, content: str) -> None:
        self._db.set_user_memory(user_id, MEM_PROFILE, content.strip())

    def set_notes(self, user_id: int, content: str) -> None:
        self._db.set_user_memory(user_id, MEM_NOTES, content.strip())

    def append_note(self, user_id: int, note: str) -> None:
        """Append a raw note line; consolidation happens asynchronously later."""
        existing = self._db.get_user_memory(user_id, MEM_NOTES)
        combined = (existing + "\n- " + note.strip()).lstrip()
        self._db.set_user_memory(user_id, MEM_NOTES, combined)

    # ── LLM consolidation ───────────────────────────────────────────────────

    def schedule_consolidation(
        self,
        user_id: int,
        new_notes: list[str],
        llm_invoke,  # callable(messages) -> AIMessage
    ) -> None:
        """Fire-and-forget: merge new_notes into stored notes via LLM.

        Must be called from an async context (uses asyncio.create_task).
        """
        if not new_notes:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("user_memory: no running event loop, skipping consolidation")
            return
        loop.create_task(
            self._consolidate_async(user_id, new_notes, llm_invoke),
            name=f"memory_consolidate_u{user_id}",
        )

    async def _consolidate_async(
        self,
        user_id: int,
        new_notes: list[str],
        llm_invoke,
    ) -> None:
        try:
            existing = self._db.get_user_memory(user_id, MEM_NOTES)
            new_block = "\n".join(f"- {n.strip()}" for n in new_notes)
            user_msg = (
                f"EXISTING:\n{existing or '(empty)'}\n\n"
                f"NEW:\n{new_block}"
            )
            result = await asyncio.to_thread(
                llm_invoke,
                [
                    {"role": "system", "content": _CONSOLIDATE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
            )
            merged = _extract_text(result)
            if merged.strip():
                self._db.set_user_memory(user_id, MEM_NOTES, merged.strip())
        except Exception:  # noqa: BLE001  # broad catch needed: any LLM or DB error must not crash the background task
            logger.exception("user_memory: consolidation failed for user %d", user_id)


def _extract_text(result) -> str:
    if hasattr(result, "content"):
        c = result.content
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in c
            )
    return str(result)


