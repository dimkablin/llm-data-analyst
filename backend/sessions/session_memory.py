"""Session memory — per-session context notes stored alongside session state.

Session memory captures observations about the *current analysis session*:
what data is loaded, key findings, intermediate conclusions, data-quality
notes, etc.  Unlike user memory (which persists across sessions in SQLite),
session memory lives in ``state.json`` inside the session directory and is
scoped to a single chat session.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SessionMemory:
    notes: str = ""

    def is_empty(self) -> bool:
        return not self.notes.strip()

    def build_block(self) -> str:
        """Return markdown block to inject into system prompt."""
        if not self.notes.strip():
            return ""
        return (
            "## Session context\n"
            "### Notes from this session\n"
            f"{self.notes.strip()}"
        )
