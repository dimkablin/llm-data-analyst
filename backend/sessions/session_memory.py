from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionArtifactRef:
    """Persisted lightweight reference to an artifact from a completed turn."""

    id: str
    name: str
    type: str  # "table" | "plot" | "value" | "error"
    turn_index: int  # which user turn (0-indexed) produced this
    schema: dict[str, str] | None
    row_count: int | None
    summary: str | None
    producer_tool: str | None = None
    parent_ids: list[str] = field(default_factory=list)


@dataclass
class StructuredSessionMemory:
    """
    Cross-turn persistent state. Replaces SessionMemory.
    Backward compatible: notes field preserved, memory_tool continues to write here.
    """

    notes: str = ""
    artifact_index: list[SessionArtifactRef] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    turn_count: int = 0
    context_summary: str = ""
    compacted_message_count: int = 0

    def is_empty(self) -> bool:
        return (
            not self.notes.strip()
            and not self.context_summary.strip()
            and not self.artifact_index
            and not self.key_findings
        )

    def build_block(self, *, include_context_summary: bool = True) -> str:
        """
        Compact context block for system prompt injection.
        Returns artifact refs and findings — NOT inline data.
        """
        parts: list[str] = []
        if include_context_summary and self.context_summary.strip():
            parts.append(f"## Compressed prior conversation\n{self.context_summary.strip()}")
        if self.notes.strip():
            parts.append(f"## Session notes\n{self.notes.strip()}")
        if self.key_findings:
            findings = "\n".join(f"- {f}" for f in self.key_findings[-10:])
            parts.append(f"## Key findings from this session\n{findings}")
        if self.artifact_index:
            lines: list[str] = []
            for a in self.artifact_index[-20:]:
                meta = a.type
                if a.row_count is not None and a.schema:
                    meta += f", {a.row_count}×{len(a.schema)}"
                if a.summary:
                    meta += f", {a.summary}"
                source = f"; source={a.producer_tool}" if a.producer_tool else ""
                parents = f"; parents={','.join(a.parent_ids)}" if a.parent_ids else ""
                lines.append(f"- artifact_id={a.id}; name={a.name} ({meta}){source}{parents}")
            parts.append("## Artifacts from this session\n" + "\n".join(lines))
        return "\n\n".join(parts)


# Backward compat alias — existing code that imports SessionMemory keeps working
SessionMemory = StructuredSessionMemory
