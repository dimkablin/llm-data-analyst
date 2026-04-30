from __future__ import annotations

from dataclasses import dataclass

from backend.agent_graph.state import ArtifactRefState, WorkingMemoryState
from backend.sessions.session_memory import SessionArtifactRef, StructuredSessionMemory


INFRA_TOOLS: frozenset[str] = frozenset({
    "database_tool",
    "get_tool_instructions",
    "planner_tool",
})


def extract_findings_from_actions(actions: list[str], turn_index: int) -> list[str]:
    findings: list[str] = []
    for action in actions:
        tool = action.split("->")[0].strip()
        if any(infra in tool for infra in INFRA_TOOLS):
            continue
        findings.append(f"[turn {turn_index}] {action}")
    return findings


@dataclass(slots=True)
class WorkingMemoryFlusher:
    """Flushes graph working memory into persistent structured session memory."""

    structured_memory: StructuredSessionMemory

    def flush(self, working_memory: WorkingMemoryState) -> None:
        existing_ids = {ref.id for ref in self.structured_memory.artifact_index}
        for artifact in working_memory.get("artifact_refs", []):
            artifact_id = str(artifact.get("id") or "")
            if not artifact_id or artifact_id in existing_ids:
                continue
            self.structured_memory.artifact_index.append(
                self._artifact_ref(artifact),
            )
            existing_ids.add(artifact_id)

        self.structured_memory.artifact_index = self.structured_memory.artifact_index[-100:]
        new_findings = extract_findings_from_actions(
            list(working_memory.get("completed_actions") or []),
            turn_index=self.structured_memory.turn_count,
        )
        self.structured_memory.key_findings = (
            self.structured_memory.key_findings + new_findings
        )[-30:]
        self.structured_memory.turn_count += 1

    def _artifact_ref(self, artifact: ArtifactRefState) -> SessionArtifactRef:
        return SessionArtifactRef(
            id=str(artifact.get("id") or ""),
            name=str(artifact.get("name") or ""),
            type=str(artifact.get("type") or ""),
            turn_index=self.structured_memory.turn_count,
            schema=artifact.get("schema"),
            row_count=artifact.get("row_count"),
            summary=artifact.get("summary"),
        )
