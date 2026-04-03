"""Execution graph tracker for real-time DAG visualization.

Builds a lightweight graph of phase → tool execution flow,
emitted via SSE as ``execution_graph`` events.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphNode:
    id: str
    type: str  # "phase" | "tool"
    label: str
    status: str = "pending"  # "pending" | "running" | "done" | "error"
    duration_ms: int | None = None
    tool_name: str | None = None
    artifact_keys: list[str] = field(default_factory=list)
    parent_id: str | None = None  # e.g. tool nodes belong to an act phase

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "status": self.status,
        }
        if self.duration_ms is not None:
            d["duration_ms"] = self.duration_ms
        if self.tool_name:
            d["tool_name"] = self.tool_name
        if self.artifact_keys:
            d["artifact_keys"] = self.artifact_keys
        if self.parent_id:
            d["parent_id"] = self.parent_id
        return d


@dataclass
class GraphEdge:
    from_id: str
    to_id: str
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"from": self.from_id, "to": self.to_id}
        if self.label:
            d["label"] = self.label
        return d


class ExecutionGraphTracker:
    """Tracks execution flow as a directed graph."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        self._last_phase_id: str | None = None
        self._last_tool_id: str | None = None
        self._current_act_id: str | None = None
        self._started_at: dict[str, float] = {}
        self._running_by_name: dict[str, str] = {}  # tool_name → node_id

    # ── Phase tracking ───────────────────────────────────────────────

    def phase_start(self, phase: str, step_index: int) -> str:
        """Register a phase node as running. Returns the node ID."""
        node_id = f"{phase}-{step_index}"
        node = GraphNode(
            id=node_id,
            type="phase",
            label=self._phase_label(phase),
            status="running",
        )
        self._nodes[node_id] = node
        self._started_at[node_id] = time.perf_counter()

        # Connect to previous phase.
        if self._last_phase_id and self._last_phase_id != node_id:
            self._edges.append(GraphEdge(self._last_phase_id, node_id))

        self._last_phase_id = node_id
        if phase == "act":
            self._current_act_id = node_id
            self._last_tool_id = None

        return node_id

    def phase_end(self, phase: str, step_index: int, status: str = "done") -> None:
        node_id = f"{phase}-{step_index}"
        node = self._nodes.get(node_id)
        if node:
            node.status = status
            start = self._started_at.get(node_id)
            if start:
                node.duration_ms = int((time.perf_counter() - start) * 1000)

    # ── Tool tracking ────────────────────────────────────────────────

    def tool_start(self, tool_name: str, step_index: int) -> str:
        """Register a tool node as running. Returns the node ID."""
        node_id = f"tool:{tool_name}-{step_index}-{len(self._nodes)}"
        node = GraphNode(
            id=node_id,
            type="tool",
            label=tool_name,
            status="running",
            tool_name=tool_name,
            parent_id=self._current_act_id,
        )
        self._nodes[node_id] = node
        self._started_at[node_id] = time.perf_counter()
        self._running_by_name[tool_name] = node_id

        # Connect: act → first tool, or tool → tool.
        if self._last_tool_id:
            self._edges.append(GraphEdge(self._last_tool_id, node_id))
        elif self._current_act_id:
            self._edges.append(GraphEdge(self._current_act_id, node_id))

        self._last_tool_id = node_id
        return node_id

    def tool_end(
        self,
        tool_name: str,
        step_index: int,
        status: str = "done",
        artifact_keys: list[str] | None = None,
    ) -> None:
        nid = self._running_by_name.pop(tool_name, None)
        node = self._nodes.get(nid) if nid else None
        if not node:
            return
        node.status = status
        if artifact_keys:
            node.artifact_keys = artifact_keys
        start = self._started_at.get(node.id)
        if start:
            node.duration_ms = int((time.perf_counter() - start) * 1000)

    # ── Data flow tracking ───────────────────────────────────────────

    def add_data_flow(self, from_tool_id: str, to_tool_id: str, var_name: str) -> None:
        self._edges.append(GraphEdge(from_tool_id, to_tool_id, label=var_name))

    # ── Snapshot ─────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return the current graph state as a JSON-serializable dict."""
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }

    def __bool__(self) -> bool:
        return bool(self._nodes)

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _phase_label(phase: str) -> str:
        labels = {
            "think": "Plan",
            "act": "Execute",
            "evaluate": "Evaluate",
            "decide": "Decide",
            "finalize": "Finalize",
        }
        return labels.get(phase, phase.capitalize())
