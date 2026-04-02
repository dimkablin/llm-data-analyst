"""Execution artifacts — typed, session-scoped intermediate results of tool execution.

Execution artifacts represent internal data produced during the analytics pipeline:
dataframes, scalar computations, chart figures, SQL query results.  They carry
full lineage metadata (producer tool, parent references, schema, content hash)
and are never sent to the UI directly.

The lifecycle is:
    tool execution  →  ExecutionArtifact  →  ExecutionStore (session-scoped)
                                                    ↓
                                          PresentationArtifact (for UI)
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ExecArtifactType(str, Enum):
    DATAFRAME = "dataframe"
    SCALAR = "scalar"
    PLOT = "plot"
    SQL_RESULT = "sql_result"
    SEARCH_RESULT = "search_result"
    FORECAST = "forecast"


def artifact_type_label(artifact_type_val: Any) -> str:
    """Return the string label for an artifact_type field (enum value or raw string)."""
    return str(
        artifact_type_val.value if isinstance(artifact_type_val, ExecArtifactType) else artifact_type_val
    ).strip().lower()


@dataclass(frozen=False)
class ExecArtifactSchema:
    """Lightweight schema descriptor for tabular artifacts."""
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    row_count: int = 0


@dataclass
class ExecutionArtifact:
    """A single typed result produced by a tool during agent execution.

    Attributes:
        id:             Stable UUID.
        session_id:     Owning session.
        artifact_type:  Discriminator (dataframe, scalar, plot, …).
        producer_tool:  Name of the tool that created this artifact.
        data:           Raw Python object (DataFrame, Figure, dict, scalar).
        name:           Human-readable short label (e.g. "revenue_by_month").
        parent_ids:     IDs of artifacts this one was derived from.
        schema:         Column metadata for tabular types.
        content_hash:   SHA-256 of a deterministic representation of *data*.
        meta:           Freeform metadata (recipe steps, provenance, hints).
        created_at:     UTC timestamp.
        version:        Monotonic version within the same logical name.
        reusable:       Whether this artifact can be served from cache.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    artifact_type: ExecArtifactType = ExecArtifactType.DATAFRAME
    producer_tool: str = ""
    data: Any = None
    name: str = ""
    parent_ids: list[str] = field(default_factory=list)
    schema: ExecArtifactSchema | None = None
    content_hash: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    version: int = 1
    reusable: bool = True

    def compute_content_hash(self) -> str:
        """Compute and cache a content hash based on artifact type."""
        try:
            raw = self._hash_input()
            self.content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        except Exception:
            self.content_hash = ""
        return self.content_hash

    def _hash_input(self) -> str:
        if self.artifact_type == ExecArtifactType.DATAFRAME:
            import pandas as pd
            if isinstance(self.data, pd.DataFrame):
                return pd.util.hash_pandas_object(self.data).sum().__str__()
        if self.artifact_type == ExecArtifactType.SCALAR:
            return json.dumps(self.data, sort_keys=True, default=str)
        if self.artifact_type == ExecArtifactType.PLOT:
            fig = self.data
            # Avoid expensive full serialization; use structural fingerprint instead.
            traces = len(getattr(fig, "data", ())) if hasattr(fig, "data") else 0
            layout_title = str(getattr(getattr(fig, "layout", None), "title", ""))
            return f"plot:{type(fig).__name__}:{traces}:{layout_title}"
        return str(self.data)

    def build_schema(self) -> ExecArtifactSchema | None:
        """Extract schema from data if it's a DataFrame."""
        if self.artifact_type != ExecArtifactType.DATAFRAME:
            return None
        import pandas as pd
        if not isinstance(self.data, pd.DataFrame):
            return None
        self.schema = ExecArtifactSchema(
            columns=list(self.data.columns),
            dtypes={col: str(dtype) for col, dtype in self.data.dtypes.items()},
            row_count=len(self.data),
        )
        return self.schema


class ExecutionStore:
    """Session-scoped registry of execution artifacts with lineage tracking.

    Provides:
    - Put / get by ID or by (name, producer_tool)
    - Content-hash deduplication (reuse instead of recompute)
    - Lineage queries: parents, children, full ancestry chain
    - Deterministic reuse rules
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._by_id: dict[str, ExecutionArtifact] = {}
        self._by_name: dict[str, list[ExecutionArtifact]] = {}
        self._by_hash: dict[str, ExecutionArtifact] = {}

    def put(self, artifact: ExecutionArtifact) -> ExecutionArtifact:
        """Register an execution artifact.  Deduplicates by content_hash."""
        artifact.session_id = self.session_id

        if artifact.artifact_type == ExecArtifactType.DATAFRAME:
            artifact.build_schema()
        artifact.compute_content_hash()

        # Content-hash dedup: return existing if identical
        if artifact.content_hash and artifact.content_hash in self._by_hash:
            existing = self._by_hash[artifact.content_hash]
            if (
                existing.reusable
                and existing.producer_tool == artifact.producer_tool
                and existing.name == artifact.name
            ):
                return existing

        # Version bump for same logical name
        existing_versions = self._by_name.get(artifact.name, [])
        if existing_versions:
            artifact.version = max(a.version for a in existing_versions) + 1

        self._by_id[artifact.id] = artifact
        self._by_name.setdefault(artifact.name, []).append(artifact)
        if artifact.content_hash:
            self._by_hash[artifact.content_hash] = artifact
        return artifact

    def get(self, artifact_id: str) -> ExecutionArtifact | None:
        return self._by_id.get(artifact_id)

    def get_latest(self, name: str) -> ExecutionArtifact | None:
        """Return the latest version of a named artifact."""
        versions = self._by_name.get(name, [])
        return versions[-1] if versions else None

    def get_by_tool(self, tool_name: str) -> list[ExecutionArtifact]:
        return [a for a in self._by_id.values() if a.producer_tool == tool_name]

    def get_parents(self, artifact_id: str) -> list[ExecutionArtifact]:
        artifact = self._by_id.get(artifact_id)
        if not artifact:
            return []
        return [self._by_id[pid] for pid in artifact.parent_ids if pid in self._by_id]

    def get_children(self, artifact_id: str) -> list[ExecutionArtifact]:
        return [a for a in self._by_id.values() if artifact_id in a.parent_ids]

    def get_lineage(self, artifact_id: str) -> list[ExecutionArtifact]:
        """Return full ancestry chain (parents, grandparents, …) in topological order."""
        visited: set[str] = set()
        chain: list[ExecutionArtifact] = []

        def _walk(aid: str) -> None:
            if aid in visited:
                return
            visited.add(aid)
            artifact = self._by_id.get(aid)
            if artifact is None:
                return
            for pid in artifact.parent_ids:
                _walk(pid)
            chain.append(artifact)

        _walk(artifact_id)
        return chain

    def all(self) -> list[ExecutionArtifact]:
        return list(self._by_id.values())

    def clear(self) -> None:
        self._by_id.clear()
        self._by_name.clear()
        self._by_hash.clear()
