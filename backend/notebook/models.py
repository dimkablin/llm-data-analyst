"""Notebook-first session model.

One chat session = one notebook.  The notebook is the canonical, persisted
representation of all analytical work.  The kernel (SessionSandbox) is an
ephemeral execution cache that can be evicted and restored by replaying
notebook cells.

The on-disk format is .ipynb-compatible JSON so that any session can later
be opened in Jupyter.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


# ── Cell metadata (LLM-to-LLM breadcrumbs) ──────────────────────────────────


@dataclass
class CellMetadata:
    """Structured context attached to every notebook cell.

    Serves two audiences:
      1. Future LLM passes that need to understand the cell's role.
      2. The restore pipeline that replays cells after kernel eviction.

    Fields are also rendered as ``# PURPOSE: …`` comments inside the cell
    source so they survive a round-trip through plain-text editors.
    """

    purpose: str = ""
    """One-sentence description of what the cell does."""

    produces: list[str] = field(default_factory=list)
    """Variable names this cell creates or overwrites."""

    depends_on: list[str] = field(default_factory=list)
    """Variable names or source aliases this cell reads."""

    source_alias: str | None = None
    """Which SessionSource this cell is bound to (source_binding cells)."""

    tool_name: str = ""
    """Tool that generated this cell (pandas_tool, plotly_tool, …)."""

    idempotent: bool = True
    """Whether the cell is safe to re-run without side-effects."""

    created_by: Literal["system", "llm", "user"] = "llm"
    """Who created this cell."""

    created_at: str = ""
    """ISO-8601 timestamp."""

    language: Literal["python", "sql"] = "python"
    """Code language."""

    question: str = ""
    """Natural-language question (used by sql_tool)."""

    artifact_refs: list[str] = field(default_factory=list)
    """IDs of artifacts produced by this cell."""

    tags: list[str] = field(default_factory=list)
    """Section markers: preamble, source_binding, analysis, visualization."""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.purpose:
            d["purpose"] = self.purpose
        if self.produces:
            d["produces"] = self.produces
        if self.depends_on:
            d["depends_on"] = self.depends_on
        if self.source_alias is not None:
            d["source_alias"] = self.source_alias
        if self.tool_name:
            d["tool_name"] = self.tool_name
        if not self.idempotent:
            d["idempotent"] = False
        if self.created_by != "llm":
            d["created_by"] = self.created_by
        if self.created_at:
            d["created_at"] = self.created_at
        if self.language != "python":
            d["language"] = self.language
        if self.question:
            d["question"] = self.question
        if self.artifact_refs:
            d["artifact_refs"] = self.artifact_refs
        if self.tags:
            d["tags"] = self.tags
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CellMetadata:
        return cls(
            purpose=raw.get("purpose", ""),
            produces=raw.get("produces", []),
            depends_on=raw.get("depends_on", []),
            source_alias=raw.get("source_alias"),
            tool_name=raw.get("tool_name", ""),
            idempotent=raw.get("idempotent", True),
            created_by=raw.get("created_by", "llm"),
            created_at=raw.get("created_at", ""),
            language=raw.get("language", "python"),
            question=raw.get("question", ""),
            artifact_refs=raw.get("artifact_refs", []),
            tags=raw.get("tags", []),
        )


# ── Cell output (ipynb-compatible) ───────────────────────────────────────────


@dataclass
class CellOutput:
    """Single output block attached to a code cell after execution.

    Follows the Jupyter nbformat v4 output schema so that .ipynb files
    produced by this system open correctly in Jupyter.
    """

    output_type: Literal[
        "execute_result", "display_data", "stream", "error"
    ] = "execute_result"

    data: dict[str, Any] = field(default_factory=dict)
    """MIME-keyed payload, e.g. {"text/plain": "DataFrame(5, 3)"}."""

    metadata: dict[str, Any] = field(default_factory=dict)

    execution_count: int | None = None

    # Error-specific fields (populated only when output_type == "error").
    ename: str = ""
    evalue: str = ""
    traceback: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"output_type": self.output_type}
        if self.output_type == "error":
            d["ename"] = self.ename
            d["evalue"] = self.evalue
            d["traceback"] = self.traceback
        else:
            d["data"] = self.data
            d["metadata"] = self.metadata
            if self.execution_count is not None:
                d["execution_count"] = self.execution_count
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CellOutput:
        return cls(
            output_type=raw.get("output_type", "execute_result"),
            data=raw.get("data", {}),
            metadata=raw.get("metadata", {}),
            execution_count=raw.get("execution_count"),
            ename=raw.get("ename", ""),
            evalue=raw.get("evalue", ""),
            traceback=raw.get("traceback", []),
        )


# ── Notebook cell ────────────────────────────────────────────────────────────


def _new_cell_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class NotebookCell:
    """A single cell in the notebook.

    ``cell_type`` is either ``code`` or ``markdown``.  There is no special
    ``source_binding`` type — source-binding cells are regular ``code`` cells
    tagged with ``["source_binding"]`` in metadata so the notebook stays
    valid ipynb.
    """

    id: str = field(default_factory=_new_cell_id)
    cell_type: Literal["code", "markdown"] = "code"
    source: str = ""
    metadata: CellMetadata = field(default_factory=CellMetadata)
    outputs: list[CellOutput] = field(default_factory=list)
    execution_count: int | None = None

    # ── ipynb serialisation ──────────────────────────────────────────────

    def to_ipynb_dict(self) -> dict[str, Any]:
        """Serialise to a Jupyter-compatible cell dict."""
        cell: dict[str, Any] = {
            "id": self.id,
            "cell_type": self.cell_type,
            # ipynb stores source as list of lines.
            "source": _str_to_source_lines(self.source),
            "metadata": self.metadata.to_dict(),
        }
        if self.cell_type == "code":
            cell["outputs"] = [o.to_dict() for o in self.outputs]
            cell["execution_count"] = self.execution_count
        return cell

    @classmethod
    def from_ipynb_dict(cls, raw: dict[str, Any]) -> NotebookCell:
        source_raw = raw.get("source", "")
        if isinstance(source_raw, list):
            source = "".join(source_raw)
        else:
            source = str(source_raw)

        outputs_raw = raw.get("outputs", [])
        outputs = [CellOutput.from_dict(o) for o in outputs_raw]

        return cls(
            id=raw.get("id", _new_cell_id()),
            cell_type=raw.get("cell_type", "code"),
            source=source,
            metadata=CellMetadata.from_dict(raw.get("metadata", {})),
            outputs=outputs,
            execution_count=raw.get("execution_count"),
        )

    # ── Convenience helpers ──────────────────────────────────────────────

    @property
    def is_source_binding(self) -> bool:
        return "source_binding" in self.metadata.tags

    def set_output_text(self, text: str, *, execution_count: int | None = None) -> None:
        """Replace outputs with a single text/plain result."""
        self.outputs = [
            CellOutput(
                output_type="execute_result",
                data={"text/plain": text},
                execution_count=execution_count,
            )
        ]
        self.execution_count = execution_count

    def set_error(self, exc: BaseException, *, execution_count: int | None = None) -> None:
        """Replace outputs with an error block."""
        self.outputs = [
            CellOutput(
                output_type="error",
                ename=type(exc).__name__,
                evalue=str(exc),
            )
        ]
        self.execution_count = execution_count


# ── Notebook document ────────────────────────────────────────────────────────


_KERNEL_SPEC: dict[str, str] = {
    "name": "analyst_kernel",
    "display_name": "AI Analyst Kernel",
    "language": "python",
}

_LANGUAGE_INFO: dict[str, str] = {
    "name": "python",
    "version": "3.11",
}


@dataclass
class NotebookDocument:
    """Top-level notebook — the canonical session representation.

    Structurally compatible with Jupyter nbformat v4.5 so the file can be
    opened in Jupyter Server without conversion.
    """

    session_id: str = ""
    created_at: str = ""
    cells: list[NotebookCell] = field(default_factory=list)

    # ── ipynb serialisation ──────────────────────────────────────────────

    def to_ipynb_dict(self) -> dict[str, Any]:
        return {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": dict(_KERNEL_SPEC),
                "language_info": dict(_LANGUAGE_INFO),
                "session_id": self.session_id,
                "created_at": self.created_at,
            },
            "cells": [c.to_ipynb_dict() for c in self.cells],
        }

    @classmethod
    def from_ipynb_dict(cls, raw: dict[str, Any]) -> NotebookDocument:
        meta = raw.get("metadata", {})
        cells_raw = raw.get("cells", [])
        return cls(
            session_id=meta.get("session_id", ""),
            created_at=meta.get("created_at", ""),
            cells=[NotebookCell.from_ipynb_dict(c) for c in cells_raw],
        )

    # ── Cell access helpers ──────────────────────────────────────────────

    def cell_by_id(self, cell_id: str) -> NotebookCell | None:
        return next((c for c in self.cells if c.id == cell_id), None)

    def cell_index(self, cell_id: str) -> int | None:
        for i, c in enumerate(self.cells):
            if c.id == cell_id:
                return i
        return None

    @property
    def source_binding_cells(self) -> list[NotebookCell]:
        return [c for c in self.cells if c.is_source_binding]

    @property
    def code_cells(self) -> list[NotebookCell]:
        return [c for c in self.cells if c.cell_type == "code"]

    @property
    def next_execution_count(self) -> int:
        counts = [c.execution_count for c in self.cells if c.execution_count is not None]
        return (max(counts) + 1) if counts else 1

    def append_cell(self, cell: NotebookCell) -> None:
        self.cells.append(cell)

    def insert_cell(self, position: int, cell: NotebookCell) -> None:
        self.cells.insert(position, cell)

    def remove_cell(self, cell_id: str) -> NotebookCell | None:
        idx = self.cell_index(cell_id)
        if idx is None:
            return None
        return self.cells.pop(idx)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _str_to_source_lines(source: str) -> list[str]:
    """Convert a source string to ipynb line-list format.

    Each line except the last ends with ``\\n``.
    """
    if not source:
        return []
    lines = source.splitlines(True)
    # Ensure every line except possibly the last ends with newline.
    if lines and not lines[-1].endswith("\n"):
        pass  # last line without trailing newline is valid ipynb
    return lines


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
