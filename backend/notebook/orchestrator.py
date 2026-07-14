"""Notebook edit protocol — the single entry point for all notebook mutations.

The LLM agent and tools never mutate notebook cells directly.  All changes
flow through ``NotebookOrchestrator.apply()`` which validates the operation
and persists the result atomically.

Supported operations::

    INSERT   — add a cell at a given position (or append)
    UPDATE   — replace a cell's source / metadata by cell_id
    DELETE   — remove a cell by cell_id
    MOVE     — reorder a cell to a new position
    EXECUTE  — mark a cell as executed and attach outputs

Each operation returns the mutated ``NotebookDocument`` so callers always
have the latest state.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from backend.notebook.models import (
    CellOutput,
    NotebookCell,
    NotebookDocument,
)
from backend.notebook.store import NotebookStore

logger = logging.getLogger(__name__)


# ── Edit operation types ─────────────────────────────────────────────────────


class CellOp(StrEnum):
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    MOVE = "move"
    EXECUTE = "execute"


@dataclass
class NotebookEdit:
    """A single atomic notebook mutation."""

    op: CellOp
    cell_id: str | None = None
    position: int | None = None
    cell: NotebookCell | None = None
    outputs: list[CellOutput] | None = None
    execution_count: int | None = None


@dataclass
class EditResult:
    """Outcome of applying an edit."""

    ok: bool
    notebook: NotebookDocument
    cell_id: str | None = None
    error: str | None = None


# ── Orchestrator ─────────────────────────────────────────────────────────────


class NotebookOrchestrator:
    """Validates and applies notebook edits, then persists the result."""

    def __init__(self, store: NotebookStore) -> None:
        self._store = store

    def apply(self, session_id: str, edit: NotebookEdit) -> EditResult:
        """Validate *edit*, apply it to the notebook, persist, and return."""
        notebook = self._store.load(session_id)
        result = self._apply_to_document(notebook, edit)
        if result.ok:
            self._store.save(session_id, notebook)
        return result

    def apply_batch(
        self, session_id: str, edits: list[NotebookEdit]
    ) -> list[EditResult]:
        """Apply multiple edits in order.  Stops on first failure."""
        notebook = self._store.load(session_id)
        results: list[EditResult] = []
        for edit in edits:
            result = self._apply_to_document(notebook, edit)
            results.append(result)
            if not result.ok:
                break
        if all(r.ok for r in results):
            self._store.save(session_id, notebook)
        return results

    def remove_source_binding(self, session_id: str, source_alias: str) -> EditResult:
        """Remove the source_binding cell owned by a removed SessionSource."""
        clean_alias = str(source_alias or "").strip()
        notebook = self._store.load(session_id)
        if not clean_alias:
            return EditResult(
                ok=False,
                notebook=notebook,
                error="source_alias is required",
            )

        target = next(
            (
                cell
                for cell in notebook.source_binding_cells
                if cell.metadata.source_alias == clean_alias
            ),
            None,
        )
        if target is None:
            return EditResult(ok=True, notebook=notebook, cell_id=None)

        notebook.remove_cell(target.id)
        self._store.save(session_id, notebook)
        return EditResult(ok=True, notebook=notebook, cell_id=target.id)

    # ── Dispatch ─────────────────────────────────────────────────────────

    def _apply_to_document(
        self, notebook: NotebookDocument, edit: NotebookEdit
    ) -> EditResult:
        try:
            if edit.op == CellOp.INSERT:
                return self._do_insert(notebook, edit)
            if edit.op == CellOp.UPDATE:
                return self._do_update(notebook, edit)
            if edit.op == CellOp.DELETE:
                return self._do_delete(notebook, edit)
            if edit.op == CellOp.MOVE:
                return self._do_move(notebook, edit)
            if edit.op == CellOp.EXECUTE:
                return self._do_execute(notebook, edit)
            return EditResult(ok=False, notebook=notebook, error=f"Unknown op: {edit.op}")
        except Exception as exc:
            logger.exception("Notebook edit failed: %s", edit.op)
            return EditResult(ok=False, notebook=notebook, error=str(exc))

    # ── Operations ───────────────────────────────────────────────────────

    def _do_insert(self, nb: NotebookDocument, edit: NotebookEdit) -> EditResult:
        if edit.cell is None:
            return EditResult(ok=False, notebook=nb, error="INSERT requires cell")

        cell = edit.cell
        if nb.cell_by_id(cell.id) is not None:
            return EditResult(ok=False, notebook=nb, error=f"Duplicate cell id: {cell.id}")

        if edit.position is not None:
            pos = max(0, min(edit.position, len(nb.cells)))
            nb.insert_cell(pos, cell)
        else:
            nb.append_cell(cell)

        return EditResult(ok=True, notebook=nb, cell_id=cell.id)

    def _do_update(self, nb: NotebookDocument, edit: NotebookEdit) -> EditResult:
        if not edit.cell_id:
            return EditResult(ok=False, notebook=nb, error="UPDATE requires cell_id")

        idx = nb.cell_index(edit.cell_id)
        if idx is None:
            return EditResult(ok=False, notebook=nb, error=f"Cell not found: {edit.cell_id}")

        existing = nb.cells[idx]

        if edit.cell is not None:
            # Full replacement — preserve id and type.
            new_cell = edit.cell
            new_cell.id = existing.id
            nb.cells[idx] = new_cell
        else:
            return EditResult(ok=False, notebook=nb, error="UPDATE requires cell data")

        return EditResult(ok=True, notebook=nb, cell_id=edit.cell_id)

    def _do_delete(self, nb: NotebookDocument, edit: NotebookEdit) -> EditResult:
        if not edit.cell_id:
            return EditResult(ok=False, notebook=nb, error="DELETE requires cell_id")

        target = nb.cell_by_id(edit.cell_id)
        if target is None:
            return EditResult(ok=False, notebook=nb, error=f"Cell not found: {edit.cell_id}")

        # Guard: source_binding cells cannot be deleted via normal edits.
        if target.is_source_binding:
            return EditResult(
                ok=False,
                notebook=nb,
                error="Cannot delete source_binding cell — remove the source instead",
            )

        nb.remove_cell(edit.cell_id)
        return EditResult(ok=True, notebook=nb, cell_id=edit.cell_id)

    def _do_move(self, nb: NotebookDocument, edit: NotebookEdit) -> EditResult:
        if not edit.cell_id:
            return EditResult(ok=False, notebook=nb, error="MOVE requires cell_id")
        if edit.position is None:
            return EditResult(ok=False, notebook=nb, error="MOVE requires position")

        cell = nb.remove_cell(edit.cell_id)
        if cell is None:
            return EditResult(ok=False, notebook=nb, error=f"Cell not found: {edit.cell_id}")

        pos = max(0, min(edit.position, len(nb.cells)))
        nb.insert_cell(pos, cell)
        return EditResult(ok=True, notebook=nb, cell_id=edit.cell_id)

    def _do_execute(self, nb: NotebookDocument, edit: NotebookEdit) -> EditResult:
        """Record execution results on a cell (outputs + execution_count).

        Actual code execution happens in the kernel — this operation only
        updates the notebook document with the results.
        """
        if not edit.cell_id:
            return EditResult(ok=False, notebook=nb, error="EXECUTE requires cell_id")

        cell = nb.cell_by_id(edit.cell_id)
        if cell is None:
            return EditResult(ok=False, notebook=nb, error=f"Cell not found: {edit.cell_id}")

        if edit.outputs is not None:
            cell.outputs = list(edit.outputs)
        if edit.execution_count is not None:
            cell.execution_count = edit.execution_count

        return EditResult(ok=True, notebook=nb, cell_id=edit.cell_id)
