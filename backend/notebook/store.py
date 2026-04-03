"""Notebook persistence — read/write .ipynb JSON files.

Each session has exactly one notebook stored at::

    {storage_dir}/sessions/{session_id}/notebook.ipynb

The store is intentionally simple: load the full document, mutate it in
memory via ``NotebookDocument`` methods, then save.  There is no partial
write or streaming — notebooks are small JSON files (typically < 1 MB).

Thread safety is handled by per-session locks.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from backend.notebook.models import (
    NotebookCell,
    NotebookDocument,
    utcnow_iso,
)

logger = logging.getLogger(__name__)


class NotebookStore:
    """File-based persistence for NotebookDocument objects."""

    def __init__(self, storage_dir: str | Path) -> None:
        self._storage_dir = Path(storage_dir)
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────────────

    def exists(self, session_id: str) -> bool:
        return self._notebook_path(session_id).is_file()

    def load(self, session_id: str) -> NotebookDocument:
        """Load notebook from disk.  Returns empty document if file missing."""
        path = self._notebook_path(session_id)
        if not path.is_file():
            return NotebookDocument(session_id=session_id, created_at=utcnow_iso())

        with self._session_lock(session_id):
            raw = json.loads(path.read_text(encoding="utf-8"))
            return NotebookDocument.from_ipynb_dict(raw)

    def save(self, session_id: str, notebook: NotebookDocument) -> Path:
        """Persist notebook to disk.  Creates parent directories if needed."""
        path = self._notebook_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._session_lock(session_id):
            payload = json.dumps(
                notebook.to_ipynb_dict(),
                ensure_ascii=False,
                indent=1,
            )
            path.write_text(payload, encoding="utf-8")

        return path

    def append_cell(self, session_id: str, cell: NotebookCell) -> NotebookDocument:
        """Load notebook, append a cell, save, and return the updated doc."""
        notebook = self.load(session_id)
        notebook.append_cell(cell)
        self.save(session_id, notebook)
        return notebook

    def create_empty(self, session_id: str) -> NotebookDocument:
        """Create a new empty notebook with a preamble cell."""
        notebook = NotebookDocument(
            session_id=session_id,
            created_at=utcnow_iso(),
        )
        self.save(session_id, notebook)
        return notebook

    def delete(self, session_id: str) -> None:
        """Remove notebook file from disk."""
        path = self._notebook_path(session_id)
        if path.is_file():
            path.unlink()

    # ── Internals ────────────────────────────────────────────────────────

    def _notebook_path(self, session_id: str) -> Path:
        return self._storage_dir / "sessions" / session_id / "notebook.ipynb"

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self._global_lock:
            if session_id not in self._locks:
                self._locks[session_id] = threading.Lock()
            return self._locks[session_id]
