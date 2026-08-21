"""Persistence for SessionManifest — the lightweight session identity + sources.

Stored as ``sessions/{session_id}/manifest.json`` alongside notebook.ipynb.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from psycopg.types.json import Jsonb

from backend.auth.app_data_postgres import AppDataPostgresStore
from backend.notebook.models import utcnow_iso
from backend.notebook.session_source import SessionManifest

logger = logging.getLogger(__name__)


class ManifestStore:
    """File-based persistence for SessionManifest objects."""

    def __init__(self, storage_dir: str | Path) -> None:
        self._storage_dir = Path(storage_dir)
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def exists(self, session_id: str) -> bool:
        return self._manifest_path(session_id).is_file()

    def load(self, session_id: str) -> SessionManifest:
        """Load manifest from disk.  Returns empty manifest if missing."""
        path = self._manifest_path(session_id)
        if not path.is_file():
            return SessionManifest(
                session_id=session_id,
                created_at=utcnow_iso(),
                last_access=utcnow_iso(),
            )

        with self._session_lock(session_id):
            raw = json.loads(path.read_text(encoding="utf-8"))
            return SessionManifest.from_dict(raw)

    def save(self, session_id: str, manifest: SessionManifest) -> Path:
        """Persist manifest to disk."""
        path = self._manifest_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        manifest.last_access = utcnow_iso()

        with self._session_lock(session_id):
            payload = json.dumps(
                manifest.to_dict(),
                ensure_ascii=False,
                indent=2,
            )
            path.write_text(payload, encoding="utf-8")

        return path

    def touch(self, session_id: str) -> None:
        """Update last_access timestamp."""
        manifest = self.load(session_id)
        manifest.last_access = utcnow_iso()
        self.save(session_id, manifest)

    def delete(self, session_id: str) -> None:
        path = self._manifest_path(session_id)
        if path.is_file():
            path.unlink()

    # ── Internals ────────────────────────────────────────────────────────

    def _manifest_path(self, session_id: str) -> Path:
        return self._storage_dir / "sessions" / session_id / "manifest.json"

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self._global_lock:
            if session_id not in self._locks:
                self._locks[session_id] = threading.Lock()
            return self._locks[session_id]


class PostgresManifestStore(ManifestStore):
    def __init__(self, store: AppDataPostgresStore) -> None:
        self._store = store
        self._locks = {}
        self._global_lock = threading.Lock()

    def exists(self, session_id: str) -> bool:
        with self._store.connect() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM session_manifests WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                is not None
            )

    def load(self, session_id: str) -> SessionManifest:
        with self._store.connect() as connection:
            row = connection.execute(
                "SELECT payload FROM session_manifests WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return SessionManifest(
                session_id=session_id,
                created_at=utcnow_iso(),
                last_access=utcnow_iso(),
            )
        return SessionManifest.from_dict(dict(row["payload"]))

    def save(self, session_id: str, manifest: SessionManifest) -> Path:
        manifest.last_access = utcnow_iso()
        with self._session_lock(session_id), self._store.connect() as connection:
            connection.execute(
                """
                INSERT INTO session_manifests(session_id, payload, version, updated_at)
                VALUES (?, ?, 1, now())
                ON CONFLICT(session_id) DO UPDATE SET
                    payload = excluded.payload,
                    version = session_manifests.version + 1,
                    updated_at = now()
                """,
                (session_id, Jsonb(manifest.to_dict())),
            )
        return Path(session_id) / "manifest.json"

    def delete(self, session_id: str) -> None:
        with self._store.connect() as connection:
            connection.execute(
                "DELETE FROM session_manifests WHERE session_id = ?",
                (session_id,),
            )
