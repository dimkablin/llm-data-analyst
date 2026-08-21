from __future__ import annotations

import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from psycopg.types.json import Jsonb

from backend.auth.app_data_postgres import AppDataPostgresStore
from backend.core.json_utils import make_json_safe
from backend.sessions.session_store import (
    SessionBusyError,
    SessionQueryLease,
    SessionState,
    SessionStore,
    _state_from_payload,
    _state_payload,
)

_MEMORY_FIELDS = (
    "session_memory",
    "artifact_index_json",
    "key_findings",
    "session_turn_count",
    "context_summary",
    "compacted_message_count",
)


def _json_safe(value: object) -> Any:
    return make_json_safe(value)


class PostgresSessionStore(SessionStore):
    """Persistent session state in app_data; DataFrames stay runtime-local."""

    def __init__(
        self,
        root_dir: str,
        *,
        app_data_store: AppDataPostgresStore,
        data_catalog_store: Any | None = None,
        artifact_blob_store: Any | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._app_data_store = app_data_store
        self._data_catalog_store = data_catalog_store
        self._artifact_blob_store = artifact_blob_store
        self._df_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._locks_guard = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}
        self._query_locks: dict[str, threading.Lock] = {}

    def initialize(self) -> None:
        self._app_data_store.ensure_schema()

    def acquire_query_lease(self, session_id: str) -> SessionQueryLease:
        """Use a PostgreSQL advisory lock so all Uvicorn workers agree."""
        clean_id = str(session_id or "").strip()
        if not clean_id:
            raise ValueError("session_id is required")
        connection = self._app_data_store.connect()
        connection.__enter__()
        lock_key = f"session_query:{clean_id}"
        try:
            row = connection.execute(
                "SELECT pg_try_advisory_lock(hashtextextended(?, 0)) AS acquired",
                (lock_key,),
            ).fetchone()
            if row is None or not bool(row["acquired"]):
                raise SessionBusyError("A request is already running in this session.")
        except Exception:
            connection.__exit__(None, None, None)
            raise

        def release() -> None:
            try:
                connection.execute(
                    "SELECT pg_advisory_unlock(hashtextextended(?, 0))",
                    (lock_key,),
                )
            finally:
                connection.__exit__(None, None, None)

        return SessionQueryLease(release)

    @staticmethod
    def _timestamp_text(value: object) -> str:
        return value.isoformat() if isinstance(value, datetime) else str(value)

    def create_session(self, session_id: str | None = None) -> SessionState:
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("PostgreSQL sessions require a registered session_id")
        self._session_dir(session_id).mkdir(parents=True, exist_ok=True)
        state = SessionState(
            session_id=session_id,
            created_at=self._now_iso(),
            last_access=self._now_iso(),
            chat_history=[],
            artifacts=[],
            selected_skill_ids=[],
            csv_table_names=[],
        )
        self._save_state(state)
        return state

    def _load_state(self, session_id: str) -> SessionState | None:
        with self._app_data_store.connect() as connection:
            row = connection.execute(
                "SELECT payload FROM session_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            messages = connection.execute(
                "SELECT payload FROM session_messages WHERE session_id = ? ORDER BY message_index",
                (session_id,),
            ).fetchall()
            artifacts = connection.execute(
                "SELECT payload FROM session_artifacts WHERE session_id = ? ORDER BY artifact_index",
                (session_id,),
            ).fetchall()
            memory = connection.execute(
                "SELECT payload FROM session_memory WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        payload = dict(row["payload"])
        payload["created_at"] = self._timestamp_text(payload["created_at"])
        payload["last_access"] = self._timestamp_text(payload["last_access"])
        payload["chat_history"] = [dict(item["payload"]) for item in messages]
        payload["artifacts"] = [dict(item["payload"]) for item in artifacts]
        if memory is not None:
            payload.update(dict(memory["payload"]))
        return _state_from_payload(payload)

    def _save_state(self, state: SessionState) -> None:
        payload = _json_safe(_state_payload(state))
        messages = list(payload.pop("chat_history"))
        artifacts = list(payload.pop("artifacts"))
        memory = {field: payload.pop(field) for field in _MEMORY_FIELDS}

        # ponytail: full replacement is enough for demo-sized histories; switch to
        # append-only writes when measured session history size makes this material.
        with self._app_data_store.connect() as connection:
            connection.execute(
                """
                INSERT INTO session_state(session_id, payload, version, updated_at)
                VALUES (?, ?, 1, now())
                ON CONFLICT(session_id) DO UPDATE SET
                    payload = excluded.payload,
                    version = session_state.version + 1,
                    updated_at = now()
                """,
                (state.session_id, Jsonb(payload)),
            )
            connection.execute("DELETE FROM session_messages WHERE session_id = ?", (state.session_id,))
            if messages:
                connection.executemany(
                    """
                    INSERT INTO session_messages(session_id, message_index, role, payload)
                    VALUES (%s, %s, %s, %s)
                    """,
                    [
                        (state.session_id, index, str(message.get("role") or "unknown"), Jsonb(message))
                        for index, message in enumerate(messages)
                    ],
                )
            connection.execute("DELETE FROM session_artifacts WHERE session_id = ?", (state.session_id,))
            if artifacts:
                connection.executemany(
                    """
                    INSERT INTO session_artifacts(artifact_id, session_id, artifact_index, payload)
                    VALUES (%s, %s, %s, %s)
                    """,
                    [
                        (
                            str(artifact.get("id") or f"{state.session_id}:{index}"),
                            state.session_id,
                            index,
                            Jsonb(artifact),
                        )
                        for index, artifact in enumerate(artifacts)
                    ],
                )
            connection.execute(
                """
                INSERT INTO session_memory(session_id, payload, updated_at)
                VALUES (?, ?, now())
                ON CONFLICT(session_id) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = now()
                """,
                (state.session_id, Jsonb(memory)),
            )
