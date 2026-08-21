from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any

from psycopg.types.json import Jsonb

from backend.auth.app_data_postgres import AppDataPostgresStore


@dataclass(frozen=True)
class BlobWrite:
    logical_name: str
    media_type: str
    content: bytes
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StoredBlob:
    blob_id: str
    logical_name: str
    media_type: str
    content: bytes


class PostgresBlobStore:
    """Durable originals and generated files; runtime derivatives stay local."""

    def __init__(self, store: AppDataPostgresStore) -> None:
        self._store = store

    def put_many(
        self,
        *,
        user_id: int,
        session_id: str | None,
        kind: str,
        items: list[BlobWrite],
    ) -> list[str]:
        blob_ids = [str(uuid.uuid4()) for _ in items]
        with self._store.connect() as connection:
            for blob_id, item in zip(blob_ids, items, strict=True):
                connection.execute(
                    """
                    INSERT INTO stored_blobs(
                        id, user_id, session_id, kind, logical_name, media_type,
                        size_bytes, sha256, content, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        blob_id,
                        user_id,
                        session_id,
                        kind,
                        item.logical_name,
                        item.media_type,
                        len(item.content),
                        hashlib.sha256(item.content).hexdigest(),
                        item.content,
                        Jsonb(item.metadata),
                    ),
                )
        return blob_ids

    def get(self, *, user_id: int, blob_id: str, kind: str | None = None) -> StoredBlob | None:
        query = """
            SELECT id, logical_name, media_type, content
            FROM stored_blobs
            WHERE user_id = ? AND id = ?
        """
        params: tuple[object, ...] = (user_id, blob_id)
        if kind is not None:
            query += " AND kind = ?"
            params = (*params, kind)
        with self._store.connect() as connection:
            row = connection.execute(query, params).fetchone()
        if row is None:
            return None
        return StoredBlob(
            blob_id=str(row["id"]),
            logical_name=str(row["logical_name"]),
            media_type=str(row["media_type"]),
            content=bytes(row["content"]),
        )

    def get_latest_for_session(self, *, session_id: str, kind: str) -> StoredBlob | None:
        with self._store.connect() as connection:
            row = connection.execute(
                """
                SELECT id, logical_name, media_type, content
                FROM stored_blobs
                WHERE session_id = ? AND kind = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (session_id, kind),
            ).fetchone()
        if row is None:
            return None
        return StoredBlob(
            blob_id=str(row["id"]),
            logical_name=str(row["logical_name"]),
            media_type=str(row["media_type"]),
            content=bytes(row["content"]),
        )

    def get_for_session(
        self,
        *,
        session_id: str,
        blob_id: str,
        kind: str | None = None,
    ) -> StoredBlob | None:
        query = """
            SELECT id, logical_name, media_type, content
            FROM stored_blobs
            WHERE session_id = ? AND id = ?
        """
        params: tuple[object, ...] = (session_id, blob_id)
        if kind is not None:
            query += " AND kind = ?"
            params = (*params, kind)
        with self._store.connect() as connection:
            row = connection.execute(query, params).fetchone()
        if row is None:
            return None
        return StoredBlob(
            blob_id=str(row["id"]),
            logical_name=str(row["logical_name"]),
            media_type=str(row["media_type"]),
            content=bytes(row["content"]),
        )

    def delete_many(self, *, user_id: int, blob_ids: list[str]) -> None:
        if not blob_ids:
            return
        with self._store.connect() as connection:
            connection.execute(
                "DELETE FROM stored_blobs WHERE user_id = ? AND id = ANY(?::uuid[])",
                (user_id, blob_ids),
            )

    def delete_for_session(
        self,
        *,
        user_id: int,
        session_id: str,
        kinds: list[str],
    ) -> None:
        if not kinds:
            return
        with self._store.connect() as connection:
            connection.execute(
                """
                DELETE FROM stored_blobs
                WHERE user_id = ? AND session_id = ? AND kind = ANY(?::text[])
                """,
                (user_id, session_id, kinds),
            )

    def delete_ids_for_session(
        self,
        *,
        session_id: str,
        blob_ids: list[str],
        kind: str,
    ) -> None:
        if not blob_ids:
            return
        with self._store.connect() as connection:
            connection.execute(
                """
                DELETE FROM stored_blobs
                WHERE session_id = ? AND kind = ? AND id = ANY(?::uuid[])
                """,
                (session_id, kind, blob_ids),
            )
