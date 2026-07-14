from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol

from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticCatalogOverlay,
    stable_id,
    utc_now_iso,
)

_TABLE_NAME = "semantic_catalog_documents"


class SemanticCatalogStore(Protocol):
    def save_generated(self, catalog: SemanticCatalog) -> None: ...
    def load_generated(self, source_key: str) -> SemanticCatalog | None: ...
    def save_published(self, catalog: SemanticCatalog) -> None: ...
    def load_published(self, source_key: str) -> SemanticCatalog | None: ...
    def load_overlay(self, source_key: str) -> SemanticCatalogOverlay: ...
    def save_overlay(self, overlay: SemanticCatalogOverlay) -> None: ...


class SemanticCatalogFileStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def save_generated(self, catalog: SemanticCatalog) -> None:
        self._write_json(self._generated_path(catalog.source_key), catalog.model_dump())

    def load_generated(self, source_key: str) -> SemanticCatalog | None:
        return _read_model(self._generated_path(source_key), SemanticCatalog)

    def save_published(self, catalog: SemanticCatalog) -> None:
        self._write_json(self._published_path(catalog.source_key), catalog.model_dump())

    def load_published(self, source_key: str) -> SemanticCatalog | None:
        return _read_model(self._published_path(source_key), SemanticCatalog)

    def load_overlay(self, source_key: str) -> SemanticCatalogOverlay:
        overlay = _read_model(self._overlay_path(source_key), SemanticCatalogOverlay)
        return overlay or SemanticCatalogOverlay(source_key=source_key)

    def save_overlay(self, overlay: SemanticCatalogOverlay) -> None:
        overlay.version = int(overlay.version or 0) + 1
        overlay.updated_at = utc_now_iso()
        self._write_json(self._overlay_path(overlay.source_key), overlay.model_dump())

    def _semantic_root(self) -> Path:
        return self.root_dir / "semantic" / "sources"

    def _source_dir(self, source_key: str) -> Path:
        return self._semantic_root() / stable_id("semantic-source-dir", source_key)

    def _generated_path(self, source_key: str) -> Path:
        return self._source_dir(source_key) / "semantic_catalog.generated.json"

    def _overlay_path(self, source_key: str) -> Path:
        return self._source_dir(source_key) / "overlays.global.json"

    def _published_path(self, source_key: str) -> Path:
        return self._source_dir(source_key) / "semantic_catalog.published.json"

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=0)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


class SemanticCatalogPostgresStore:
    def __init__(self, dsn: str) -> None:
        clean = str(dsn or "").strip()
        if not clean:
            raise ValueError("SEMANTIC_CATALOG_POSTGRES_DSN must be set")
        self.dsn = clean
        self._ready = False

    def save_generated(self, catalog: SemanticCatalog) -> None:
        self._save("generated", catalog.source_key, catalog.model_dump(), catalog)

    def load_generated(self, source_key: str) -> SemanticCatalog | None:
        return _model_from_payload(self._load("generated", source_key), SemanticCatalog)

    def save_published(self, catalog: SemanticCatalog) -> None:
        self._save("published", catalog.source_key, catalog.model_dump(), catalog)

    def load_published(self, source_key: str) -> SemanticCatalog | None:
        return _model_from_payload(self._load("published", source_key), SemanticCatalog)

    def load_overlay(self, source_key: str) -> SemanticCatalogOverlay:
        overlay = _model_from_payload(self._load("overlay", source_key), SemanticCatalogOverlay)
        return overlay or SemanticCatalogOverlay(source_key=source_key)

    def save_overlay(self, overlay: SemanticCatalogOverlay) -> None:
        overlay.version = int(overlay.version or 0) + 1
        overlay.updated_at = utc_now_iso()
        self._save("overlay", overlay.source_key, overlay.model_dump(), overlay)

    def _save(
        self,
        doc_type: str,
        source_key: str,
        payload: dict[str, Any],
        model: SemanticCatalog | SemanticCatalogOverlay,
    ) -> None:
        import psycopg
        from psycopg.types.json import Jsonb

        with psycopg.connect(self.dsn) as conn:
            self._ensure_schema(conn)
            conn.execute(
                f"""
                INSERT INTO {_TABLE_NAME}(
                    source_key, doc_type, catalog_id, source_fingerprint,
                    version, payload, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (source_key, doc_type) DO UPDATE SET
                    catalog_id = EXCLUDED.catalog_id,
                    source_fingerprint = EXCLUDED.source_fingerprint,
                    version = EXCLUDED.version,
                    payload = EXCLUDED.payload,
                    updated_at = now()
                """,
                (
                    source_key,
                    doc_type,
                    getattr(model, "catalog_id", None),
                    getattr(model, "source_fingerprint", ""),
                    _document_version(model),
                    Jsonb(payload),
                ),
            )

    def _load(self, doc_type: str, source_key: str) -> dict[str, Any] | None:
        import psycopg

        with psycopg.connect(self.dsn) as conn:
            self._ensure_schema(conn)
            row = conn.execute(
                f"SELECT payload FROM {_TABLE_NAME} WHERE source_key = %s AND doc_type = %s",
                (source_key, doc_type),
            ).fetchone()
        if row is None:
            return None
        payload = row[0]
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            return json.loads(payload)
        return None

    def _ensure_schema(self, conn: Any) -> None:
        if self._ready:
            return
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
                source_key text NOT NULL,
                doc_type text NOT NULL,
                catalog_id text,
                source_fingerprint text NOT NULL DEFAULT '',
                version integer NOT NULL DEFAULT 0,
                payload jsonb NOT NULL,
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (source_key, doc_type)
            )
            """
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{_TABLE_NAME}_catalog_id ON {_TABLE_NAME}(catalog_id)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{_TABLE_NAME}_fingerprint ON {_TABLE_NAME}(source_fingerprint)"
        )
        self._ready = True


def semantic_catalog_store_from_settings(settings: Any, root_dir: Path) -> SemanticCatalogStore:
    kind = str(getattr(settings, "semantic_catalog_store", "file") or "file").strip().lower()
    if kind == "postgres":
        return SemanticCatalogPostgresStore(
            str(getattr(settings, "semantic_catalog_postgres_dsn", "") or "")
        )
    return SemanticCatalogFileStore(root_dir)


def _read_model(path: Path, model_cls: Any) -> Any | None:
    if not path.exists():
        return None
    try:
        return model_cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return None


def _model_from_payload(payload: dict[str, Any] | None, model_cls: Any) -> Any | None:
    if payload is None:
        return None
    try:
        return model_cls.model_validate(payload)
    except (ValueError, TypeError):
        return None


def _document_version(model: Any) -> int:
    for attr in ("published_version", "overlay_version", "version"):
        raw = getattr(model, attr, None)
        if raw is None:
            continue
        try:
            return int(raw or 0)
        except (TypeError, ValueError):
            continue
    return 0
