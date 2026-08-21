from pathlib import Path

import pytest

from backend.data_access.data_catalog import DataCatalogSnapshot
from backend.data_access.semantic_catalog_service import SemanticCatalogService
from backend.sessions.session_store import SessionStore


def test_profile_save_has_no_json_fallback(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)

    with pytest.raises(RuntimeError, match="PostgreSQL metadata store is required"):
        store.save_data_catalog("session-1", DataCatalogSnapshot(source_fingerprint="csv:test"))

    assert not (tmp_path / "session-1" / "data_catalog.json").exists()


def test_semantic_catalog_has_no_file_store_fallback(tmp_path: Path) -> None:
    service = SemanticCatalogService(store=SessionStore(str(tmp_path), ttl_days=1))

    with pytest.raises(RuntimeError, match="PostgreSQL semantic metadata store is required"):
        _ = service.catalog_store
