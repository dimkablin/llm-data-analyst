from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from backend.api.models import DBConnectionUpdateRequest
from backend.api.routes import db_connections


def _connection(*, name: str = "Sales DB", host: str = "db.example") -> SimpleNamespace:
    return SimpleNamespace(
        id="conn-1",
        name=name,
        db_type="postgresql",
        host=host,
        port=5432,
        database="sales",
        username="analyst",
        options_json={"schema": "public"},
        password_present=True,
        last_test_at=None,
        last_test_ok=None,
        last_error=None,
        created_at="2026-08-11T00:00:00Z",
        updated_at="2026-08-11T00:00:00Z",
    )


def test_connection_scope_change_marks_catalog_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = Mock()
    service.get_connection.return_value = _connection()
    service.update_connection.return_value = _connection(host="new-db.example")
    catalog = Mock()
    monkeypatch.setattr(db_connections, "_db_connections_service", service)
    monkeypatch.setattr(db_connections, "_semantic_catalog_service", catalog)

    response = db_connections.update_db_connection(
        "conn-1",
        DBConnectionUpdateRequest(host="new-db.example"),
        SimpleNamespace(id=7),
    )

    assert response.host == "new-db.example"
    catalog.mark_stale_for_connection.assert_called_once_with(
        connection_id="conn-1",
        reason="Database connection settings changed. Refresh the semantic catalog.",
    )


def test_connection_name_change_keeps_catalog_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = Mock()
    service.get_connection.return_value = _connection()
    service.update_connection.return_value = _connection(name="Revenue DB")
    catalog = Mock()
    monkeypatch.setattr(db_connections, "_db_connections_service", service)
    monkeypatch.setattr(db_connections, "_semantic_catalog_service", catalog)

    db_connections.update_db_connection(
        "conn-1",
        DBConnectionUpdateRequest(name="Revenue DB"),
        SimpleNamespace(id=7),
    )

    catalog.mark_stale_for_connection.assert_not_called()


def test_connection_delete_clears_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    service = Mock()
    catalog = Mock()
    monkeypatch.setattr(db_connections, "_db_connections_service", service)
    monkeypatch.setattr(db_connections, "_semantic_catalog_service", catalog)

    db_connections.delete_db_connection("conn-1", SimpleNamespace(id=7))

    service.delete_connection.assert_called_once_with(7, "conn-1")
    catalog.clear_for_connection.assert_called_once_with(connection_id="conn-1", user_id=7)


def test_connection_delete_stops_when_catalog_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = Mock()
    catalog = Mock()
    catalog.clear_for_connection.side_effect = RuntimeError("metadata unavailable")
    monkeypatch.setattr(db_connections, "_db_connections_service", service)
    monkeypatch.setattr(db_connections, "_semantic_catalog_service", catalog)

    with pytest.raises(HTTPException) as exc_info:
        db_connections.delete_db_connection("conn-1", SimpleNamespace(id=7))

    assert getattr(exc_info.value, "status_code", None) == 409
    service.delete_connection.assert_not_called()
