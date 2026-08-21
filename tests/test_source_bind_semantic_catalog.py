from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import BackgroundTasks

from backend.api.models import SessionBindDBConnectionSourceRequest
from backend.api.routes import semantic_catalog as semantic_catalog_route
from backend.api.routes import sources


def test_db_bind_reuses_connection_catalog_without_session_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_user = SimpleNamespace(id=7)
    connection = SimpleNamespace(id="conn-1", name="Sales DB", user_id=7)
    refreshed = SimpleNamespace(
        source_type="db_connection",
        source_ref_id="conn-1",
        source_label="Sales DB",
        source_mode=None,
    )
    store = Mock()
    connections = Mock()
    connections.get_connection.return_value = connection
    semantic_catalog = Mock()
    operation = SimpleNamespace(operation_id=11)
    semantic_catalog.claim_connection_build.return_value = (SimpleNamespace(), operation)
    background_tasks = BackgroundTasks()

    monkeypatch.setattr(sources, "_store", store)
    monkeypatch.setattr(sources, "_db_connections_service", connections)
    monkeypatch.setattr(sources, "_db_runtime_service", Mock())
    monkeypatch.setattr(sources, "_semantic_catalog_service", semantic_catalog)
    monkeypatch.setattr(sources, "settings", SimpleNamespace(semantic_layer_enabled=True))
    monkeypatch.setattr(sources, "_add_source_to_manifest", Mock())
    monkeypatch.setattr(
        sources,
        "_load_owned_session",
        Mock(side_effect=[SimpleNamespace(), refreshed]),
    )

    response = sources.bind_session_db_connection_source(
        "session-1",
        background_tasks,
        SessionBindDBConnectionSourceRequest(connection_id="conn-1"),
        current_user,
    )

    store.bind_db_connection_source.assert_called_once_with(
        "session-1",
        connection_id="conn-1",
        label="Sales DB",
        source_mode=None,
    )
    semantic_catalog.claim_connection_build.assert_called_once_with(
        connection_id="conn-1",
        user_id=7,
        source_label="Sales DB",
    )
    assert len(background_tasks.tasks) == 1
    semantic_catalog.ensure_for_connection.assert_not_called()
    semantic_catalog.refresh.assert_not_called()
    assert response.source_ref_id == "conn-1"


def test_db_grantee_bind_does_not_build_shared_semantic_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_user = SimpleNamespace(id=7)
    connection = SimpleNamespace(id="conn-1", name="Sales DB", user_id=42)
    refreshed = SimpleNamespace(
        source_type="db_connection",
        source_ref_id="conn-1",
        source_label="Sales DB",
        source_mode=None,
    )
    store = Mock()
    connections = Mock()
    connections.get_connection.return_value = connection
    semantic_catalog = Mock()

    monkeypatch.setattr(sources, "_store", store)
    monkeypatch.setattr(sources, "_db_connections_service", connections)
    monkeypatch.setattr(sources, "_db_runtime_service", Mock())
    monkeypatch.setattr(sources, "_semantic_catalog_service", semantic_catalog)
    monkeypatch.setattr(sources, "settings", SimpleNamespace(semantic_layer_enabled=True))
    monkeypatch.setattr(sources, "_add_source_to_manifest", Mock())
    monkeypatch.setattr(
        sources,
        "_load_owned_session",
        Mock(side_effect=[SimpleNamespace(), refreshed]),
    )

    sources.bind_session_db_connection_source(
        "session-1",
        BackgroundTasks(),
        SessionBindDBConnectionSourceRequest(connection_id="conn-1"),
        current_user,
    )

    semantic_catalog.claim_connection_build.assert_not_called()


def test_connection_build_returns_pending_without_profiling_in_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_user = SimpleNamespace(id=7)
    runtime = SimpleNamespace(name="Sales DB")
    pending = SimpleNamespace(status="indexing", source_label="Sales DB")
    semantic_catalog = Mock()
    operation = SimpleNamespace(operation_id=12)
    semantic_catalog.claim_connection_build.return_value = (pending, operation)
    runtime_service = Mock()
    runtime_service.get_runtime_config.return_value = runtime
    background_tasks = BackgroundTasks()

    monkeypatch.setattr(semantic_catalog_route, "_semantic_catalog_service", semantic_catalog)
    monkeypatch.setattr(semantic_catalog_route, "_db_runtime_service", runtime_service)
    monkeypatch.setattr(semantic_catalog_route, "_require_connection_owner", Mock(return_value=runtime))

    response = semantic_catalog_route.build_connection_semantic_catalog(
        "conn-1",
        background_tasks,
        current_user,
    )

    assert response.accepted is True
    assert response.status == "indexing"
    assert response.operation_id == 12
    assert len(background_tasks.tasks) == 1
    semantic_catalog.build_for_connection.assert_not_called()


def test_session_refresh_returns_pending_without_profiling_in_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_user = SimpleNamespace(id=7)
    pending = SimpleNamespace(status="indexing", source_key="csv:source")
    semantic_catalog = Mock()
    operation = SimpleNamespace(operation_id=13)
    semantic_catalog.claim_session_build.return_value = (pending, operation)
    background_tasks = BackgroundTasks()

    monkeypatch.setattr(semantic_catalog_route, "_semantic_catalog_service", semantic_catalog)
    monkeypatch.setattr(
        semantic_catalog_route,
        "_load_owned_session",
        Mock(return_value=SimpleNamespace(source_type="csv", source_ref_id="sha256:source")),
    )

    response = semantic_catalog_route.refresh_semantic_catalog(
        "session-1",
        background_tasks,
        current_user,
    )

    assert response.accepted is True
    assert response.status == "indexing"
    assert response.operation_id == 13
    assert len(background_tasks.tasks) == 1
    semantic_catalog.refresh.assert_not_called()


def test_source_removal_queues_profile_refresh_instead_of_blocking_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_user = SimpleNamespace(id=7)
    state = SimpleNamespace(source_type=None, source_ref_id=None)
    source_service = Mock()
    background_tasks = BackgroundTasks()
    monkeypatch.setattr(sources, "_load_owned_session", Mock(return_value=state))
    manifest_store = Mock()
    manifest_store.load.return_value.source_by_alias.return_value = SimpleNamespace()
    monkeypatch.setattr(sources, "_manifest_store", manifest_store)
    monkeypatch.setattr(sources, "_semantic_catalog_service", None)
    monkeypatch.setattr(sources, "SessionSourceService", Mock(return_value=source_service))
    monkeypatch.setattr(sources, "list_session_sources", Mock(return_value=[]))

    response = sources.remove_session_source(
        "session-1",
        "orders",
        background_tasks,
        current_user,
    )

    assert response == []
    source_service.remove_source.assert_called_once_with(
        session_id="session-1",
        alias="orders",
        refresh_catalog=False,
    )
    assert len(background_tasks.tasks) == 1
    source_service.refresh_catalog.assert_not_called()


def test_source_removal_clears_unshared_session_semantic_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_user = SimpleNamespace(id=7)
    state = SimpleNamespace(source_type="csv", source_ref_id="sha256:source")
    cleared_catalog = SimpleNamespace(source_key="csv:catalog", source_type="csv")
    semantic_catalog = Mock()
    semantic_catalog.load_for_session.return_value = cleared_catalog
    semantic_catalog.claim_session_build.return_value = (SimpleNamespace(), None)
    source_service = Mock()
    background_tasks = BackgroundTasks()
    auth_db = Mock()
    auth_db.list_sessions.return_value = [{"session_id": "session-1"}]
    monkeypatch.setattr(sources, "_auth_db", auth_db)
    monkeypatch.setattr(sources, "_load_owned_session", Mock(return_value=state))
    manifest_store = Mock()
    manifest_store.load.return_value.source_by_alias.return_value = SimpleNamespace()
    monkeypatch.setattr(sources, "_manifest_store", manifest_store)
    monkeypatch.setattr(sources, "_semantic_catalog_service", semantic_catalog)
    monkeypatch.setattr(sources, "SessionSourceService", Mock(return_value=source_service))
    monkeypatch.setattr(sources, "list_session_sources", Mock(return_value=[]))

    response = sources.remove_session_source(
        "session-1",
        "orders",
        background_tasks,
        current_user,
    )

    assert response == []
    semantic_catalog.clear_source.assert_called_once_with("csv:catalog")
    source_service.remove_source.assert_called_once_with(
        session_id="session-1",
        alias="orders",
        refresh_catalog=False,
    )
