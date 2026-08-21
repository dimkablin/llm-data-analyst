from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.auth.auth_db import AuthDB
from backend.data_access.semantic_catalog_service import SemanticCatalogService
from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticMetricCreate,
    SemanticRelationshipCreate,
    SemanticRelationshipUpdate,
)
from tests.in_memory_semantic_store import (
    InMemorySemanticCatalogStore,
)
from tests.in_memory_semantic_store import (
    SemanticSessionStore as SessionStore,
)


class _FakeDBAnalyticsHelper:
    def __init__(self, *_, **__) -> None:
        self.preview_calls = 0

    def list_tables_with_columns(self):
        return [
            {
                "schema": "public",
                "table_name": "sales",
                "qualified_name": "public.sales",
                "table_type": "BASE TABLE",
                "columns": ["order_date", "amount", "region"],
            }
        ]

    def preview_table(self, *_args, **_kwargs):
        self.preview_calls += 1
        return pd.DataFrame(
            {
                "order_date": ["2026-01-01", "2026-01-02"],
                "amount": [100, 250],
                "region": ["EU", "US"],
            }
        )


class _FakeRelationalDBAnalyticsHelper(_FakeDBAnalyticsHelper):
    def list_tables_with_columns(self):
        return [
            {
                "schema": "public",
                "table_name": "sales",
                "qualified_name": "public.sales",
                "table_type": "BASE TABLE",
                "columns": ["order_date", "amount", "search_id"],
            },
            {
                "schema": "public",
                "table_name": "searches",
                "qualified_name": "public.searches",
                "table_type": "BASE TABLE",
                "columns": ["search_id", "query"],
            },
        ]

    def preview_table(self, table_name, **_kwargs):
        if table_name == "searches":
            return pd.DataFrame({"search_id": [1, 2], "query": ["LED", "SVO"]})
        return pd.DataFrame(
            {
                "order_date": ["2026-01-01", "2026-01-02"],
                "amount": [100, 250],
                "search_id": [1, 2],
            }
        )


class _CountingDBAnalyticsHelper(_FakeDBAnalyticsHelper):
    list_calls = 0

    def list_tables_with_columns(self):
        type(self).list_calls += 1
        return super().list_tables_with_columns()


def test_connection_catalog_is_not_built_until_explicit_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("backend.tools.impl.db_helpers.DBAnalyticsHelper", _FakeDBAnalyticsHelper)
    service = SemanticCatalogService(store=SessionStore(str(tmp_path), ttl_days=1))
    runtime = SimpleNamespace(connection_id="conn-1", name="Sales DB")

    assert service.status_for_connection(connection_id="conn-1").status == "not_built"

    catalog = service.build_for_connection(user_id=7, runtime=runtime, source_label="Sales DB")

    assert catalog.status == "ready"
    assert catalog.connection_id == "conn-1"
    assert catalog.source_key == "db_connection:conn-1"
    assert catalog.metrics == []
    assert service._load_overlay(catalog.source_key).version == 0
    assert service.load_for_connection(connection_id="conn-1", user_id=42).catalog_id == catalog.catalog_id


def test_ensure_connection_catalog_builds_once_and_reuses_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _CountingDBAnalyticsHelper.list_calls = 0
    monkeypatch.setattr(
        "backend.tools.impl.db_helpers.DBAnalyticsHelper",
        _CountingDBAnalyticsHelper,
    )
    service = SemanticCatalogService(store=SessionStore(str(tmp_path), ttl_days=1))
    runtime = SimpleNamespace(connection_id="conn-1", name="Sales DB")

    first = service.ensure_for_connection(user_id=7, runtime=runtime)
    second = service.ensure_for_connection(user_id=42, runtime=runtime)

    assert first.status == "ready"
    assert second.catalog_id == first.catalog_id
    assert second.user_id == 42
    assert _CountingDBAnalyticsHelper.list_calls == 1


def test_connection_catalog_build_claim_is_atomic() -> None:
    store = InMemorySemanticCatalogStore()

    def claim(user_id: int) -> bool:
        return store.save_published_if_absent(
            SemanticCatalog(
                catalog_id="catalog-1",
                connection_id="conn-1",
                user_id=user_id,
                source_key="db_connection:conn-1",
                source_type="db_connection",
                source_ref_id="conn-1",
                source_fingerprint="db:conn-1",
                status="indexing",
            )
        )

    with ThreadPoolExecutor(max_workers=20) as executor:
        claimed = list(executor.map(claim, range(100)))

    assert sum(claimed) == 1


def test_connection_refresh_keeps_last_ready_catalog_while_operation_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("backend.tools.impl.db_helpers.DBAnalyticsHelper", _FakeDBAnalyticsHelper)
    service = SemanticCatalogService(store=SessionStore(str(tmp_path), ttl_days=1))
    runtime = SimpleNamespace(connection_id="conn-1", name="Sales DB")
    ready = service.build_for_connection(user_id=7, runtime=runtime)

    visible, operation = service.claim_connection_build(
        connection_id="conn-1",
        user_id=7,
        force=True,
    )

    assert operation is not None
    assert visible.status == ready.status == "ready"
    assert service.load_for_connection(connection_id="conn-1", user_id=7).status == "ready"


def test_clear_cancels_operation_and_rejects_late_publish(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    service = SemanticCatalogService(store=store)
    pending, operation = service.claim_connection_build(
        connection_id="conn-1",
        user_id=7,
    )
    assert operation is not None
    candidate = pending.model_copy(update={"status": "ready"}, deep=True)

    service.clear_for_connection(connection_id="conn-1", user_id=7)

    assert (
        store.metadata_store.save_build_result_if_current(
            operation_id=operation.operation_id,
            generated=candidate,
            published=candidate,
        )
        is False
    )
    assert service.load_for_connection(connection_id="conn-1", user_id=7) is None
    assert service.latest_operation_for_connection(connection_id="conn-1").status == "cancelled"


def test_clear_pending_csv_catalog_without_profile(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.bind_csv_source(
        state.session_id,
        filename="orders.csv",
        source_ref_id="sha256:" + ("d" * 64),
    )
    service = SemanticCatalogService(store=store)
    pending, operation = service.claim_session_build(session_id=state.session_id, user_id=7)
    assert operation is not None

    service.clear_for_session(session_id=state.session_id, user_id=7)

    assert store.metadata_store.load_published(pending.source_key) is None
    assert store.metadata_store.load_latest_operation(pending.source_key).status == "cancelled"


def test_session_reuses_connection_catalog_without_session_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("backend.tools.impl.db_helpers.DBAnalyticsHelper", _FakeDBAnalyticsHelper)
    store = SessionStore(str(tmp_path), ttl_days=1)
    session = store.create_session()
    store.bind_db_connection_source(
        session.session_id,
        connection_id="conn-1",
        label="Sales DB",
    )
    service = SemanticCatalogService(store=store)
    built = service.build_for_connection(
        user_id=7,
        runtime=SimpleNamespace(connection_id="conn-1", name="Sales DB"),
    )

    loaded = service.load_for_session(session_id=session.session_id, user_id=42)

    assert store.load_data_catalog(session.session_id) is None
    assert loaded is not None
    assert loaded.catalog_id == built.catalog_id
    assert loaded.session_id == session.session_id
    assert loaded.user_id == 42


def test_connection_catalog_can_be_marked_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("backend.tools.impl.db_helpers.DBAnalyticsHelper", _FakeDBAnalyticsHelper)
    service = SemanticCatalogService(store=SessionStore(str(tmp_path), ttl_days=1))
    service.build_for_connection(
        user_id=7,
        runtime=SimpleNamespace(connection_id="conn-1", name="Sales DB"),
    )

    service.mark_stale_for_connection(
        connection_id="conn-1",
        reason="Connection settings changed.",
    )

    catalog = service.load_for_connection(connection_id="conn-1", user_id=7)
    assert catalog is not None
    assert catalog.status == "stale"
    assert catalog.error == "Connection settings changed."


def test_connection_metric_rejects_unsafe_formula(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("backend.tools.impl.db_helpers.DBAnalyticsHelper", _FakeDBAnalyticsHelper)
    service = SemanticCatalogService(store=SessionStore(str(tmp_path), ttl_days=1))
    service.build_for_connection(
        user_id=7,
        runtime=SimpleNamespace(connection_id="conn-1", name="Sales DB"),
        source_label="Sales DB",
    )

    with pytest.raises(ValueError, match="Only read-only aggregate expressions are allowed"):
        service.create_metric_for_connection(
            connection_id="conn-1",
            user_id=7,
            payload=SemanticMetricCreate(
                key="bad_revenue",
                name="Bad revenue",
                type="derived",
                base_table="public.sales",
                formula="SUM(amount); DROP TABLE sales",
            ),
        )


def test_connection_metric_accepts_qualified_base_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("backend.tools.impl.db_helpers.DBAnalyticsHelper", _FakeDBAnalyticsHelper)
    service = SemanticCatalogService(store=SessionStore(str(tmp_path), ttl_days=1))
    service.build_for_connection(
        user_id=7,
        runtime=SimpleNamespace(connection_id="conn-1", name="Sales DB"),
        source_label="Sales DB",
    )

    metric = service.create_metric_for_connection(
        connection_id="conn-1",
        user_id=7,
        payload=SemanticMetricCreate(
            key="adjusted_revenue",
            name="Adjusted revenue",
            type="derived",
            base_table="public.sales",
            formula="SUM(public.sales.amount) * 7.5",
        ),
    )

    assert metric.formula == "SUM(public.sales.amount) * 7.5"


def test_connection_relationship_crud_persists_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "backend.tools.impl.db_helpers.DBAnalyticsHelper",
        _FakeRelationalDBAnalyticsHelper,
    )
    service = SemanticCatalogService(store=SessionStore(str(tmp_path), ttl_days=1))
    service.build_for_connection(
        user_id=7,
        runtime=SimpleNamespace(connection_id="conn-1", name="Sales DB"),
        source_label="Sales DB",
    )

    created = service.create_relationship_for_connection(
        connection_id="conn-1",
        user_id=7,
        payload=SemanticRelationshipCreate(
            from_table="public.sales",
            from_column="search_id",
            to_table="public.searches",
            to_column="search_id",
            cardinality="many_to_one",
            description="Each sale belongs to one search.",
        ),
    )
    assert service.load_for_connection(connection_id="conn-1", user_id=7).relationships == [created]

    updated = service.update_relationship_for_connection(
        connection_id="conn-1",
        user_id=7,
        relationship_id=created.relationship_id,
        payload=SemanticRelationshipUpdate(description="Each sale has exactly one search."),
    )
    assert updated.description == "Each sale has exactly one search."

    service.delete_relationship_for_connection(
        connection_id="conn-1",
        user_id=7,
        relationship_id=created.relationship_id,
    )
    assert service.load_for_connection(connection_id="conn-1", user_id=7).relationships == []


def test_metric_sql_validation_accepts_unicode_qualified_column(tmp_path: Path) -> None:
    service = SemanticCatalogService(store=SessionStore(str(tmp_path), ttl_days=1))

    service._validate_metric_sql(
        "SUM(public.продажи.сумма) * 7.5",
        {"public.продажи.сумма"},
        aggregate_refs=set(),
    )


def test_db_connection_acl_shares_one_connection_without_global_visibility(tmp_path: Path) -> None:
    auth = AuthDB(str(tmp_path / "auth.db"), token_ttl_days=30)
    owner = auth.create_user("owner_user", "secret", is_admin=False)
    other = auth.create_user("other_user", "secret", is_admin=False)
    outsider = auth.create_user("outsider_user", "secret", is_admin=False)
    connection = auth.create_db_connection(
        owner.id,
        name="Sales DB",
        db_type="postgresql",
        host="db.example",
        port=5432,
        database="sales",
        username="analyst",
        options_json=None,
    )

    assert auth.list_db_connections(other.id) == []

    assert auth.grant_db_connection_access(owner.id, connection.id, other.id)

    assert [item.id for item in auth.list_db_connections(other.id)] == [connection.id]
    assert auth.get_db_connection(other.id, connection.id) is not None
    assert auth.get_db_connection(outsider.id, connection.id) is None
