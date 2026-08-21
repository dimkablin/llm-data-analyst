from types import SimpleNamespace
from unittest.mock import ANY, Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.api.deps import get_current_user
from backend.api.routes import semantic_catalog
from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticCatalogOperation,
    SemanticRelationship,
)


def test_metric_delete_dependency_error_is_returned_to_client(monkeypatch: pytest.MonkeyPatch) -> None:
    detail = "Metric base_metric is referenced by active metrics: derived_metric"
    service = Mock()
    service.delete_metric_for_connection.side_effect = ValueError(detail)
    service.delete_metric.side_effect = ValueError(detail)
    monkeypatch.setattr(semantic_catalog, "_semantic_catalog_service", service)
    monkeypatch.setattr(semantic_catalog, "_require_connection_owner", Mock())
    monkeypatch.setattr(semantic_catalog, "_require_semantic_editor", Mock())
    app = FastAPI()
    app.include_router(semantic_catalog.router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=7)

    client = TestClient(app)
    responses = (
        client.delete("/db-connections/connection/semantic-catalog/metrics/metric%3Abase"),
        client.delete("/sessions/session/semantic-catalog/metrics/metric%3Abase"),
    )

    for response in responses:
        assert response.status_code == 400
        assert response.json() == {"detail": detail}


def test_connection_relationship_crud_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    relationship = SemanticRelationship(
        relationship_id="relationship:sales_customers",
        from_table="public.sales",
        from_column="customer_id",
        to_table="public.customers",
        to_column="customer_id",
        cardinality="many_to_one",
        description="Each sale belongs to one customer.",
    )
    service = Mock()
    service.create_relationship_for_connection.return_value = relationship
    service.update_relationship_for_connection.return_value = relationship
    monkeypatch.setattr(semantic_catalog, "_semantic_catalog_service", service)
    monkeypatch.setattr(semantic_catalog, "_require_connection_owner", Mock())
    app = FastAPI()
    app.include_router(semantic_catalog.router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=7)
    client = TestClient(app)
    payload = relationship.model_dump(exclude={"relationship_id"})

    created = client.post(
        "/db-connections/connection/semantic-catalog/relationships",
        json=payload,
    )
    updated = client.patch(
        "/db-connections/connection/semantic-catalog/relationships/relationship%3Asales_customers",
        json={"description": relationship.description},
    )
    deleted = client.delete(
        "/db-connections/connection/semantic-catalog/relationships/relationship%3Asales_customers"
    )

    assert created.status_code == 200
    assert created.json()["description"] == relationship.description
    assert updated.status_code == 200
    assert deleted.status_code == 204


def test_connection_grantee_cannot_mutate_shared_semantic_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_db = Mock()
    auth_db.get_db_connection.return_value = SimpleNamespace(user_id=42)
    monkeypatch.setattr(semantic_catalog, "_auth_db", auth_db)

    with pytest.raises(HTTPException) as exc_info:
        semantic_catalog._require_connection_owner("connection", SimpleNamespace(id=7))

    assert getattr(exc_info.value, "status_code", None) == 403


def test_background_generation_reports_operation_without_mutating_catalog_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = SemanticCatalog(
        catalog_id="catalog",
        source_key="csv:user:source",
        source_type="csv",
        status="ready",
    )
    operation = SemanticCatalogOperation(
        operation_id=17,
        source_key=catalog.source_key,
        catalog_id=catalog.catalog_id,
        operation_type="generate",
        actor_user_id=7,
    )
    service = Mock()
    service.load_for_session.return_value = catalog
    service.claim_session_build.return_value = (catalog, operation)
    generator = Mock()
    monkeypatch.setattr(semantic_catalog, "_semantic_catalog_service", service)
    monkeypatch.setattr(semantic_catalog, "_semantic_generation_service", generator)
    monkeypatch.setattr(
        semantic_catalog,
        "_require_semantic_editor",
        Mock(return_value=SimpleNamespace(source_type="csv")),
    )
    app = FastAPI()
    app.include_router(semantic_catalog.router)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=7)

    response = TestClient(app).post(
        "/sessions/session/semantic-catalog/generate?background=true",
        json={"sample_rows": 0},
    )

    assert response.status_code == 200
    assert response.json() == {"accepted": True, "status": "ready", "operation_id": 17}
    assert catalog.status == "ready"
    generator.generate.assert_called_once_with(
        session_id="session",
        user_id=7,
        request=ANY,
        operation_id=17,
    )
