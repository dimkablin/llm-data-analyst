from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.api import deps
from backend.api.routes import data
from backend.auth.auth_db import AuthUser
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.orchestrator import NotebookOrchestrator
from backend.notebook.store import NotebookStore
from tests.in_memory_semantic_store import SemanticSessionStore as SessionStore


def test_readonly_sql_allows_keywords_in_literals_but_rejects_multiple_statements() -> None:
    sql = "SELECT * FROM dataset WHERE contact_center = 'Call Center'"

    assert data._ensure_safe_readonly_sql(sql) == sql
    with pytest.raises(HTTPException, match="Only read-only queries are allowed"):
        data._ensure_safe_readonly_sql("SELECT 1; DROP TABLE dataset")


class _FakeAuthDB:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.has_dataset: bool | None = None
        self.semantic_refreshes: list[tuple[str, int]] = []

    def get_user_by_token(self, token: str) -> AuthUser | None:
        if token != "token":
            return None
        return AuthUser(id=7, username="analyst", is_admin=False, created_at="now")

    def is_session_owner(self, session_id: str, user_id: int) -> bool:
        return session_id == self.session_id and user_id == 7

    def mark_session_has_dataset(self, session_id: str, has_dataset: bool) -> None:
        assert session_id == self.session_id
        self.has_dataset = has_dataset


class _FakeSemanticCatalogService:
    def __init__(self, store: SessionStore, auth_db: _FakeAuthDB) -> None:
        self.store = store
        self.auth_db = auth_db

    def refresh(self, *, session_id: str, user_id: int, operation_id: int) -> None:
        assert operation_id == 23
        snapshot = self.store.load_data_catalog(session_id)
        assert snapshot is not None and snapshot.tables
        self.auth_db.semantic_refreshes.append((session_id, user_id))

    def claim_session_build(self, *, session_id: str, user_id: int):
        return (
            SimpleNamespace(source_key=f"csv:{user_id}:{session_id}"),
            SimpleNamespace(operation_id=23),
        )

    def mark_build_failed(self, *, source_key: str, error: str, operation_id: int) -> None:
        assert operation_id == 23
        raise AssertionError(f"Unexpected semantic build failure for {source_key}: {error}")


def _client(tmp_path: Path) -> tuple[TestClient, SessionStore, CSVSessionRuntime, _FakeAuthDB, str]:
    store = SessionStore(str(tmp_path / "legacy"), ttl_days=7)
    session = store.create_session()
    auth_db = _FakeAuthDB(session.session_id)
    csv_runtime = CSVSessionRuntime(base_dir=tmp_path / "duckdb", default_ttl_sec=3600)
    notebook_store = NotebookStore(tmp_path)
    manifest_store = ManifestStore(tmp_path)
    orchestrator = NotebookOrchestrator(notebook_store)
    deps.set_auth_db(auth_db)
    data.setup(
        auth_db=auth_db,  # type: ignore[arg-type]
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        notebook_orchestrator=orchestrator,
        storage_dir=tmp_path,
        semantic_catalog_service=_FakeSemanticCatalogService(store, auth_db),
    )
    app = FastAPI()
    app.include_router(data.router)
    return TestClient(app), store, csv_runtime, auth_db, session.session_id


def test_batch_upload_endpoint_ingests_multiple_files_into_one_duckdb(tmp_path: Path) -> None:
    client, _store, csv_runtime, auth_db, session_id = _client(tmp_path)

    response = client.post(
        f"/sessions/{session_id}/data/batch",
        headers={"Authorization": "Bearer token"},
        files=[
            ("files", ("orders.csv", b"order_id,customer_id,amount\n1,10,120\n", "text/csv")),
            ("files", ("customers.csv", b"customer_id,segment\n10,enterprise\n", "text/csv")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == session_id
    assert payload["csv_session_id"] == session_id
    assert payload["table_names"] == ["customers", "orders"]
    assert [item["table_name"] for item in payload["files"]] == ["orders", "customers"]
    assert auth_db.has_dataset is True
    assert auth_db.semantic_refreshes == [(session_id, 7)]

    joined = csv_runtime.query_dataframe(
        session_id,
        "SELECT o.order_id, c.segment FROM orders o JOIN customers c USING (customer_id)",
    )
    assert joined.to_dict(orient="records") == [{"order_id": 1, "segment": "enterprise"}]


def test_single_upload_endpoint_accepts_xlsx_and_keeps_legacy_response(tmp_path: Path) -> None:
    client, store, _csv_runtime, _auth_db, session_id = _client(tmp_path)
    workbook = BytesIO()
    pd.DataFrame({"sku": ["A", "B"], "price": [10, 20]}).to_excel(workbook, index=False)

    response = client.post(
        f"/sessions/{session_id}/data",
        headers={"Authorization": "Bearer token"},
        files={
            "file": (
                "prices.xlsx",
                workbook.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 200
    assert response.json() == {"session_id": session_id, "rows": 2, "columns": 2}
    state = store.load_session(session_id)
    assert state is not None
    assert state.csv_table_names == ["prices"]


def test_batch_upload_endpoint_applies_preprocessing_options(tmp_path: Path) -> None:
    client, _store, csv_runtime, _auth_db, session_id = _client(tmp_path)
    workbook = BytesIO()
    pd.DataFrame(
        [
            ["Report", None, None],
            ["customer_id", "segment", "amount"],
            [10, "enterprise", 120],
            [None, "technical note", None],
        ]
    ).to_excel(workbook, index=False, header=False)

    response = client.post(
        f"/sessions/{session_id}/data/batch",
        headers={"Authorization": "Bearer token"},
        data={
            "preprocessing_options": json.dumps(
                {
                    "drop_sparse_rows": False,
                }
            )
        },
        files=[
            (
                "files",
                (
                    "customers.xlsx",
                    workbook.getvalue(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ),
            )
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["files"][0]["preprocessing"]["detected_header_row"] == 2
    assert payload["files"][0]["preprocessing"]["cleaned_rows"] == 2
    queried = csv_runtime.query_dataframe(
        session_id,
        "SELECT customer_id, segment, amount FROM customers ORDER BY segment",
    )
    assert queried["segment"].tolist() == ["enterprise", "technical note"]


def test_batch_upload_endpoint_rejects_invalid_preprocessing_options(tmp_path: Path) -> None:
    client, _store, _csv_runtime, _auth_db, session_id = _client(tmp_path)

    response = client.post(
        f"/sessions/{session_id}/data/batch",
        headers={"Authorization": "Bearer token"},
        data={"preprocessing_options": json.dumps({"header_scan_rows": 0})},
        files=[
            (
                "files",
                ("orders.csv", b"order_id,amount\n1,120\n", "text/csv"),
            )
        ],
    )

    assert response.status_code == 400
    assert "Invalid preprocessing options" in response.json()["detail"]
