from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

import backend.tools.impl  # noqa: F401 - side-effect import avoids sql_table_service circular import
from backend.auth.blob_store import BlobWrite, StoredBlob
from backend.data_access.csv_runtime_state_service import CSVRuntimeStateService
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.data_access.sql_table_service import SQLTableService
from backend.data_access.tabular_upload_service import (
    TabularUploadError,
    TabularUploadFile,
    TabularUploadService,
)
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.orchestrator import EditResult, NotebookEdit, NotebookOrchestrator
from backend.notebook.store import NotebookStore
from backend.sessions.session_store import SessionStore
from backend.tools.impl.plotly_tool import PlotlyTool
from backend.tools.impl.sql_tool import SQLTool
from backend.tools.sandbox import SessionSandbox


def _service(tmp_path: Path) -> tuple[TabularUploadService, SessionStore, CSVSessionRuntime, str]:
    store = SessionStore(str(tmp_path / "legacy"), ttl_days=7)
    session = store.create_session()
    csv_runtime = CSVSessionRuntime(base_dir=tmp_path / "duckdb", default_ttl_sec=3600)
    notebook_store = NotebookStore(tmp_path)
    manifest_store = ManifestStore(tmp_path)
    orchestrator = NotebookOrchestrator(notebook_store)
    service = TabularUploadService(
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        notebook_orchestrator=orchestrator,
        storage_dir=tmp_path,
    )
    return service, store, csv_runtime, session.session_id


def test_ingest_multiple_csv_files_registers_joinable_tables_in_one_duckdb(tmp_path: Path) -> None:
    service, store, csv_runtime, session_id = _service(tmp_path)

    result = service.ingest_files(
        session_id=session_id,
        files=[
            TabularUploadFile(
                file_name="orders.csv",
                content=b"order_id,customer_id,amount\n1,10,120\n2,20,80\n",
            ),
            TabularUploadFile(
                file_name="customers.csv",
                content=b"customer_id,segment\n10,enterprise\n20,retail\n",
            ),
        ],
        ttl_seconds=3600,
    )

    assert result.session_id == session_id
    assert result.csv_session_id == session_id
    assert [item.table_name for item in result.files] == ["orders", "customers"]
    assert result.table_names == ["customers", "orders"]

    joined = csv_runtime.query_dataframe(
        session_id,
        """
        SELECT o.order_id, c.segment, o.amount
        FROM orders AS o
        JOIN customers AS c USING (customer_id)
        ORDER BY o.order_id
        """,
    )
    assert joined.to_dict(orient="records") == [
        {"order_id": 1, "segment": "enterprise", "amount": 120},
        {"order_id": 2, "segment": "retail", "amount": 80},
    ]

    state = store.load_session(session_id)
    assert state is not None
    assert state.csv_loaded is True
    assert state.csv_session_id == session_id
    assert state.csv_table_names == ["customers", "orders"]
    assert state.dataset_name == "orders.csv, customers.csv"
    assert state.source_label == "orders.csv, customers.csv"
    assert state.source_ref_id is not None
    assert state.source_ref_id.startswith("sha256:")
    assert state.source_ref_id != state.source_label

    manifest = ManifestStore(tmp_path).load(session_id)
    assert [source.display_name for source in manifest.sources] == ["orders.csv", "customers.csv"]
    assert [source.csv_table_names for source in manifest.sources] == [["orders"], ["customers"]]
    for source in manifest.sources:
        assert source.parquet_path is not None
        assert (tmp_path / "sessions" / session_id / source.parquet_path).is_file()

    notebook = NotebookStore(tmp_path).load(session_id)
    assert len(notebook.code_cells) == 2
    assert all(cell.is_source_binding for cell in notebook.code_cells)


def test_ingest_links_manifest_source_to_durable_original(tmp_path: Path) -> None:
    class FakeBlobStore:
        def __init__(self) -> None:
            self.items: list[BlobWrite] = []

        def put_many(self, **kwargs) -> list[str]:
            self.items = kwargs["items"]
            return ["blob-1"]

        def delete_many(self, **_kwargs) -> None:
            raise AssertionError("successful ingest must not delete its original")

    blob_store = FakeBlobStore()
    store = SessionStore(str(tmp_path / "legacy"), ttl_days=7)
    session_id = store.create_session().session_id
    service = TabularUploadService(
        store=store,
        csv_runtime=CSVSessionRuntime(base_dir=tmp_path / "duckdb", default_ttl_sec=3600),
        manifest_store=ManifestStore(tmp_path),
        notebook_orchestrator=NotebookOrchestrator(NotebookStore(tmp_path)),
        storage_dir=tmp_path,
        blob_store=blob_store,  # type: ignore[arg-type]
    )

    service.ingest_files(
        session_id=session_id,
        user_id=7,
        files=[TabularUploadFile(file_name="orders.csv", content=b"id,amount\n1,10\n")],
    )

    source = ManifestStore(tmp_path).load(session_id).sources[0]
    assert source.blob_id == "blob-1"
    assert blob_store.items[0].content == b"id,amount\n1,10\n"


def test_csv_runtime_restores_from_postgres_original_after_local_files_are_lost(
    tmp_path: Path,
) -> None:
    class FakeBlobStore:
        content = b"id,amount\n1,10\n2,20\n"

        def put_many(self, **_kwargs) -> list[str]:
            return ["blob-1"]

        def delete_many(self, **_kwargs) -> None:
            return None

        def get_for_session(self, **_kwargs) -> StoredBlob:
            return StoredBlob(
                blob_id="blob-1",
                logical_name="orders.csv",
                media_type="text/csv",
                content=self.content,
            )

    store = SessionStore(str(tmp_path / "legacy"), ttl_days=7)
    session_id = store.create_session().session_id
    csv_runtime = CSVSessionRuntime(base_dir=tmp_path / "duckdb", default_ttl_sec=3600)
    manifest_store = ManifestStore(tmp_path)
    blob_store = FakeBlobStore()
    service = TabularUploadService(
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        notebook_orchestrator=NotebookOrchestrator(NotebookStore(tmp_path)),
        storage_dir=tmp_path,
        blob_store=blob_store,  # type: ignore[arg-type]
    )
    service.ingest_files(
        session_id=session_id,
        user_id=7,
        files=[TabularUploadFile(file_name="orders.csv", content=blob_store.content)],
    )
    source = manifest_store.load(session_id).sources[0]
    (tmp_path / "sessions" / session_id / str(source.parquet_path)).unlink()
    csv_runtime.delete_session(session_id)

    CSVRuntimeStateService(
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        storage_dir=tmp_path,
        blob_store=blob_store,  # type: ignore[arg-type]
    ).ensure_csv_runtime(session_id=session_id)

    frame = csv_runtime.query_dataframe(session_id, "SELECT * FROM orders ORDER BY id")
    assert frame.to_dict(orient="records") == [{"id": 1, "amount": 10}, {"id": 2, "amount": 20}]


class _FailingNotebookOrchestrator(NotebookOrchestrator):
    def apply_batch(
        self,
        session_id: str,
        edits: list[NotebookEdit],
    ) -> list[EditResult]:
        notebook = self._store.load(session_id)
        return [EditResult(ok=False, notebook=notebook, error="simulated notebook failure")]


def test_ingest_rolls_back_duckdb_manifest_and_files_when_notebook_binding_fails(
    tmp_path: Path,
) -> None:
    store = SessionStore(str(tmp_path / "legacy"), ttl_days=7)
    session = store.create_session()
    csv_runtime = CSVSessionRuntime(base_dir=tmp_path / "duckdb", default_ttl_sec=3600)
    service = TabularUploadService(
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=ManifestStore(tmp_path),
        notebook_orchestrator=_FailingNotebookOrchestrator(NotebookStore(tmp_path)),
        storage_dir=tmp_path,
    )

    with pytest.raises(TabularUploadError, match="simulated notebook failure"):
        service.ingest_files(
            session_id=session.session_id,
            files=[
                TabularUploadFile(
                    file_name="orders.csv",
                    content=b"order_id,amount\n1,120\n",
                )
            ],
            ttl_seconds=3600,
        )

    manifest = ManifestStore(tmp_path).load(session.session_id)
    assert manifest.sources == []
    assert csv_runtime.list_tables(session.session_id) == []
    source_dir = tmp_path / "sessions" / session.session_id / "sources"
    assert not list(source_dir.glob("*.parquet")) if source_dir.exists() else True


def test_ingest_xlsx_file_registers_first_sheet_as_duckdb_table(tmp_path: Path) -> None:
    service, store, csv_runtime, session_id = _service(tmp_path)

    workbook = BytesIO()
    pd.DataFrame(
        {
            "sku": ["A", "B"],
            "price": [10.5, 20.0],
        }
    ).to_excel(workbook, index=False, sheet_name="prices")

    result = service.ingest_files(
        session_id=session_id,
        files=[
            TabularUploadFile(
                file_name="prices.xlsx",
                content=workbook.getvalue(),
            )
        ],
        ttl_seconds=3600,
    )

    assert [item.table_name for item in result.files] == ["prices"]
    queried = csv_runtime.query_dataframe(session_id, "SELECT sku, price FROM prices ORDER BY sku")
    assert queried.to_dict(orient="records") == [
        {"sku": "A", "price": 10.5},
        {"sku": "B", "price": 20.0},
    ]

    state = store.load_session(session_id)
    assert state is not None
    assert state.dataset_name == "prices.xlsx"


def test_csv_runtime_state_restores_all_manifest_tables_after_duckdb_eviction(tmp_path: Path) -> None:
    service, store, csv_runtime, session_id = _service(tmp_path)
    service.ingest_files(
        session_id=session_id,
        files=[
            TabularUploadFile(
                file_name="orders.csv",
                content=b"order_id,customer_id,amount\n1,10,120\n",
            ),
            TabularUploadFile(
                file_name="customers.csv",
                content=b"customer_id,segment\n10,enterprise\n",
            ),
        ],
        ttl_seconds=3600,
    )

    csv_runtime.delete_session(session_id)
    store.clear_csv_runtime_state(session_id)

    restored = CSVRuntimeStateService(
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=ManifestStore(tmp_path),
        storage_dir=tmp_path,
    ).ensure_csv_runtime(
        session_id=session_id,
        ttl_seconds=3600,
    )

    assert restored.csv_loaded is True
    assert restored.csv_session_id == session_id
    assert restored.csv_table_names == ["customers", "orders"]
    joined = csv_runtime.query_dataframe(
        session_id,
        "SELECT o.order_id, c.segment FROM orders o JOIN customers c USING (customer_id)",
    )
    assert joined.to_dict(orient="records") == [{"order_id": 1, "segment": "enterprise"}]


def test_refresh_catalog_uses_duckdb_tables_without_legacy_df_duplicate(tmp_path: Path) -> None:
    service, store, csv_runtime, session_id = _service(tmp_path)
    service.ingest_files(
        session_id=session_id,
        files=[
            TabularUploadFile(
                file_name="orders.csv",
                content=b"order_id,customer_id,amount\n1,10,120\n",
            ),
            TabularUploadFile(
                file_name="customers.csv",
                content=b"customer_id,segment\n10,enterprise\n",
            ),
        ],
        ttl_seconds=3600,
    )

    from backend.data_access.catalog_refresh import refresh_session_catalog

    snapshot = refresh_session_catalog(
        store,
        session_id,
        csv_runtime=csv_runtime,
    )
    state = store.load_session(session_id)

    assert [table.qualified_name for table in snapshot.tables] == ["customers", "orders"]
    assert all(table.source_kind == "csv_session" for table in snapshot.tables)
    assert state is not None
    assert snapshot.source_fingerprint == f"csv:{state.source_ref_id}"


def test_tabular_upload_hash_does_not_include_file_name(tmp_path: Path) -> None:
    service, store, _csv_runtime, session_id = _service(tmp_path)
    content = b"order_id,amount\n1,120\n"

    service.ingest_files(
        session_id=session_id,
        files=[
            TabularUploadFile(
                file_name="renamed.csv",
                content=content,
            )
        ],
        ttl_seconds=3600,
    )

    state = store.load_session(session_id)

    assert state is not None
    assert state.source_label == "renamed.csv"
    assert state.source_ref_id == f"sha256:{hashlib.sha256(content).hexdigest()}"


def test_sql_table_candidates_include_upload_source_metadata(tmp_path: Path) -> None:
    service, _store, csv_runtime, session_id = _service(tmp_path)
    service.ingest_files(
        session_id=session_id,
        files=[
            TabularUploadFile(
                file_name="orders.csv",
                content=b"order_id,customer_id,amount\n1,10,120\n",
            ),
            TabularUploadFile(
                file_name="customers.csv",
                content=b"customer_id,segment\n10,enterprise\n",
            ),
        ],
        ttl_seconds=3600,
    )

    sql_service = SQLTableService(
        csv_loaded=True,
        csv_session_id=session_id,
        storage_dir=tmp_path,
    )
    sql_service.csv_runtime = csv_runtime

    candidates = {candidate.table_name: candidate for candidate in sql_service.collect_candidates()}

    assert candidates["orders"].file_name == "orders.csv"
    assert candidates["orders"].display_name == "orders.csv"
    assert candidates["orders"].source_alias == "orders_csv"
    assert candidates["orders"].schema_hint["amount"]
    assert candidates["customers"].file_name == "customers.csv"


def test_joined_multi_file_upload_result_can_feed_plotly_tool(tmp_path: Path) -> None:
    service, _store, csv_runtime, session_id = _service(tmp_path)
    service.ingest_files(
        session_id=session_id,
        files=[
            TabularUploadFile(
                file_name="orders.csv",
                content=b"order_id,customer_id,amount\n1,10,120\n2,20,80\n3,10,30\n",
            ),
            TabularUploadFile(
                file_name="customers.csv",
                content=b"customer_id,segment\n10,enterprise\n20,retail\n",
            ),
        ],
        ttl_seconds=3600,
    )

    sql_service = SQLTableService(
        csv_loaded=True,
        csv_session_id=session_id,
        max_rows=50,
    )
    sql_service.csv_runtime = csv_runtime
    sandbox = SessionSandbox()
    sql_tool = SQLTool(
        csv_loaded=True,
        csv_session_id=session_id,
        max_rows=50,
        sandbox=sandbox,
    )
    sql_tool._service = sql_service
    join_sql = """
        SELECT c.segment, SUM(o.amount) AS total_amount
        FROM orders AS o
        JOIN customers AS c USING (customer_id)
        GROUP BY c.segment
        ORDER BY c.segment
    """

    _text, table_payload = sql_tool._run(
        mode="execute_sql",
        sql=join_sql,
        question="join orders and customers by customer_id",
        artifact_name="sales_by_segment",
    )

    joined = table_payload["items"]["sales_by_segment"]
    assert isinstance(joined, pd.DataFrame)
    assert joined.to_dict(orient="records") == [
        {"segment": "enterprise", "total_amount": 150.0},
        {"segment": "retail", "total_amount": 80.0},
    ]
    assert "sales_by_segment" in sandbox.get_user_scope()

    plotly_tool = PlotlyTool(
        pd.DataFrame(),
        execution_timeout_sec=20.0,
        tool_cache_size=0,
        sandbox=sandbox,
    )
    _plot_text, plot_payload = plotly_tool._run(
        """
fig = px.bar(sales_by_segment, x="segment", y="total_amount", title="Sales by segment")
tool_result = chart.result(fig, artifact_name="sales_by_segment_chart")
tool_result
"""
    )

    assert isinstance(plot_payload["plot"]["sales_by_segment_chart"], go.Figure)
