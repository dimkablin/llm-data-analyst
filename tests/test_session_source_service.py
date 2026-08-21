from __future__ import annotations

from pathlib import Path

from backend.auth.blob_store import BlobWrite
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.data_access.session_source_service import SessionSourceService
from backend.data_access.tabular_upload_service import TabularUploadFile, TabularUploadService
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.orchestrator import NotebookOrchestrator
from backend.notebook.session_source import SessionSource
from backend.notebook.store import NotebookStore
from backend.sessions.session_store import SessionStore


def _service_stack(tmp_path: Path):
    store = SessionStore(str(tmp_path / "sessions_state"), ttl_days=7)
    session = store.create_session()
    csv_runtime = CSVSessionRuntime(base_dir=tmp_path / "duckdb", default_ttl_sec=3600)
    manifest_store = ManifestStore(tmp_path)
    notebook_store = NotebookStore(tmp_path)
    orchestrator = NotebookOrchestrator(notebook_store)
    upload_service = TabularUploadService(
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        notebook_orchestrator=orchestrator,
        storage_dir=tmp_path,
    )
    source_service = SessionSourceService(
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        notebook_orchestrator=orchestrator,
        storage_dir=tmp_path,
    )
    return (
        store,
        csv_runtime,
        manifest_store,
        notebook_store,
        upload_service,
        source_service,
        session.session_id,
    )


def test_remove_csv_source_drops_duckdb_table_state_catalog_and_notebook_binding(
    tmp_path: Path,
) -> None:
    (
        store,
        csv_runtime,
        manifest_store,
        notebook_store,
        upload_service,
        source_service,
        session_id,
    ) = _service_stack(tmp_path)
    upload_service.ingest_files(
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

    removed = source_service.remove_source(session_id=session_id, alias="customers_csv")

    assert removed.alias == "customers_csv"
    assert [row["table_name"] for row in csv_runtime.list_tables(session_id)] == ["orders"]
    state = store.load_session(session_id)
    assert state is not None
    assert state.csv_loaded is True
    assert state.csv_table_names == ["orders"]
    manifest = manifest_store.load(session_id)
    assert [source.alias for source in manifest.sources] == ["orders_csv"]
    notebook = notebook_store.load(session_id)
    assert [cell.metadata.source_alias for cell in notebook.source_binding_cells] == [
        "orders_csv"
    ]
    catalog = store.load_data_catalog(session_id)
    assert catalog is not None
    assert [table.qualified_name for table in catalog.tables] == ["orders"]


def test_remove_last_csv_source_clears_active_csv_runtime_state(tmp_path: Path) -> None:
    store, csv_runtime, _manifest_store, _notebook_store, upload_service, source_service, session_id = (
        _service_stack(tmp_path)
    )
    upload_service.ingest_files(
        session_id=session_id,
        files=[
            TabularUploadFile(
                file_name="orders.csv",
                content=b"order_id,amount\n1,120\n",
            )
        ],
        ttl_seconds=3600,
    )

    source_service.remove_source(session_id=session_id, alias="orders_csv")

    state = store.load_session(session_id)
    assert state is not None
    assert state.source_type is None
    assert state.dataset_name is None
    assert state.df_path is None
    assert state.csv_loaded is False
    assert state.csv_session_id is None
    assert state.csv_table_names == []
    assert csv_runtime.list_tables(session_id) == []
    catalog = store.load_data_catalog(session_id)
    assert catalog is not None
    assert catalog.tables == []


def test_remove_source_deletes_its_durable_blobs(tmp_path: Path) -> None:
    class BlobStore:
        def __init__(self) -> None:
            self.deleted_ids: list[str] = []
            self.deleted_kinds: list[str] = []

        def put_many(self, *, items: list[BlobWrite], **_kwargs) -> list[str]:
            return [f"blob-{index}" for index, _item in enumerate(items)]

        def delete_many(self, *, blob_ids: list[str], **_kwargs) -> None:
            self.deleted_ids.extend(blob_ids)

        def delete_for_session(self, *, kinds: list[str], **_kwargs) -> None:
            self.deleted_kinds.extend(kinds)

    store, csv_runtime, manifest_store, notebook_store, _upload, _source, session_id = (
        _service_stack(tmp_path)
    )
    orchestrator = NotebookOrchestrator(notebook_store)
    blob_store = BlobStore()
    upload = TabularUploadService(
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        notebook_orchestrator=orchestrator,
        storage_dir=tmp_path,
        blob_store=blob_store,  # type: ignore[arg-type]
    )
    source_service = SessionSourceService(
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        notebook_orchestrator=orchestrator,
        storage_dir=tmp_path,
        blob_store=blob_store,  # type: ignore[arg-type]
        user_id=7,
    )
    upload.ingest_files(
        session_id=session_id,
        user_id=7,
        files=[TabularUploadFile(file_name="orders.csv", content=b"order_id\n1\n")],
    )

    source_service.remove_source(
        session_id=session_id,
        alias="orders_csv",
        refresh_catalog=False,
    )

    assert blob_store.deleted_ids == ["blob-0"]

    manifest = manifest_store.load(session_id)
    manifest.add_source(SessionSource(alias="planfact", source_type="planfact"))
    manifest_store.save(session_id, manifest)
    store.set_source(
        session_id,
        source_type="planfact",
        source_ref_id="planfact",
        source_label="Plan/fact",
        source_mode="duckdb",
    )
    source_service.remove_source(
        session_id=session_id,
        alias="planfact",
        refresh_catalog=False,
    )

    assert set(blob_store.deleted_kinds) == {
        "planfact_plan",
        "planfact_fact",
        "planfact_mapping",
        "planfact_config",
        "runtime_snapshot",
    }
