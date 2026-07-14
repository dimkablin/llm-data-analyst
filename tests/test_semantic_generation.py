from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from backend.data_access.data_catalog import CatalogColumn, CatalogTable, DataCatalogSnapshot
from backend.data_access.semantic_catalog_service import SemanticCatalogService
from backend.data_access.semantic_generation_service import (
    GeneratedMetricDraft,
    GeneratedTablePatch,
    GeneratedTermDraft,
    SemanticCatalogGenerationService,
    SemanticGenerationDraft,
)
from backend.data_access.semantic_catalog_store import (
    SemanticCatalogFileStore,
    _document_version,
    semantic_catalog_store_from_settings,
)
from backend.data_access.semantic_models import SemanticCatalog, SemanticCatalogOverlay
from backend.data_access.semantic_models import SemanticSearchResultItem
from backend.data_access.semantic_vector_store import LocalHashEmbeddings, SemanticVectorStore
from backend.sessions.session_store import SessionStore


class _VectorStore:
    @property
    def enabled(self) -> bool:
        return False

    def upsert_catalog(self, catalog) -> None:
        _ = catalog

    def search(self, *, catalog, query: str, top_k: int):
        _ = catalog, query, top_k
        return list[SemanticSearchResultItem]()


class _RuntimeService:
    def get_runtime_config(self, *, user_id: int, connection_id: str):
        return SimpleNamespace(
            user_id=user_id,
            connection_id=connection_id,
            db_type="postgresql",
        )


class _Helper:
    def __init__(self, *, runtime, timeout_sec: float) -> None:
        self.runtime = runtime
        self.timeout_sec = timeout_sec

    def preview_table(self, table: str, *, schema: str | None = None, limit: int = 5):
        _ = schema, limit
        if table == "customers":
            return pd.DataFrame([{"customer_id": 1, "email": "demo@example.com"}])
        return pd.DataFrame([{"order_id": 10, "customer_id": 1, "amount": 120}])

    def list_effective_relationships(self):
        return [
            {
                "from_schema": "public",
                "from_table": "orders",
                "from_column": "customer_id",
                "to_schema": "public",
                "to_table": "customers",
                "to_column": "customer_id",
            }
        ]


def _snapshot() -> DataCatalogSnapshot:
    return DataCatalogSnapshot(
        source_fingerprint="db:conn-1",
        tables=[
            CatalogTable(
                qualified_name="public.orders",
                table_name="orders",
                source_kind="db",
                schema="public",
                columns=[
                    CatalogColumn(name="order_id", dtype="integer"),
                    CatalogColumn(name="customer_id", dtype="integer"),
                    CatalogColumn(name="amount", dtype="numeric"),
                ],
            ),
            CatalogTable(
                qualified_name="public.customers",
                table_name="customers",
                source_kind="db",
                schema="public",
                columns=[
                    CatalogColumn(name="customer_id", dtype="integer"),
                    CatalogColumn(name="email", dtype="text"),
                ],
            ),
        ],
    )


def test_ai_generation_applies_valid_overlay_and_fk_relationship(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "backend.data_access.semantic_generation_service.refresh_session_catalog",
        lambda *args, **kwargs: _snapshot(),
    )
    monkeypatch.setattr(
        "backend.data_access.semantic_generation_service.DBAnalyticsHelper",
        _Helper,
    )
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.bind_db_connection_source(
        state.session_id,
        connection_id="conn-1",
        label="Warehouse",
    )
    store.save_data_catalog(state.session_id, _snapshot())
    catalog_service = SemanticCatalogService(store=store, vector_store=_VectorStore())

    def fake_llm(payload):
        assert payload["samples"]["public.customers"][0]["email"] == "<email>"
        return SemanticGenerationDraft(
            tables=[
                GeneratedTablePatch(
                    table="public.orders",
                    description="Customer orders.",
                    semantic_role="fact",
                )
            ],
            metrics=[
                GeneratedMetricDraft(
                    key="gross_sales",
                    name="Gross sales",
                    base_table="public.orders",
                    expr="amount",
                    agg="sum",
                )
            ],
            terms=[
                GeneratedTermDraft(
                    name="Revenue",
                    description="Total order amount.",
                    synonyms=["sales"],
                )
            ],
        )

    generator = SemanticCatalogGenerationService(
        store=store,
        catalog_service=catalog_service,
        db_runtime_service=_RuntimeService(),  # type: ignore[arg-type]
        settings=SimpleNamespace(),
        llm_generate=fake_llm,
    )

    result = generator.generate(session_id=state.session_id, user_id=7)

    assert result.summary.metrics_added == 1
    assert result.summary.terms_added == 1
    assert result.summary.relationships_added == 1
    assert result.summary.rejected_items == []
    assert any(metric.key == "gross_sales" for metric in result.catalog.metrics)
    assert any(rel.from_table == "public.orders" for rel in result.catalog.relationships)
    assert next(table for table in result.catalog.tables if table.qualified_name == "public.orders").description == "Customer orders."


def test_ai_generation_supports_uploaded_csv(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.bind_csv_source(state.session_id, filename="sales.csv")
    store.save_data_catalog(
        state.session_id,
        DataCatalogSnapshot(
            source_fingerprint="csv:sales",
            tables=[
                CatalogTable(
                    qualified_name="sales",
                    table_name="sales",
                    source_kind="csv_session",
                    columns=[
                        CatalogColumn(name="region", dtype="string", examples=["North"]),
                        CatalogColumn(name="amount", dtype="float64", examples=["120"]),
                    ],
                )
            ],
        ),
    )

    def fake_llm(payload):
        assert payload["db_type"] == "csv_duckdb"
        assert payload["samples"]["sales"][0] == {"region": "North", "amount": "120"}
        return SemanticGenerationDraft(
            metrics=[
                GeneratedMetricDraft(
                    key="csv_revenue",
                    name="Revenue",
                    base_table="sales",
                    expr="amount",
                    agg="sum",
                )
            ]
        )

    generator = SemanticCatalogGenerationService(
        store=store,
        catalog_service=SemanticCatalogService(store=store, vector_store=_VectorStore()),
        db_runtime_service=_RuntimeService(),  # type: ignore[arg-type]
        settings=SimpleNamespace(),
        llm_generate=fake_llm,
    )

    result = generator.generate(session_id=state.session_id, user_id=7)

    assert result.summary.metrics_added == 1
    assert any(metric.key == "csv_revenue" for metric in result.catalog.metrics)


def test_semantic_catalog_store_factory_defaults_to_file(tmp_path: Path) -> None:
    settings = SimpleNamespace(semantic_catalog_store="file")

    store = semantic_catalog_store_from_settings(settings, tmp_path)

    assert isinstance(store, SemanticCatalogFileStore)


def test_document_version_ignores_semantic_catalog_string_version() -> None:
    catalog = SemanticCatalog(
        catalog_id="cat",
        user_id=0,
        session_id="",
        source_key="source",
        version="2.0",
        published_version=3,
    )
    overlay = SemanticCatalogOverlay(source_key="source", version=4)

    assert _document_version(catalog) == 3
    assert _document_version(overlay) == 4


def test_local_embeddings_enable_qdrant_without_external_credentials() -> None:
    store = SemanticVectorStore.from_settings(
        SimpleNamespace(
            semantic_vector_enabled=True,
            semantic_qdrant_url="http://qdrant:6333",
            semantic_qdrant_collection="semantic_catalog_chunks",
            semantic_embedding_provider="local",
            semantic_embedding_dim=32,
        )
    )

    assert store.enabled is True
    embeddings = store._embeddings()
    assert isinstance(embeddings, LocalHashEmbeddings)
    assert embeddings.embed_query("revenue sales") == embeddings.embed_query("revenue sales")
    assert len(embeddings.embed_query("revenue")) == 32
