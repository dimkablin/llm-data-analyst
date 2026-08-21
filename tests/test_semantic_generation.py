from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.data_access.data_catalog import CatalogColumn, CatalogTable, DataCatalogSnapshot
from backend.data_access.semantic_catalog_service import SemanticCatalogService
from backend.data_access.semantic_catalog_store import (
    SemanticCatalogPostgresStore,
    _document_version,
    semantic_catalog_store_from_settings,
)
from backend.data_access.semantic_generation_service import (
    GeneratedColumnPatch,
    GeneratedMetricDraft,
    GeneratedTablePatch,
    GeneratedTermDraft,
    SemanticCatalogGenerationRequest,
    SemanticCatalogGenerationService,
    SemanticGenerationDraft,
)
from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticCatalogOverlay,
    SemanticMetricCreate,
    SemanticSearchResultItem,
    SemanticTerm,
)
from backend.data_access.semantic_vector_store import LocalHashEmbeddings, SemanticVectorStore
from tests.in_memory_semantic_store import SemanticSessionStore as SessionStore


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
    assert (
        next(table for table in result.catalog.tables if table.qualified_name == "public.orders").description
        == "Customer orders."
    )


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


def test_ai_generation_preserves_explicit_metric_aggregation(
    tmp_path: Path,
) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.bind_csv_source(state.session_id, filename="survey.csv")
    store.save_data_catalog(
        state.session_id,
        DataCatalogSnapshot(
            source_fingerprint="csv:survey",
            tables=[
                CatalogTable(
                    qualified_name="customer_survey",
                    table_name="customer_survey",
                    source_kind="csv_session",
                    columns=[
                        CatalogColumn(name="safety_score", dtype="float64"),
                        CatalogColumn(name="response_count", dtype="integer"),
                    ],
                )
            ],
        ),
    )

    def fake_llm(_payload):
        return SemanticGenerationDraft(
            tables=[
                GeneratedTablePatch(
                    table="customer_survey",
                    description="Customer satisfaction survey",
                    semantic_role="fact",
                )
            ],
            metrics=[
                GeneratedMetricDraft(
                    key="safety_score",
                    name="Safety score",
                    base_table="customer_survey",
                    expr="safety_score",
                    agg="sum",
                ),
                GeneratedMetricDraft(
                    key="responses",
                    name="Response count",
                    base_table="customer_survey",
                    expr="response_count",
                    agg="sum",
                ),
            ],
        )

    generator = SemanticCatalogGenerationService(
        store=store,
        catalog_service=SemanticCatalogService(store=store, vector_store=_VectorStore()),
        db_runtime_service=_RuntimeService(),  # type: ignore[arg-type]
        settings=SimpleNamespace(),
        llm_generate=fake_llm,
    )

    result = generator.generate(session_id=state.session_id, user_id=7)
    metrics = {metric.key: metric for metric in result.catalog.metrics}

    assert metrics["safety_score"].agg == "sum"
    assert metrics["responses"].agg == "sum"


def test_ai_generation_can_complete_missing_metrics_for_one_click_fill(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.bind_csv_source(state.session_id, filename="sales.csv")
    store.save_data_catalog(
        state.session_id,
        DataCatalogSnapshot(
            source_fingerprint="csv:metric-completion",
            tables=[
                CatalogTable(
                    qualified_name="sales",
                    table_name="sales",
                    source_kind="csv_session",
                    columns=[CatalogColumn(name="custom_numeric", dtype="float64")],
                )
            ],
        ),
    )
    tasks: list[str] = []

    def fake_llm(payload):
        tasks.append(str(payload.get("task") or "general"))
        if payload.get("task"):
            return SemanticGenerationDraft(
                metrics=[
                    GeneratedMetricDraft(
                        key="total_amount",
                        name="Total amount",
                        base_table="sales",
                        expr="custom_numeric",
                        agg="sum",
                    )
                ]
            )
        return SemanticGenerationDraft(tables=[GeneratedTablePatch(table="sales", description="Sales facts")])

    generator = SemanticCatalogGenerationService(
        store=store,
        catalog_service=SemanticCatalogService(store=store, vector_store=_VectorStore()),
        db_runtime_service=_RuntimeService(),  # type: ignore[arg-type]
        settings=SimpleNamespace(semantic_generation_batch_tables=2),
        llm_generate=fake_llm,
    )

    result = generator.generate(
        session_id=state.session_id,
        user_id=7,
        request=SemanticCatalogGenerationRequest(
            sample_rows=0,
            max_tables=1,
            ensure_metrics=True,
        ),
    )

    assert tasks == [
        "general",
        "metrics_only: create 1-3 conservative simple business metrics for every "
        "provided table that has clear numeric measures; return all other arrays empty",
    ]
    assert result.summary.metrics_added == 1
    assert any(metric.key == "total_amount" for metric in result.catalog.metrics)


def test_one_click_fill_skips_ambiguous_metric_fallbacks(
    tmp_path: Path,
) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.bind_csv_source(state.session_id, filename="survey.csv")
    store.save_data_catalog(
        state.session_id,
        DataCatalogSnapshot(
            source_fingerprint="csv:fallback-metrics",
            tables=[
                CatalogTable(
                    qualified_name="customer_survey",
                    table_name="customer_survey",
                    source_kind="csv_session",
                    columns=[
                        CatalogColumn(name="survey_date", dtype="date"),
                        CatalogColumn(name="region", dtype="string"),
                        CatalogColumn(name="safety_score", dtype="float64"),
                        CatalogColumn(name="response_count", dtype="integer"),
                        CatalogColumn(name="opaque_numeric", dtype="float64"),
                        CatalogColumn(name="безопасность", dtype="float64"),
                    ],
                )
            ],
        ),
    )
    catalog_service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    statuses_seen_by_llm: list[str] = []

    def empty_llm(payload):
        current = catalog_service.load_for_session(session_id=state.session_id, user_id=7)
        assert current is not None
        statuses_seen_by_llm.append(current.status)
        if payload.get("task"):
            return SemanticGenerationDraft()
        return SemanticGenerationDraft(
            tables=[
                GeneratedTablePatch(
                    table="customer_survey",
                    description="Customer satisfaction survey",
                    semantic_role="fact",
                )
            ]
        )

    generator = SemanticCatalogGenerationService(
        store=store,
        catalog_service=catalog_service,
        db_runtime_service=_RuntimeService(),  # type: ignore[arg-type]
        settings=SimpleNamespace(semantic_generation_batch_tables=2),
        llm_generate=empty_llm,
    )

    result = generator.generate(
        session_id=state.session_id,
        user_id=7,
        request=SemanticCatalogGenerationRequest(
            sample_rows=0,
            max_tables=1,
            ensure_metrics=True,
        ),
    )
    assert statuses_seen_by_llm == ["ready", "ready"]
    assert result.summary.metrics_added == 0
    assert result.catalog.metrics == []
    assert result.catalog.status == "ready"


def test_generation_repairs_non_latin_metrics_terms_and_technical_dimensions(
    tmp_path: Path,
) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.bind_csv_source(state.session_id, filename="ratings.csv")
    store.save_data_catalog(
        state.session_id,
        DataCatalogSnapshot(
            source_fingerprint="csv:non-latin-metric",
            tables=[
                CatalogTable(
                    qualified_name="ratings",
                    table_name="ratings",
                    source_kind="csv_session",
                    columns=[
                        CatalogColumn(name="month", dtype="date"),
                        CatalogColumn(name="region", dtype="string"),
                        CatalogColumn(name="metric", dtype="string"),
                        CatalogColumn(name="is_deleted", dtype="boolean"),
                        CatalogColumn(name="безопасность", dtype="float64"),
                    ],
                )
            ],
        ),
    )

    def fake_llm(_payload):
        return SemanticGenerationDraft(
            tables=[
                GeneratedTablePatch(
                    table="ratings",
                    description="Passenger satisfaction ratings",
                    semantic_role="fact",
                )
            ],
            columns=[
                GeneratedColumnPatch(
                    table="ratings",
                    column="безопасность",
                    description="Average passenger safety score",
                    semantic_role="dimension",
                )
            ],
            metrics=[
                GeneratedMetricDraft(
                    key="индекс_безопасности",
                    name="Индекс безопасности",
                    base_table="ratings",
                    expr="безопасность",
                    agg="avg",
                    default_time_dimension="month",
                    allowed_dimensions=[
                        "region",
                        "metric",
                        "is_deleted",
                        "безопасность",
                    ],
                )
            ],
            terms=[
                GeneratedTermDraft(
                    name="safety_rating",
                    synonyms=["safety_score"],
                    entity_refs=["ratings"],
                ),
                GeneratedTermDraft(
                    name="Safety Score",
                    synonyms=["safety_rating"],
                    entity_refs=["ratings.безопасность"],
                ),
            ],
        )

    result = SemanticCatalogGenerationService(
        store=store,
        catalog_service=SemanticCatalogService(store=store, vector_store=_VectorStore()),
        db_runtime_service=_RuntimeService(),  # type: ignore[arg-type]
        settings=SimpleNamespace(semantic_generation_batch_tables=2),
        llm_generate=fake_llm,
    ).generate(
        session_id=state.session_id,
        user_id=7,
        request=SemanticCatalogGenerationRequest(
            sample_rows=0,
            max_tables=1,
            ensure_metrics=True,
        ),
    )

    assert len(result.catalog.metrics) == 1
    metric = result.catalog.metrics[0]
    assert metric.key.isascii()
    assert metric.key.startswith("measure_")
    assert metric.metric_id == f"metric:{metric.key}"
    assert metric.allowed_dimensions == ["region"]
    safety = next(column for column in result.catalog.columns if column.name == "безопасность")
    assert safety.semantic_role == "metric_candidate"
    matching_terms = [term for term in result.catalog.terms if term.name in {"safety_rating", "Safety Score"}]
    assert len(matching_terms) == 1
    assert matching_terms[0].name == "Safety Score"
    assert set(matching_terms[0].entity_refs) == {"ratings", "ratings.безопасность"}


def test_invalid_generated_metric_does_not_create_unverified_fallback(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.bind_csv_source(state.session_id, filename="ratings.csv")
    store.save_data_catalog(
        state.session_id,
        DataCatalogSnapshot(
            source_fingerprint="csv:invalid-generated-metric",
            tables=[
                CatalogTable(
                    qualified_name="ratings",
                    table_name="ratings",
                    source_kind="csv_session",
                    columns=[
                        CatalogColumn(name="month", dtype="date"),
                        CatalogColumn(name="region", dtype="string"),
                        CatalogColumn(name="safety_score", dtype="float64"),
                    ],
                )
            ],
        ),
    )

    def fake_llm(_payload):
        return SemanticGenerationDraft(
            tables=[
                GeneratedTablePatch(
                    table="ratings",
                    description="Passenger safety ratings",
                    semantic_role="fact",
                )
            ],
            metrics=[
                GeneratedMetricDraft(
                    key="broken_metric",
                    name="Broken metric",
                    base_table="ratings",
                    expr="missing_column",
                    agg="sum",
                )
            ],
        )

    result = SemanticCatalogGenerationService(
        store=store,
        catalog_service=SemanticCatalogService(store=store, vector_store=_VectorStore()),
        db_runtime_service=_RuntimeService(),  # type: ignore[arg-type]
        settings=SimpleNamespace(semantic_generation_batch_tables=2),
        llm_generate=fake_llm,
    ).generate(
        session_id=state.session_id,
        user_id=7,
        request=SemanticCatalogGenerationRequest(
            sample_rows=0,
            max_tables=1,
            ensure_metrics=True,
        ),
    )

    assert result.catalog.metrics == []
    assert any("missing_column" in item for item in result.summary.rejected_items)


def test_ai_generation_namespaces_duplicate_metric_keys_across_tables(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.bind_csv_source(state.session_id, filename="multi.csv")
    store.save_data_catalog(
        state.session_id,
        DataCatalogSnapshot(
            source_fingerprint="csv:duplicate-metrics",
            tables=[
                CatalogTable(
                    qualified_name=table,
                    table_name=table,
                    source_kind="csv_session",
                    columns=[CatalogColumn(name="custom_numeric", dtype="float64")],
                )
                for table in ("table_a", "table_b")
            ],
        ),
    )

    def fake_llm(payload):
        table = payload["tables"][0]["qualified_name"]
        return SemanticGenerationDraft(
            metrics=[
                GeneratedMetricDraft(
                    key="total",
                    name="Total",
                    base_table=table,
                    expr="custom_numeric",
                    agg="sum",
                )
            ]
        )

    generator = SemanticCatalogGenerationService(
        store=store,
        catalog_service=SemanticCatalogService(store=store, vector_store=_VectorStore()),
        db_runtime_service=_RuntimeService(),  # type: ignore[arg-type]
        settings=SimpleNamespace(semantic_generation_batch_tables=1),
        llm_generate=fake_llm,
    )

    result = generator.generate(session_id=state.session_id, user_id=7)

    assert {metric.key for metric in result.catalog.metrics} == {
        "table_a_total",
        "table_b_total",
    }


def test_ai_generation_does_not_overwrite_existing_manual_metric(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.bind_csv_source(state.session_id, filename="manual.csv")
    store.save_data_catalog(
        state.session_id,
        DataCatalogSnapshot(
            source_fingerprint="csv:manual-metric",
            tables=[
                CatalogTable(
                    qualified_name="survey",
                    table_name="survey",
                    source_kind="csv_session",
                    columns=[CatalogColumn(name="score", dtype="float64")],
                )
            ],
        ),
    )
    catalog_service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    catalog_service.refresh(session_id=state.session_id, user_id=7)
    catalog_service.create_metric(
        session_id=state.session_id,
        user_id=7,
        payload=SemanticMetricCreate(
            key="quality_score",
            name="Manually verified quality score",
            type="simple",
            base_table="survey",
            expr="score",
            agg="avg",
        ),
    )

    def fake_llm(_payload):
        return SemanticGenerationDraft(
            metrics=[
                GeneratedMetricDraft(
                    key="quality_score",
                    name="Generated score",
                    base_table="survey",
                    expr="score",
                    agg="sum",
                )
            ]
        )

    generator = SemanticCatalogGenerationService(
        store=store,
        catalog_service=catalog_service,
        db_runtime_service=_RuntimeService(),  # type: ignore[arg-type]
        settings=SimpleNamespace(),
        llm_generate=fake_llm,
    )

    result = generator.generate(session_id=state.session_id, user_id=7)
    metric = next(item for item in result.catalog.metrics if item.key == "quality_score")

    assert metric.name == "Manually verified quality score"
    assert metric.agg == "avg"


def test_ai_generation_batches_tables_and_applies_once(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.bind_csv_source(state.session_id, filename="batch.csv")
    store.save_data_catalog(
        state.session_id,
        DataCatalogSnapshot(
            source_fingerprint="csv:batch",
            tables=[
                CatalogTable(
                    qualified_name=f"table_{index}",
                    table_name=f"table_{index}",
                    source_kind="csv_session",
                    columns=[CatalogColumn(name="value", dtype="float64")],
                )
                for index in range(5)
            ],
        ),
    )
    batch_names: list[list[str]] = []

    def fake_llm(payload):
        names = [table["qualified_name"] for table in payload["tables"]]
        batch_names.append(names)
        return SemanticGenerationDraft(
            tables=[
                GeneratedTablePatch(
                    table=name,
                    description=f"Description for {name}",
                    semantic_role="fact",
                )
                for name in names
            ]
        )

    generator = SemanticCatalogGenerationService(
        store=store,
        catalog_service=SemanticCatalogService(store=store, vector_store=_VectorStore()),
        db_runtime_service=_RuntimeService(),  # type: ignore[arg-type]
        settings=SimpleNamespace(semantic_generation_batch_tables=2),
        llm_generate=fake_llm,
    )

    result = generator.generate(session_id=state.session_id, user_id=7)

    assert batch_names == [["table_0", "table_1"], ["table_2", "table_3"], ["table_4"]]
    assert result.summary.tables_scanned == 5
    assert result.summary.table_patches == 5
    assert all(table.description for table in result.catalog.tables)


def test_ai_generation_splits_batch_after_output_length_error(tmp_path: Path) -> None:
    class LengthFinishReasonError(RuntimeError):
        pass

    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.bind_csv_source(state.session_id, filename="wide.csv")
    store.save_data_catalog(
        state.session_id,
        DataCatalogSnapshot(
            source_fingerprint="csv:wide",
            tables=[
                CatalogTable(
                    qualified_name=f"wide_{index}",
                    table_name=f"wide_{index}",
                    source_kind="csv_session",
                    columns=[CatalogColumn(name=f"value_{column}", dtype="float64") for column in range(20)],
                )
                for index in range(4)
            ],
        ),
    )
    calls: list[list[str]] = []

    def fake_llm(payload):
        names = [table["qualified_name"] for table in payload["tables"]]
        calls.append(names)
        assert payload["response_limits"]["column_patches"] == 16
        if len(names) > 1:
            raise LengthFinishReasonError("length limit was reached")
        return SemanticGenerationDraft(
            tables=[GeneratedTablePatch(table=names[0], description="Generated separately")]
        )

    generator = SemanticCatalogGenerationService(
        store=store,
        catalog_service=SemanticCatalogService(store=store, vector_store=_VectorStore()),
        db_runtime_service=_RuntimeService(),  # type: ignore[arg-type]
        settings=SimpleNamespace(semantic_generation_batch_tables=2),
        llm_generate=fake_llm,
    )

    result = generator.generate(session_id=state.session_id, user_id=7)

    assert calls == [
        ["wide_0", "wide_1"],
        ["wide_0"],
        ["wide_1"],
        ["wide_2", "wide_3"],
        ["wide_2"],
        ["wide_3"],
    ]
    assert result.summary.tables_scanned == 4
    assert result.summary.table_patches == 4
    assert all(table.description == "Generated separately" for table in result.catalog.tables)


def test_ai_generation_does_not_apply_partial_batches(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.bind_csv_source(state.session_id, filename="batch.csv")
    store.save_data_catalog(
        state.session_id,
        DataCatalogSnapshot(
            source_fingerprint="csv:atomic-batch",
            tables=[
                CatalogTable(
                    qualified_name=f"table_{index}",
                    table_name=f"table_{index}",
                    source_kind="csv_session",
                    columns=[CatalogColumn(name="value", dtype="float64")],
                )
                for index in range(3)
            ],
        ),
    )
    catalog_service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    calls = 0

    def failing_llm(payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("batch generation failed")
        return SemanticGenerationDraft(
            tables=[
                GeneratedTablePatch(
                    table=table["qualified_name"],
                    description="Must not be applied",
                )
                for table in payload["tables"]
            ]
        )

    generator = SemanticCatalogGenerationService(
        store=store,
        catalog_service=catalog_service,
        db_runtime_service=_RuntimeService(),  # type: ignore[arg-type]
        settings=SimpleNamespace(semantic_generation_batch_tables=2),
        llm_generate=failing_llm,
    )

    with pytest.raises(RuntimeError, match="batch generation failed"):
        generator.generate(session_id=state.session_id, user_id=7)

    catalog = catalog_service.load_for_session(session_id=state.session_id, user_id=7)
    assert catalog is not None
    assert all(not table.description for table in catalog.tables)


def test_semantic_catalog_store_factory_uses_configured_postgres_schema() -> None:
    settings = SimpleNamespace(
        semantic_catalog_postgres_dsn="postgresql://example/catalog",
        semantic_catalog_postgres_schema="dev_semantic_layer",
    )

    store = semantic_catalog_store_from_settings(settings)

    assert isinstance(store, SemanticCatalogPostgresStore)
    assert store.schema == "dev_semantic_layer"


def test_semantic_catalog_store_factory_rejects_missing_postgres_dsn() -> None:
    settings = SimpleNamespace(semantic_catalog_postgres_dsn="")

    with pytest.raises(ValueError, match="SEMANTIC_METADATA_DATABASE_URL"):
        semantic_catalog_store_from_settings(settings)


def test_postgres_store_keeps_generated_separate_from_published(monkeypatch) -> None:
    store = SemanticCatalogPostgresStore("postgresql://metadata")
    catalog = SemanticCatalog(catalog_id="catalog", source_key="source", source_fingerprint="fp")
    saved_documents: list[tuple[str, str, dict]] = []
    saved_published: list[SemanticCatalog] = []

    monkeypatch.setattr(
        store,
        "_save",
        lambda doc_type, source_key, payload, _model: saved_documents.append((doc_type, source_key, payload)),
    )
    monkeypatch.setattr(store, "_save_catalog", saved_published.append)
    monkeypatch.setattr(store, "_load", lambda _doc_type, _source_key: saved_documents[0][2])

    store.save_generated(catalog)
    store.save_published(catalog.model_copy(update={"status": "ready"}))

    assert saved_documents[0][:2] == ("generated_catalog", "source")
    assert store.load_generated("source") == catalog
    assert saved_published[0].status == "ready"


def test_postgres_store_operation_results_share_one_transaction_connection(monkeypatch) -> None:
    import psycopg

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def transaction(self):
            return self

        def execute(self, _query, _params=None):
            return self

        def fetchone(self):
            return ("source",)

    store = SemanticCatalogPostgresStore("postgresql://metadata")
    connection = Connection()
    calls: list[tuple[str, Connection | None]] = []
    monkeypatch.setattr(psycopg, "connect", lambda _dsn: connection)
    monkeypatch.setattr(store, "_configure_connection", lambda _conn: None)
    monkeypatch.setattr(store, "_ensure_schema", lambda _conn: None)
    monkeypatch.setattr(
        store,
        "_save",
        lambda doc_type, _source_key, _payload, _model, *, conn=None: calls.append((doc_type, conn)),
    )
    monkeypatch.setattr(
        store,
        "_save_catalog",
        lambda _catalog, *, conn=None: calls.append(("published", conn)),
    )
    catalog = SemanticCatalog(catalog_id="catalog", source_key="source")
    overlay = SemanticCatalogOverlay(source_key="source", version=1)

    assert store.save_build_result_if_current(
        operation_id=1,
        generated=catalog,
        published=catalog,
    )
    assert store.save_generation_result_if_current(
        operation_id=2,
        overlay=overlay,
        published=catalog,
    )

    assert calls == [
        ("generated_catalog", connection),
        ("published", connection),
        ("overlay", connection),
        ("published", connection),
    ]


def test_postgres_store_keeps_session_profile_in_metadata_documents(monkeypatch) -> None:
    store = SemanticCatalogPostgresStore("postgresql://metadata")
    snapshot = DataCatalogSnapshot(source_fingerprint="csv:abc")
    saved: list[tuple[str, str, dict]] = []

    monkeypatch.setattr(
        store,
        "_save",
        lambda doc_type, source_key, payload, _model: saved.append((doc_type, source_key, payload)),
    )
    monkeypatch.setattr(store, "_load", lambda _doc_type, _source_key: saved[0][2])

    store.save_data_profile("session-1", snapshot)

    assert saved[0][:2] == ("data_profile", "session:session-1")
    assert store.load_data_profile("session-1") == snapshot


def test_semantic_catalog_postgres_store_rejects_unsafe_schema() -> None:
    with pytest.raises(ValueError, match="valid PostgreSQL identifier"):
        SemanticCatalogPostgresStore(
            "postgresql://example/catalog",
            schema="dev_semantic_layer; DROP SCHEMA public",
        )


def test_postgres_store_serializes_schema_initialization() -> None:
    class Connection:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []

        def execute(self, query, params=None):
            self.calls.append((query, params))

    store = SemanticCatalogPostgresStore("postgresql://example/catalog", schema="semantic")
    connection = Connection()

    store._configure_connection(connection)

    assert connection.calls[0] == (
        "SELECT pg_advisory_xact_lock(hashtext(%s))",
        ("semantic_catalog_schema:semantic",),
    )


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


def test_openai_compatible_embeddings_leave_token_limits_to_custom_endpoint() -> None:
    store = SemanticVectorStore(
        url="http://qdrant:6333",
        collection="semantic_catalog_chunks",
        vector_enabled=True,
        embedding_provider="openai",
        embedding_model="custom-embedding-model",
        embedding_base_url="http://embeddings:8000/v1",
        embedding_api_key="test",
    )

    assert store._embeddings().check_embedding_ctx_length is False


def test_local_embeddings_match_inflected_metric_names() -> None:
    embeddings = LocalHashEmbeddings(dimension=1536)
    query = embeddings.embed_query("покажи месячную метрику успеха")
    relevant = embeddings.embed_query("Месячная метрика успеха ФПК")
    unrelated = embeddings.embed_query("Индекс удовлетворенности пассажиров")

    def similarity(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=True))

    assert similarity(query, relevant) > similarity(query, unrelated)


def test_vector_search_returns_one_global_ranked_top_k() -> None:
    general_hit = SimpleNamespace(
        payload={"entity_type": "term", "entity_id": "term:service"},
        score=0.9,
    )
    metric_hit = SimpleNamespace(
        payload={"entity_type": "metric", "entity_id": "metric:service_index"},
        score=0.8,
    )

    class FakeClient:
        calls = 0

        def search(self, **kwargs):
            self.calls += 1
            return [metric_hit, general_hit]

    store = SemanticVectorStore(
        url="http://qdrant:6333",
        collection="semantic_catalog_chunks",
        vector_enabled=True,
        embedding_provider="local",
    )
    store._client = FakeClient()
    store._embeddings_client = SimpleNamespace(embed_query=lambda _query: [1.0])
    filter_args: dict[str, object] = {}

    def capture_filter(**kwargs):
        filter_args.update(kwargs)
        return kwargs.get("entity_type")

    store._filter = capture_filter  # type: ignore[method-assign]
    catalog = SemanticCatalog(
        catalog_id="cat",
        source_key="source",
        source_fingerprint="fingerprint",
        published_version=7,
    )

    items = store.search(catalog=catalog, query="service performance", top_k=2)

    assert [(item.entity_type, item.entity_id) for item in items] == [
        ("term", "term:service"),
        ("metric", "metric:service_index"),
    ]
    assert store._client.calls == 1
    assert filter_args["published_version"] == 7


def test_failed_embedding_keeps_previous_vector_index() -> None:
    deleted: list[str] = []
    store = SemanticVectorStore(
        url="http://qdrant:6333",
        collection="semantic_catalog_chunks",
        vector_enabled=True,
        embedding_provider="local",
    )
    store._ensure_collection = lambda: None  # type: ignore[method-assign]
    store.delete_catalog = lambda catalog: deleted.append(catalog.catalog_id)  # type: ignore[method-assign]
    store._embeddings_client = SimpleNamespace(
        embed_documents=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    catalog = SemanticCatalog(
        catalog_id="cat",
        source_key="source",
        source_fingerprint="fp",
        terms=[SemanticTerm(term_id="term:revenue", name="Revenue")],
    )

    with pytest.raises(RuntimeError, match="offline"):
        store.upsert_catalog(catalog)

    assert deleted == []


def test_existing_qdrant_collection_must_match_embedding_dimension() -> None:
    store = SemanticVectorStore(
        url="http://qdrant:6333",
        collection="semantic_catalog_chunks",
        vector_enabled=True,
        embedding_provider="local",
        embedding_dim=1024,
    )
    store._client = SimpleNamespace(
        collection_exists=lambda _name: True,
        get_collection=lambda _name: SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=1536)),
            )
        ),
    )
    store._qdrant = lambda: (None, SimpleNamespace())  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="expected 1024"):
        store._ensure_collection()


def test_vector_search_deduplicates_and_caps_global_results() -> None:
    general_hits = [
        SimpleNamespace(payload={"entity_type": "term", "entity_id": f"term:{index}"}, score=0.9)
        for index in range(2)
    ]
    metric_hits = [
        SimpleNamespace(
            payload={"entity_type": "metric", "entity_id": f"metric:index_{index}"},
            score=0.8,
        )
        for index in range(2)
    ]

    class FakeClient:
        def search(self, **kwargs):
            return [general_hits[0], metric_hits[1], metric_hits[0], metric_hits[1], general_hits[1]]

    store = SemanticVectorStore(
        url="http://qdrant:6333",
        collection="semantic_catalog_chunks",
        vector_enabled=True,
        embedding_provider="local",
    )
    store._client = FakeClient()
    store._embeddings_client = SimpleNamespace(embed_query=lambda _query: [1.0])
    store._filter = lambda **kwargs: kwargs.get("entity_type")  # type: ignore[method-assign]
    catalog = SemanticCatalog(catalog_id="cat", source_key="source", source_fingerprint="fp")

    items = store.search(catalog=catalog, query="service performance", top_k=3)

    assert len(items) == 3
    assert [item.entity_id for item in items] == ["term:0", "term:1", "metric:index_1"]
