from __future__ import annotations

from pathlib import Path

import pytest

from backend.data_access.data_catalog import (
    CatalogColumn,
    CatalogTable,
    DataCatalogSnapshot,
)
from backend.data_access.semantic_catalog_service import SemanticCatalogService
from backend.data_access.semantic_context import (
    SemanticContextBuilder,
    format_semantic_context_prompt,
)
from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticColumn,
    SemanticDimension,
    SemanticEntity,
    SemanticFact,
    SemanticMetric,
    SemanticMetricCreate,
    SemanticRelationship,
    SemanticRelationshipCreate,
    SemanticSearchResultItem,
    SemanticTable,
    SemanticTerm,
    SemanticTermCreate,
    SemanticValidationResult,
)
from backend.data_access.semantic_seed import load_semantic_seed_packs
from backend.data_access.semantic_validator import validate_semantic_catalog
from tests.in_memory_semantic_store import SemanticSessionStore as SessionStore


def _snapshot() -> DataCatalogSnapshot:
    return DataCatalogSnapshot(
        built_at="2026-07-02T00:00:00+00:00",
        source_fingerprint="fp-1",
        tables=[
            CatalogTable(
                qualified_name="sales",
                table_name="sales",
                source_kind="csv_session",
                columns=[
                    CatalogColumn(name="order_date", dtype="datetime"),
                    CatalogColumn(name="amount", dtype="numeric"),
                    CatalogColumn(name="region", dtype="string"),
                ],
            ),
        ],
    )


def _orders_snapshot() -> DataCatalogSnapshot:
    return DataCatalogSnapshot(
        built_at="2026-07-02T00:00:00+00:00",
        source_fingerprint="fp-orders",
        tables=[
            CatalogTable(
                qualified_name="orders",
                table_name="orders",
                source_kind="csv_session",
                columns=[
                    CatalogColumn(name="order_id", dtype="int"),
                    CatalogColumn(name="customer_id", dtype="int"),
                    CatalogColumn(name="order_date", dtype="datetime"),
                    CatalogColumn(name="amount", dtype="numeric"),
                    CatalogColumn(name="region", dtype="string"),
                ],
            ),
        ],
    )


def _orders_customers_snapshot() -> DataCatalogSnapshot:
    return DataCatalogSnapshot(
        built_at="2026-07-02T00:00:00+00:00",
        source_fingerprint="fp-orders-customers",
        tables=[
            CatalogTable(
                qualified_name="orders",
                table_name="orders",
                source_kind="csv_session",
                columns=[
                    CatalogColumn(name="order_id", dtype="int"),
                    CatalogColumn(name="customer_id", dtype="int"),
                    CatalogColumn(name="order_date", dtype="datetime"),
                    CatalogColumn(name="amount", dtype="numeric"),
                ],
            ),
            CatalogTable(
                qualified_name="customers",
                table_name="customers",
                source_kind="csv_session",
                columns=[
                    CatalogColumn(name="customer_id", dtype="int"),
                    CatalogColumn(name="region", dtype="string"),
                ],
            ),
        ],
    )


def _ecommerce_snapshot() -> DataCatalogSnapshot:
    return DataCatalogSnapshot(
        built_at="2026-07-02T00:00:00+00:00",
        source_fingerprint="fp-ecommerce",
        tables=[
            CatalogTable(
                qualified_name="orders",
                table_name="orders",
                source_kind="csv_session",
                columns=[
                    CatalogColumn(name="order_id", dtype="int"),
                    CatalogColumn(name="customer_id", dtype="int"),
                    CatalogColumn(name="order_date", dtype="datetime"),
                    CatalogColumn(name="gmv", dtype="numeric"),
                    CatalogColumn(name="refund_amount", dtype="numeric"),
                    CatalogColumn(name="discount_amount", dtype="numeric"),
                    CatalogColumn(name="quantity", dtype="numeric"),
                ],
            ),
        ],
    )


def test_semantic_seed_packs_are_loaded_from_json() -> None:
    load_semantic_seed_packs.cache_clear()
    packs = load_semantic_seed_packs()

    assert {pack.name for pack in packs} >= {"global_common", "ecommerce", "saas", "marketing", "finance"}
    by_name = {pack.name: pack for pack in packs}
    assert len(by_name["global_common"].terms) >= 10
    assert "Revenue" in {term.name for term in by_name["global_common"].terms}
    assert by_name["global_common"].domains == ["common"]
    assert by_name["ecommerce"].domains == ["ecommerce", "retail"]
    assert all(pack.enabled_by_default for pack in packs)


def test_seed_terms_do_not_contain_mojibake() -> None:
    load_semantic_seed_packs.cache_clear()
    packs = load_semantic_seed_packs()
    all_synonyms = [synonym for pack in packs for term in pack.terms for synonym in term.synonyms]

    assert not any("Р" in item or "С" in item for item in all_synonyms)


def test_semantic_catalog_v2_accepts_business_objects() -> None:
    catalog = SemanticCatalog(
        catalog_id="cat-1",
        user_id=0,
        session_id="",
        source_key="source:1",
        version="2.0",
        entities=[
            SemanticEntity(
                entity_id="entity:orders.order",
                name="order",
                table="orders",
                expr="order_id",
                type="primary",
            )
        ],
        dimensions=[
            SemanticDimension(
                dimension_id="dimension:orders.order_date",
                name="order_date",
                table="orders",
                expr="order_date",
                type="time",
                grains=["day", "month"],
            )
        ],
        facts=[
            SemanticFact(
                fact_id="fact:orders.amount",
                name="amount",
                table="orders",
                expr="amount",
                type="number",
            )
        ],
        metrics=[
            SemanticMetric(
                metric_id="metric:revenue",
                key="revenue",
                name="Revenue",
                type="simple",
                agg="sum",
                expr="amount",
                base_table="orders",
            )
        ],
        validation=SemanticValidationResult(errors=[], warnings=[], quality_score=1.0),
    )

    assert catalog.version == "2.0"
    assert "metrics_v2" not in catalog.model_dump()
    assert catalog.metrics[0].key == "revenue"
    assert catalog.validation.quality_score == 1.0


class _VectorStore:
    def __init__(self) -> None:
        self.indexed = []
        self.deleted = []
        self.results: list[SemanticSearchResultItem] = []

    @property
    def enabled(self) -> bool:
        return True

    def upsert_catalog(self, catalog) -> None:
        self.indexed.append(catalog)

    def delete_catalog(self, catalog, *, published_version=None) -> None:
        _ = published_version
        self.deleted.append(catalog)

    def search(self, *, catalog, query: str, top_k: int):
        return self.results[:top_k]


def test_refresh_does_not_publish_metrics_from_static_column_name_guesses(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    store.create_session()
    session_id = next(tmp_path.iterdir()).name
    store.save_data_catalog(session_id, _snapshot())
    vectors = _VectorStore()
    service = SemanticCatalogService(store=store, vector_store=vectors)

    catalog = service.refresh(session_id=session_id, user_id=7)

    assert catalog.status == "ready"
    assert catalog.source_fingerprint == "fp-1"
    assert [table.qualified_name for table in catalog.tables] == ["sales"]
    assert [column.name for column in catalog.columns] == ["order_date", "amount", "region"]
    assert catalog.metrics == []
    assert catalog.terms == []
    assert service._load_overlay(catalog.source_key).version == 0
    assert vectors.indexed == [catalog]
    assert catalog.source_key
    assert service.load_for_session(session_id=session_id, user_id=7).catalog_id == catalog.catalog_id


def test_clear_semantic_catalog_removes_source_documents_and_vector_index(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _snapshot())
    vectors = _VectorStore()
    service = SemanticCatalogService(store=store, vector_store=vectors)

    catalog = service.refresh(session_id=state.session_id, user_id=7)

    service.clear_for_session(session_id=state.session_id, user_id=7)

    assert vectors.deleted == [catalog]
    assert service.catalog_store.load_published(catalog.source_key) is None


def test_profile_error_is_visible_as_degraded_catalog(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.bind_csv_source(state.session_id, filename="sales.csv")
    store.save_data_catalog(
        state.session_id,
        DataCatalogSnapshot(
            source_fingerprint="csv:test",
            tables=[
                CatalogTable(
                    qualified_name="sales",
                    table_name="sales",
                    source_kind="csv_session",
                    columns=[CatalogColumn(name="amount", dtype="double")],
                )
            ],
            errors=["profiling timeout"],
        ),
    )

    catalog = SemanticCatalogService(store=store, vector_store=_VectorStore()).refresh(
        session_id=state.session_id,
        user_id=7,
    )

    assert catalog.status == "degraded"
    assert catalog.error == "profiling timeout"


def test_empty_session_returns_unbound_global_glossary(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())

    catalog = service.load_for_session(session_id=state.session_id, user_id=7)

    assert catalog is not None
    assert catalog.status == "unbound"
    assert catalog.error is None
    assert catalog.tables == []
    assert "Revenue" in {term.name for term in catalog.terms}


def test_custom_term_requires_a_bound_source(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    empty = store.create_session()
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())

    with pytest.raises(ValueError, match="Bind a data source"):
        service.create_term(
            session_id=empty.session_id,
            user_id=7,
            payload=SemanticTermCreate(
                name="GMV custom",
                description="Source glossary term.",
                synonyms=["gross merchandise value custom"],
            ),
        )


def test_refresh_builds_v2_entities_dimensions_and_facts_without_guessing_metrics(
    tmp_path: Path,
) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _orders_snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())

    catalog = service.refresh(session_id=state.session_id, user_id=7)

    assert catalog.version == "2.0"
    assert any(entity.name == "order" for entity in catalog.entities)
    assert any(dim.name == "order_date" and dim.type == "time" for dim in catalog.dimensions)
    assert any(fact.name == "amount" for fact in catalog.facts)
    assert catalog.metrics == []
    assert catalog.validation.errors == []


def test_validator_rejects_unknown_metric_references() -> None:
    catalog = SemanticCatalog(
        catalog_id="cat",
        user_id=0,
        session_id="",
        source_key="source",
        version="2.0",
        metrics=[
            SemanticMetric(
                metric_id="metric:aov",
                key="aov",
                name="AOV",
                type="ratio",
                base_table="orders",
                numerator="revenue",
                denominator="missing_orders",
            )
        ],
    )

    result = validate_semantic_catalog(catalog)

    assert any(issue.code == "unknown_metric_reference" for issue in result.errors)
    assert result.quality_score < 1.0


def test_validator_reports_incomplete_semantic_layer() -> None:
    catalog = SemanticCatalog(
        catalog_id="cat",
        source_key="source",
        tables=[
            SemanticTable(
                table_id="table:orders",
                qualified_name="orders",
                table_name="orders",
                source_kind="db",
                semantic_role="fact",
            ),
            SemanticTable(
                table_id="table:customers",
                qualified_name="customers",
                table_name="customers",
                source_kind="db",
                description="Customers directory.",
                semantic_role="dimension",
            ),
        ],
        columns=[
            SemanticColumn(
                column_id="column:orders.turnover",
                table="orders",
                name="turnover",
                dtype="numeric",
                semantic_role="dimension",
            )
        ],
    )

    result = validate_semantic_catalog(catalog)
    codes = {issue.code for issue in result.warnings}

    assert "missing_table_description" in codes
    assert "fact_table_without_metrics" in codes
    assert "numeric_measure_as_dimension" in codes
    assert "catalog_without_relationships" in codes
    assert result.quality_score < 1.0


def test_validator_trusts_explicit_metric_aggregation() -> None:
    catalog = SemanticCatalog(
        catalog_id="cat",
        source_key="source",
        tables=[
            SemanticTable(
                table_id="table:survey",
                qualified_name="survey",
                table_name="survey",
                source_kind="db",
                description="Customer satisfaction survey",
                semantic_role="fact",
            )
        ],
        columns=[
            SemanticColumn(
                column_id="column:survey.safety_score",
                table="survey",
                name="safety_score",
                dtype="numeric",
                semantic_role="metric_candidate",
                description="Safety rating from the survey",
            ),
            SemanticColumn(
                column_id="column:survey.response_count",
                table="survey",
                name="response_count",
                dtype="numeric",
                semantic_role="metric_candidate",
            ),
        ],
        metrics=[
            SemanticMetric(
                metric_id="metric:safety_score",
                key="safety_score",
                name="Safety score",
                type="simple",
                base_table="survey",
                expr="safety_score",
                agg="sum",
            ),
            SemanticMetric(
                metric_id="metric:responses",
                key="responses",
                name="Response count",
                type="simple",
                base_table="survey",
                expr="response_count",
                agg="sum",
            ),
        ],
    )

    result = validate_semantic_catalog(catalog)

    assert all(issue.code != "suspicious_metric_aggregation" for issue in result.warnings)
    assert [metric.agg for metric in catalog.metrics] == ["sum", "sum"]


def test_numeric_column_roles_distinguish_measures_from_low_cardinality_dimensions(
    tmp_path: Path,
) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(
        state.session_id,
        DataCatalogSnapshot(
            source_fingerprint="numeric-roles",
            tables=[
                CatalogTable(
                    qualified_name="stats",
                    table_name="stats",
                    source_kind="csv_session",
                    columns=[
                        CatalogColumn(name="turnover", dtype="numeric", distinct_count=100),
                        CatalogColumn(name="segment", dtype="integer", distinct_count=5),
                        CatalogColumn(name="measurement", dtype="float64"),
                    ],
                )
            ],
        ),
    )

    catalog = SemanticCatalogService(store=store, vector_store=_VectorStore()).refresh(
        session_id=state.session_id,
        user_id=7,
    )
    roles = {column.name: column.semantic_role for column in catalog.columns}

    assert roles == {
        "turnover": "metric_candidate",
        "segment": "dimension",
        "measurement": "metric_candidate",
    }


def test_validator_rejects_metric_cycles() -> None:
    catalog = SemanticCatalog(
        catalog_id="cat",
        user_id=0,
        session_id="",
        source_key="source",
        version="2.0",
        metrics=[
            SemanticMetric(
                metric_id="metric:a",
                key="a",
                name="A",
                type="derived",
                base_table="t",
                formula="b",
            ),
            SemanticMetric(
                metric_id="metric:b",
                key="b",
                name="B",
                type="derived",
                base_table="t",
                formula="a",
            ),
        ],
    )

    result = validate_semantic_catalog(catalog)

    assert any(issue.code == "metric_cycle" for issue in result.errors)


def test_validator_does_not_treat_qualified_columns_as_metric_refs() -> None:
    catalog = SemanticCatalog(
        catalog_id="cat",
        user_id=0,
        session_id="",
        source_key="source",
        version="2.0",
        metrics=[
            SemanticMetric(
                metric_id="metric:orders",
                key="orders",
                name="Orders",
                type="derived",
                base_table="orders",
                formula="COUNT(DISTINCT orders.id)",
            )
        ],
    )

    result = validate_semantic_catalog(catalog)

    assert not any(issue.code == "metric_cycle" for issue in result.errors)


def test_metric_validation_rejects_unknown_columns_and_unsafe_sql(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    service.refresh(session_id=state.session_id, user_id=7)

    with pytest.raises(ValueError, match="Unknown metric column"):
        service.create_metric(
            session_id=state.session_id,
            user_id=7,
            payload=SemanticMetricCreate(
                key="revenue",
                name="Выручка",
                type="simple",
                base_table="sales",
                expr="missing",
                agg="sum",
            ),
        )

    with pytest.raises(ValueError, match="Unknown metric reference"):
        service.create_metric(
            session_id=state.session_id,
            user_id=7,
            payload=SemanticMetricCreate(
                key="bad",
                name="Bad",
                type="ratio",
                base_table="sales",
                numerator="revenue",
                denominator="missing_orders",
            ),
        )


@pytest.mark.parametrize(
    "time_ref",
    ["order_date", "sales.order_date", "dimension:sales.order_date"],
)
def test_metric_default_time_dimension_accepts_compiler_references(
    tmp_path: Path,
    time_ref: str,
) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    service.refresh(session_id=state.session_id, user_id=7)

    metric = service.create_metric(
        session_id=state.session_id,
        user_id=7,
        payload=SemanticMetricCreate(
            key="time_aware_amount",
            name="Time-aware amount",
            type="simple",
            base_table="sales",
            expr="amount",
            agg="sum",
            default_time_dimension=time_ref,
        ),
    )

    assert metric.default_time_dimension == time_ref


@pytest.mark.parametrize(
    ("time_ref", "deactivate"),
    [("region", False), ("order_date", True)],
)
def test_metric_default_time_dimension_rejects_non_time_or_inactive_dimensions(
    tmp_path: Path,
    time_ref: str,
    deactivate: bool,
) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    catalog = service.refresh(session_id=state.session_id, user_id=7)
    if deactivate:
        catalog.dimensions = [
            dimension.model_copy(update={"is_active": False}) if dimension.name == time_ref else dimension
            for dimension in catalog.dimensions
        ]

    with pytest.raises(ValueError, match="time dimension"):
        service.validate_metric_candidate(
            catalog,
            SemanticMetric(
                metric_id="metric:time_aware_amount",
                key="time_aware_amount",
                name="Time-aware amount",
                type="simple",
                base_table="sales",
                expr="amount",
                agg="sum",
                default_time_dimension=time_ref,
            ),
        )


def test_derived_metric_may_reference_existing_metrics(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    service.refresh(session_id=state.session_id, user_id=7)

    for key, agg in (("amount_total", "sum"), ("amount_rows", "count")):
        service.create_metric(
            session_id=state.session_id,
            user_id=7,
            payload=SemanticMetricCreate(
                key=key,
                name=key,
                type="simple",
                base_table="sales",
                expr="amount",
                agg=agg,
            ),
        )

    metric = service.create_metric(
        session_id=state.session_id,
        user_id=7,
        payload=SemanticMetricCreate(
            key="scaled_average",
            name="Scaled average",
            type="derived",
            base_table="sales",
            formula="amount_total / NULLIF(amount_rows, 0) * 1000",
        ),
    )

    assert metric.formula == "amount_total / NULLIF(amount_rows, 0) * 1000"

    normalized = service.create_metric(
        session_id=state.session_id,
        user_id=7,
        payload=SemanticMetricCreate(
            key="normalized_total",
            name="Normalized total",
            type="derived",
            base_table="sales",
            formula="amount_total / 12",
        ),
    )

    assert normalized.formula == "amount_total / 12"

    catalog = service.load_for_session(session_id=state.session_id, user_id=7)
    assert catalog is not None
    amount_total = next(metric for metric in catalog.metrics if metric.key == "amount_total")
    with pytest.raises(
        ValueError,
        match=(
            r"Метрику amount_total нельзя удалить: от неё зависят активные метрики: "
            r"normalized_total, scaled_average\. Сначала измените или удалите зависимые метрики\."
        ),
    ):
        service.delete_metric(
            session_id=state.session_id,
            user_id=7,
            metric_id=amount_total.metric_id,
        )


def test_validator_rejects_transitive_cross_table_metric_dependencies() -> None:
    catalog = SemanticCatalog(
        catalog_id="cat",
        source_key="source",
        metrics=[
            SemanticMetric(
                metric_id="metric:orders_total",
                key="orders_total",
                name="Orders total",
                base_table="orders",
                expr="amount",
                agg="sum",
            ),
            SemanticMetric(
                metric_id="metric:customer_count",
                key="customer_count",
                name="Customer count",
                base_table="customers",
                expr="customer_id",
                agg="count",
            ),
            SemanticMetric(
                metric_id="metric:blended",
                key="blended",
                name="Blended",
                type="derived",
                base_table="orders",
                formula="orders_total / NULLIF(customer_count, 0)",
            ),
            SemanticMetric(
                metric_id="metric:normalized",
                key="normalized",
                name="Normalized",
                type="derived",
                base_table="orders",
                formula="blended / 12",
            ),
        ],
    )

    result = validate_semantic_catalog(catalog)

    assert {issue.object_id for issue in result.errors if issue.code == "cross_table_metric_dependency"} == {
        "metric:blended",
        "metric:normalized",
    }


def test_validator_ignores_inactive_cross_table_metric_dependencies() -> None:
    catalog = SemanticCatalog(
        catalog_id="cat",
        source_key="source",
        metrics=[
            SemanticMetric(
                metric_id="metric:orders_total",
                key="orders_total",
                name="Orders total",
                base_table="orders",
                expr="amount",
                agg="sum",
            ),
            SemanticMetric(
                metric_id="metric:customer_count",
                key="customer_count",
                name="Customer count",
                base_table="customers",
                expr="customer_id",
                agg="count",
            ),
            SemanticMetric(
                metric_id="metric:inactive_blend",
                key="inactive_blend",
                name="Inactive blend",
                type="derived",
                base_table="orders",
                formula="orders_total / NULLIF(customer_count, 0)",
                is_active=False,
            ),
        ],
    )

    result = validate_semantic_catalog(catalog)

    assert all(issue.code != "cross_table_metric_dependency" for issue in result.errors)


def test_validator_rejects_active_dependency_on_inactive_metric() -> None:
    catalog = SemanticCatalog(
        catalog_id="cat",
        source_key="source",
        metrics=[
            SemanticMetric(
                metric_id="metric:base",
                key="base",
                name="Base",
                base_table="orders",
                expr="amount",
                agg="sum",
                is_active=False,
            ),
            SemanticMetric(
                metric_id="metric:normalized",
                key="normalized",
                name="Normalized",
                type="derived",
                base_table="orders",
                formula="base / 12",
            ),
        ],
    )

    result = validate_semantic_catalog(catalog)

    assert any(
        issue.code == "inactive_metric_dependency" and issue.object_id == "metric:normalized"
        for issue in result.errors
    )


def test_metric_creation_rejects_cross_table_dependency(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _orders_customers_snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    service.refresh(session_id=state.session_id, user_id=7)

    for key, table, expr, agg in (
        ("orders_total", "orders", "amount", "sum"),
        ("customer_count", "customers", "customer_id", "count"),
    ):
        service.create_metric(
            session_id=state.session_id,
            user_id=7,
            payload=SemanticMetricCreate(
                key=key,
                name=key,
                base_table=table,
                expr=expr,
                agg=agg,
            ),
        )

    with pytest.raises(ValueError, match="other base tables"):
        service.create_metric(
            session_id=state.session_id,
            user_id=7,
            payload=SemanticMetricCreate(
                key="blended",
                name="Blended",
                type="derived",
                base_table="orders",
                formula="orders_total / NULLIF(customer_count, 0)",
            ),
        )


def test_created_simple_metric_is_published_as_canonical_metric(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    service.refresh(session_id=state.session_id, user_id=7)

    service.create_metric(
        session_id=state.session_id,
        user_id=7,
        payload=SemanticMetricCreate(
            key="avg_amount",
            name="Average amount",
            type="simple",
            base_table="sales",
            expr="amount",
            agg="avg",
            allowed_dimensions=["region"],
        ),
    )

    catalog = service.load_for_session(session_id=state.session_id, user_id=7)
    assert catalog is not None
    assert "metrics_v2" not in catalog.model_dump()
    metric = next(item for item in catalog.metrics if item.key == "avg_amount")
    assert metric.type == "simple"
    assert metric.expr == "amount"
    assert metric.agg == "avg"


def test_generated_overlay_replaces_metric_only_when_explicitly_requested(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    service.refresh(session_id=state.session_id, user_id=7)
    existing = service.create_metric(
        session_id=state.session_id,
        user_id=7,
        payload=SemanticMetricCreate(
            key="revenue",
            name="Verified revenue",
            base_table="sales",
            expr="amount",
            agg="sum",
        ),
    )
    replacement = SemanticMetric(
        **SemanticMetricCreate(
            key="revenue",
            name="Localized revenue",
            description="Confirmed scenario metadata.",
            base_table="sales",
            expr="amount",
            agg="sum",
        ).model_dump(),
        metric_id=existing.metric_id,
        created_at=existing.created_at,
    )

    unchanged, _ = service.apply_generated_overlay(
        session_id=state.session_id,
        user_id=7,
        metrics=[replacement],
    )
    assert next(item for item in unchanged.metrics if item.key == "revenue").name == "Verified revenue"

    updated, _ = service.apply_generated_overlay(
        session_id=state.session_id,
        user_id=7,
        metrics=[replacement],
        replace_metrics=True,
    )
    metrics = [item for item in updated.metrics if item.key == "revenue"]
    assert len(metrics) == 1
    assert metrics[0].name == "Localized revenue"


def test_generated_overlay_resolves_metric_dependencies_independent_of_input_order(
    tmp_path: Path,
) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    service.refresh(session_id=state.session_id, user_id=7)
    base = SemanticMetric(
        metric_id="metric:generated_total",
        key="generated_total",
        name="Generated total",
        type="simple",
        base_table="sales",
        expr="amount",
        agg="sum",
    )
    derived = SemanticMetric(
        metric_id="metric:generated_scaled",
        key="generated_scaled",
        name="Generated scaled",
        type="derived",
        base_table="sales",
        formula="generated_total / 12",
    )

    published, rejected = service.apply_generated_overlay(
        session_id=state.session_id,
        user_id=7,
        metrics=[derived, base],
    )

    assert rejected == []
    assert {"generated_total", "generated_scaled"} <= {metric.key for metric in published.metrics}


def test_context_builder_uses_qdrant_results_and_lexical_fallback(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _snapshot())
    vectors = _VectorStore()
    service = SemanticCatalogService(store=store, vector_store=vectors)
    service.refresh(session_id=state.session_id, user_id=7)
    metric = service.create_metric(
        session_id=state.session_id,
        user_id=7,
        payload=SemanticMetricCreate(
            key="revenue",
            name="Выручка",
            type="simple",
            base_table="sales",
            expr="amount",
            agg="sum",
            default_time_dimension="order_date",
            allowed_dimensions=["region"],
            synonyms=["оборот"],
        ),
    )
    vectors.results = [
        SemanticSearchResultItem(
            entity_type="metric",
            entity_id=metric.metric_id,
            score=0.93,
        ),
    ]
    builder = SemanticContextBuilder(store=store, vector_store=vectors, catalog_service=service)

    context = builder.build(session_id=state.session_id, user_id=7, query="оборот по регионам")

    assert "SEMANTIC DATA CONTEXT" in context.prompt
    assert "inspect catalog relationships" in context.prompt
    assert "a term is not a metric unless explicitly linked" in context.prompt
    assert (
        "metric_resolution: status=resolved; confirmed_metric_keys=revenue; "
        "calculation_action=execute_only_if_requested_grain_is_allowed"
    ) in context.prompt
    assert "allowed_dimensions=region" in context.prompt
    assert "revenue / Выручка" not in context.prompt
    assert "amount" not in context.prompt
    assert context.items[0].entity_type == "metric"

    vectors.search = lambda **_: (_ for _ in ()).throw(RuntimeError("qdrant down"))  # type: ignore[method-assign]
    fallback = builder.build(session_id=state.session_id, user_id=7, query="оборот")

    assert fallback.status == "degraded"
    assert "revenue / Выручка" not in fallback.prompt
    assert service.load_for_session(session_id=state.session_id, user_id=7).status == "ready"


def test_context_builder_keeps_exact_term_with_nonempty_vector_results(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _snapshot())
    vectors = _VectorStore()
    service = SemanticCatalogService(store=store, vector_store=vectors)
    service.refresh(session_id=state.session_id, user_id=7)
    term = service.create_term(
        session_id=state.session_id,
        user_id=7,
        payload=SemanticTermCreate(name="Gross margin", description="Revenue less direct costs"),
    )
    vectors.results = [SemanticSearchResultItem(entity_type="table", entity_id="table:sales", score=0.99)]

    context = SemanticContextBuilder(
        store=store,
        vector_store=vectors,
        catalog_service=service,
        top_k=2,
    ).build(session_id=state.session_id, user_id=7, query="Explain Gross margin")

    assert [(item.entity_type, item.entity_id) for item in context.items] == [
        ("term", term.term_id),
        ("table", "table:sales"),
    ]
    assert context.hints["term_resolution_status"] == "resolved"
    assert context.hints["metric_resolution_status"] == "not_found"
    assert "term_resolution: status=resolved" in context.prompt


def test_semantic_context_includes_only_table_descriptions() -> None:
    catalog = SemanticCatalog(
        catalog_id="catalog-1",
        status="ready",
        tables=[
            SemanticTable(
                table_id="table:monthly_plan",
                qualified_name="demo.monthly_plan",
                table_name="monthly_plan",
                source_kind="db",
                description="Monthly internal plan values.\nUse the named measure column.",
                semantic_role="fact",
                grain="month and measure",
                aliases=["internal plan"],
            )
        ],
        columns=[
            SemanticColumn(
                column_id="column:monthly_plan.measure",
                table="demo.monthly_plan",
                name="measure",
                semantic_role="dimension",
                description="Selects which planned metric the numeric value represents.",
                aliases=["metric name"],
            ),
            SemanticColumn(
                column_id="column:monthly_plan.value",
                table="demo.monthly_plan",
                name="value",
                semantic_role="metric_candidate",
                description="Numeric plan value.",
            ),
        ],
    )

    prompt = format_semantic_context_prompt(catalog)

    assert "description=Monthly internal plan values. Use the named measure column." in prompt
    assert "grain=month and measure" in prompt
    assert "aliases=internal plan" in prompt
    assert "measure: Selects which planned metric" not in prompt
    assert "aliases=metric name" not in prompt
    assert "value: Numeric plan value." not in prompt


def test_exact_term_adds_referenced_relationship_contract_to_context(tmp_path: Path) -> None:
    catalog = SemanticCatalog(
        catalog_id="catalog-1",
        status="ready",
        relationships=[
            SemanticRelationship(
                relationship_id="relationship:orders_customers",
                from_table="orders",
                from_column="customer_id",
                to_table="customers",
                to_column="customer_id",
                cardinality="many_to_one",
                description="Each order belongs to exactly one customer.",
            )
        ],
        terms=[
            SemanticTerm(
                term_id="term:customer_relationship",
                name="Customer relationship",
                entity_refs=["relationship:orders_customers"],
            )
        ],
    )
    builder = SemanticContextBuilder(store=SessionStore(str(tmp_path), ttl_days=1))

    context = builder.build_from_catalog(catalog=catalog, query="Explain customer relationship")

    assert [item.entity_type for item in context.items] == ["term", "relationship"]
    assert "contract=orders.customer_id -> customers.customer_id (many_to_one)" in context.prompt
    assert "Each order belongs to exactly one customer." in context.prompt


def test_semantic_context_lists_visible_tables_without_query_ranking() -> None:
    tables = [
        SemanticTable(
            table_id=f"table:{index}",
            qualified_name=f"demo.table_{index}",
            table_name=f"table_{index}",
            source_kind="db",
        )
        for index in range(7)
    ]
    tables.append(
        SemanticTable(
            table_id="table:hidden",
            qualified_name="demo.hidden",
            table_name="hidden",
            source_kind="db",
            is_hidden=True,
        )
    )
    catalog = SemanticCatalog(catalog_id="catalog-1", status="ready", tables=tables)

    prompt = format_semantic_context_prompt(catalog)

    assert all(f"demo.table_{index}" in prompt for index in range(7))
    assert "demo.hidden" not in prompt


def test_source_terms_are_reused_only_by_matching_source(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    first = store.create_session()
    second = store.create_session()
    store.bind_csv_source(first.session_id, filename="sales.csv")
    store.bind_csv_source(second.session_id, filename="sales.csv")
    store.save_data_catalog(first.session_id, _snapshot())
    store.save_data_catalog(second.session_id, _snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    service.refresh(session_id=first.session_id, user_id=7)

    term = service.create_term(
        session_id=first.session_id,
        user_id=7,
        payload=SemanticTermCreate(
            name="Net sales",
            description="Revenue after returns and discounts.",
            synonyms=["net revenue"],
            entity_refs=["metric:revenue"],
        ),
    )

    reused = service.load_for_session(session_id=second.session_id, user_id=7)
    assert reused is not None
    assert term.term_id in {item.term_id for item in reused.terms}

    context = SemanticContextBuilder(
        store=store,
        vector_store=None,
        catalog_service=service,
    ).build(session_id=second.session_id, user_id=7, query="net revenue")

    assert context.status == "ready"
    assert "Net sales" in context.prompt
    assert any(item["term_id"] == term.term_id for item in context.hints["terms"])


def test_csv_overlay_reuse_is_scoped_to_the_same_user(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    first = store.create_session()
    second = store.create_session()
    source_hash = "sha256:" + ("a" * 64)
    store.bind_csv_source(first.session_id, filename="orders.csv", source_ref_id=source_hash)
    store.bind_csv_source(second.session_id, filename="renamed.csv", source_ref_id=source_hash)
    snapshot = _orders_customers_snapshot()
    snapshot.source_fingerprint = f"csv:{source_hash}"
    store.save_data_catalog(first.session_id, snapshot)
    store.save_data_catalog(second.session_id, snapshot)
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    first_catalog = service.refresh(session_id=first.session_id, user_id=7)

    rel = service.create_relationship(
        session_id=first.session_id,
        user_id=7,
        payload=SemanticRelationshipCreate(
            from_table="orders",
            from_column="customer_id",
            to_table="customers",
            to_column="customer_id",
            cardinality="many_to_one",
        ),
    )

    reused = service.load_for_session(session_id=second.session_id, user_id=7)

    assert reused is not None
    assert reused.source_key == first_catalog.source_key
    assert rel.relationship_id in {item.relationship_id for item in reused.relationships}


def test_ready_csv_catalog_is_reused_before_duplicate_session_profile(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    first = store.create_session()
    second = store.create_session()
    source_hash = "sha256:" + ("d" * 64)
    store.bind_csv_source(first.session_id, filename="orders.csv", source_ref_id=source_hash)
    store.bind_csv_source(second.session_id, filename="orders-copy.csv", source_ref_id=source_hash)
    snapshot = _orders_customers_snapshot()
    snapshot.source_fingerprint = f"csv:{source_hash}"
    store.save_data_catalog(first.session_id, snapshot)
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    first_catalog = service.refresh(session_id=first.session_id, user_id=7)

    reused = service.load_for_session(session_id=second.session_id, user_id=7)

    assert reused is not None
    assert reused.status == "ready"
    assert reused.source_key == first_catalog.source_key


def test_csv_overlay_isolated_between_users(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    first = store.create_session()
    second = store.create_session()
    source_hash = "sha256:" + ("b" * 64)
    store.bind_csv_source(first.session_id, filename="orders.csv", source_ref_id=source_hash)
    store.bind_csv_source(second.session_id, filename="orders-copy.csv", source_ref_id=source_hash)
    snapshot = _orders_customers_snapshot()
    snapshot.source_fingerprint = f"csv:{source_hash}"
    store.save_data_catalog(first.session_id, snapshot)
    store.save_data_catalog(second.session_id, snapshot)
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())

    first_catalog = service.refresh(session_id=first.session_id, user_id=7)
    service.create_relationship(
        session_id=first.session_id,
        user_id=7,
        payload=SemanticRelationshipCreate(
            from_table="orders",
            from_column="customer_id",
            to_table="customers",
            to_column="customer_id",
            cardinality="many_to_one",
        ),
    )
    term = service.create_term(
        session_id=first.session_id,
        user_id=7,
        payload=SemanticTermCreate(
            name="Private fulfillment score",
            description="A source-owned glossary definition.",
        ),
    )
    second_catalog = service.refresh(session_id=second.session_id, user_id=42)

    assert second_catalog.source_key != first_catalog.source_key
    assert second_catalog.relationships == []
    assert term.term_id not in {item.term_id for item in second_catalog.terms}


def test_csv_async_claim_and_profile_refresh_use_the_same_source_key(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    source_hash = "sha256:" + ("c" * 64)
    store.bind_csv_source(state.session_id, filename="orders.csv", source_ref_id=source_hash)
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())

    pending, operation = service.claim_session_build(session_id=state.session_id, user_id=7)
    assert operation is not None
    snapshot = _orders_customers_snapshot()
    snapshot.source_fingerprint = f"csv:{source_hash}"
    store.save_data_catalog(state.session_id, snapshot)
    ready = service.refresh(
        session_id=state.session_id,
        user_id=7,
        operation_id=operation.operation_id,
    )

    assert pending.source_key == ready.source_key
    assert ready.status == "ready"

    refreshed, refresh_operation = service.claim_session_build(
        session_id=state.session_id,
        user_id=7,
        force=True,
    )
    assert refresh_operation is not None
    assert refreshed.status == "ready"
    assert refreshed.source_key == ready.source_key


def test_planfact_async_claim_and_profile_refresh_use_the_same_source_key(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.set_source(
        state.session_id,
        source_type="planfact",
        source_ref_id="planfact",
        source_label="План-факт",
        source_mode="duckdb",
    )
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())

    pending, operation = service.claim_session_build(session_id=state.session_id, user_id=7)
    assert operation is not None
    snapshot = _orders_customers_snapshot()
    snapshot.source_fingerprint = "csv-session:planfact"
    store.save_data_catalog(state.session_id, snapshot)
    ready = service.refresh(
        session_id=state.session_id,
        user_id=7,
        operation_id=operation.operation_id,
    )

    assert pending.source_key == ready.source_key
    assert ready.status == "ready"


def test_clear_cancels_late_ai_generation_publish(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _orders_snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    ready = service.refresh(session_id=state.session_id, user_id=7)
    _catalog, operation = service.claim_session_build(
        session_id=state.session_id,
        user_id=7,
        operation_type="generate",
    )
    assert operation is not None

    service.clear_for_session(session_id=state.session_id, user_id=7)

    with pytest.raises(RuntimeError, match="cancelled"):
        service.apply_generated_overlay(
            session_id=state.session_id,
            user_id=7,
            terms=[SemanticTerm(term_id="term:late", name="Late")],
            operation_id=operation.operation_id,
        )
    assert service._load_published(ready.source_key) is None


def test_ai_generation_publishes_one_version_and_completes_operation(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _orders_snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    service.refresh(session_id=state.session_id, user_id=7)
    _catalog, operation = service.claim_session_build(
        session_id=state.session_id,
        user_id=7,
        operation_type="generate",
    )
    assert operation is not None

    published, rejected = service.apply_generated_overlay(
        session_id=state.session_id,
        user_id=7,
        terms=[SemanticTerm(term_id="term:margin", name="Margin")],
        operation_id=operation.operation_id,
    )

    assert rejected == []
    assert published.published_version == 1
    assert service._load_overlay(published.source_key).version == 1
    assert service.latest_operation(source_key=published.source_key).status == "completed"


def test_loaded_catalog_preserves_published_version_when_overlay_advances(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _orders_snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    published = service.refresh(session_id=state.session_id, user_id=7)
    overlay = service._load_overlay(published.source_key)
    service._save_overlay(overlay)

    loaded = service._load_published(published.source_key)

    assert loaded is not None
    assert loaded.overlay_version == overlay.version == 1
    assert loaded.published_version == published.published_version == 0


def test_relationship_crud_publishes_overlay(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    snapshot = DataCatalogSnapshot(
        built_at="2026-07-02T00:00:00+00:00",
        source_fingerprint="fp-rel",
        tables=[
            CatalogTable(
                qualified_name="orders",
                table_name="orders",
                source_kind="csv_session",
                columns=[
                    CatalogColumn(name="customer_id", dtype="int"),
                    CatalogColumn(name="amount", dtype="numeric"),
                ],
            ),
            CatalogTable(
                qualified_name="customers",
                table_name="customers",
                source_kind="csv_session",
                columns=[
                    CatalogColumn(name="customer_id", dtype="int"),
                    CatalogColumn(name="region", dtype="string"),
                ],
            ),
        ],
    )
    store.save_data_catalog(state.session_id, snapshot)
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    service.refresh(session_id=state.session_id, user_id=7)

    rel = service.create_relationship(
        session_id=state.session_id,
        user_id=7,
        payload=SemanticRelationshipCreate(
            from_table="orders",
            from_column="customer_id",
            to_table="customers",
            to_column="customer_id",
            cardinality="many_to_one",
        ),
    )

    catalog = service.load_for_session(session_id=state.session_id, user_id=7)
    assert catalog is not None
    assert rel.relationship_id in {item.relationship_id for item in catalog.relationships}


def test_relationship_validation_rejects_unsafe_join_and_keeps_degraded_status(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _orders_customers_snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    catalog = service.refresh(session_id=state.session_id, user_id=7)

    with pytest.raises(ValueError, match="Unsafe relationship"):
        service.create_relationship(
            session_id=state.session_id,
            user_id=7,
            payload=SemanticRelationshipCreate(
                from_table="orders",
                from_column="amount",
                to_table="customers",
                to_column="customer_id",
                cardinality="many_to_one",
            ),
        )

    overlay = service._load_overlay(catalog.source_key)
    overlay.relationships.append(
        SemanticRelationship(
            relationship_id="relationship:unsafe",
            from_table="orders",
            from_column="amount",
            to_table="customers",
            to_column="customer_id",
            cardinality="many_to_one",
        )
    )
    service._save_overlay(overlay)

    republished = service._republish_from_overlay(catalog, overlay)

    assert republished.status == "degraded"
    assert any(issue.code == "unsafe_relationship" for issue in republished.validation.errors)


def test_domain_seed_packs_do_not_automatically_publish_metrics(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _ecommerce_snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())

    catalog = service.refresh(session_id=state.session_id, user_id=7)

    assert catalog.metrics == []


def test_user_created_standard_revenue_metric_survives_reloads(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _orders_snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    service.refresh(session_id=state.session_id, user_id=7)
    metric = service.create_metric(
        session_id=state.session_id,
        user_id=7,
        payload=SemanticMetricCreate(
            key="revenue",
            name="Revenue",
            base_table="orders",
            expr="amount",
            agg="sum",
            default_time_dimension="order_date",
            allowed_dimensions=["region"],
        ),
    )

    for _ in range(2):
        loaded = service.load_for_session(session_id=state.session_id, user_id=7)
        assert loaded is not None
        assert metric.metric_id in {item.metric_id for item in loaded.metrics}


def test_user_term_description_survives_starter_merge_and_refresh(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    service.refresh(session_id=state.session_id, user_id=7)
    metric = service.create_metric(
        session_id=state.session_id,
        user_id=7,
        payload=SemanticMetricCreate(
            key="revenue",
            name="Revenue",
            base_table="sales",
            expr="amount",
            agg="sum",
        ),
    )
    user_description = "Net billed revenue."
    user_term = SemanticTerm(
        term_id="term:user-revenue",
        name="Revenue",
        description=user_description,
        entity_refs=[metric.metric_id],
    )

    published, rejected = service.apply_generated_overlay(
        session_id=state.session_id,
        user_id=7,
        terms=[user_term],
    )
    assert rejected == []
    assert next(term for term in published.terms if term.name == "Revenue").description == user_description

    refreshed = service.refresh(session_id=state.session_id, user_id=7)
    assert next(term for term in refreshed.terms if term.name == "Revenue").description == user_description

    context = service.search(session_id=state.session_id, user_id=7, query="Revenue")
    assert user_description in context.prompt
    assert any(item["description"] == user_description for item in context.hints["terms"])


def test_semantic_context_includes_metric_and_catalog_hints(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _orders_snapshot())
    vectors = _VectorStore()
    service = SemanticCatalogService(store=store, vector_store=vectors)
    service.refresh(session_id=state.session_id, user_id=7)
    metric = service.create_metric(
        session_id=state.session_id,
        user_id=7,
        payload=SemanticMetricCreate(
            key="revenue",
            name="Revenue",
            type="simple",
            base_table="orders",
            expr="amount",
            agg="sum",
            default_time_dimension="order_date",
            allowed_dimensions=["region"],
        ),
    )
    vectors.results = [
        SemanticSearchResultItem(
            entity_type="metric",
            entity_id=metric.metric_id,
            score=0.9,
        ),
    ]

    context = SemanticContextBuilder(store=store, vector_store=vectors, catalog_service=service).build(
        session_id=state.session_id,
        user_id=7,
        query="revenue by region",
    )

    assert "Metric:" not in context.prompt
    assert context.hints["metrics"][0]["key"] == metric.key
    assert "catalog" in context.hints


def test_cyrillic_semantic_names_survive_persistence_round_trip() -> None:
    dimension = SemanticDimension(
        dimension_id="dimension:receipts.Группа товара",
        name="Группа товара",
        table="receipts",
        expr="Группа товара",
        type="categorical",
    )

    restored = SemanticDimension.model_validate(dimension.model_dump())

    assert restored.name == "группа_товара"
