from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.data_access.data_catalog import (
    CatalogColumn,
    CatalogTable,
    DataCatalogSnapshot,
)
from backend.data_access.semantic_catalog_service import SemanticCatalogService
from backend.data_access.semantic_context import SemanticContextBuilder
from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticDimension,
    SemanticEntity,
    SemanticFact,
    SemanticMetric,
    SemanticMetricCreate,
    SemanticRelationship,
    SemanticRelationshipCreate,
    SemanticSearchResultItem,
    SemanticTermCreate,
    SemanticValidationResult,
)
from backend.data_access.semantic_seed import load_semantic_seed, load_semantic_seed_packs
from backend.data_access.semantic_validator import validate_semantic_catalog
from backend.sessions.session_store import SessionStore


SEED_PATH = Path(__file__).resolve().parents[1] / "backend" / "data_access" / "semantic_seed.global.json"


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


def test_semantic_seed_file_contains_starter_terms() -> None:
    raw_seed = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    load_semantic_seed.cache_clear()
    seed = load_semantic_seed(SEED_PATH)

    assert len(seed.terms) == 10
    assert len(seed.metric_templates) == 10
    assert "Revenue" in {item.name for item in seed.terms}
    assert "revenue" in seed.metric_columns
    assert "metric_templates" in raw_seed


def test_semantic_seed_packs_are_loaded_from_json() -> None:
    load_semantic_seed_packs.cache_clear()
    packs = load_semantic_seed_packs()

    assert {pack.name for pack in packs} >= {"global_common", "ecommerce", "saas", "marketing", "finance"}
    by_name = {pack.name: pack for pack in packs}
    assert len(by_name["global_common"].terms) >= 10
    assert len(by_name["ecommerce"].metric_templates) >= 7
    assert len(by_name["saas"].metric_templates) >= 6
    assert len(by_name["marketing"].metric_templates) >= 10
    assert len(by_name["finance"].metric_templates) >= 7


def test_seed_terms_do_not_contain_mojibake() -> None:
    load_semantic_seed_packs.cache_clear()
    packs = load_semantic_seed_packs()
    all_synonyms = [
        synonym
        for pack in packs
        for term in pack.terms
        for synonym in term.synonyms
    ]

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
        self.results: list[SemanticSearchResultItem] = []

    @property
    def enabled(self) -> bool:
        return True

    def upsert_catalog(self, catalog) -> None:
        self.indexed.append(catalog)

    def search(self, *, catalog, query: str, top_k: int):
        return self.results[:top_k]


def test_refresh_builds_semantic_catalog_and_indexes_qdrant(tmp_path: Path) -> None:
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
    assert len(catalog.terms) == 10
    assert "Revenue" in {term.name for term in catalog.terms}
    assert any(metric.key == "revenue" and metric.formula == "SUM(sales.amount)" for metric in catalog.metrics)
    assert vectors.indexed == [catalog]
    assert catalog.source_key
    assert service.load_for_session(session_id=session_id, user_id=7).catalog_id == catalog.catalog_id


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


def test_global_term_created_without_source_is_available_after_binding(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    empty = store.create_session()
    bound = store.create_session()
    store.save_data_catalog(bound.session_id, _snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())

    term = service.create_term(
        session_id=empty.session_id,
        user_id=7,
        payload=SemanticTermCreate(
            name="GMV custom",
            description="User glossary term without a bound source.",
            synonyms=["gross merchandise value custom"],
        ),
    )

    catalog = service.refresh(session_id=bound.session_id, user_id=7)

    assert catalog.status == "ready"
    assert term.term_id in {item.term_id for item in catalog.terms}


def test_refresh_builds_v2_entities_dimensions_facts_and_metrics(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _orders_snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())

    catalog = service.refresh(session_id=state.session_id, user_id=7)

    assert catalog.version == "2.0"
    assert any(entity.name == "order" for entity in catalog.entities)
    assert any(dim.name == "order_date" and dim.type == "time" for dim in catalog.dimensions)
    assert any(fact.name == "amount" for fact in catalog.facts)
    assert any(metric.key == "revenue" for metric in catalog.metrics)
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


def test_context_builder_uses_qdrant_results_and_lexical_fallback(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _snapshot())
    vectors = _VectorStore()
    service = SemanticCatalogService(store=store, vector_store=vectors)
    catalog = service.refresh(session_id=state.session_id, user_id=7)
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
    assert "revenue / Выручка" in context.prompt
    assert "amount" in context.prompt
    assert context.items[0].entity_type == "metric"

    vectors.search = lambda **_: (_ for _ in ()).throw(RuntimeError("qdrant down"))  # type: ignore[method-assign]
    fallback = builder.build(session_id=state.session_id, user_id=7, query="оборот")

    assert fallback.status == "degraded"
    assert "revenue / Выручка" in fallback.prompt
    assert service.load_for_session(session_id=state.session_id, user_id=7).status == "degraded"


def test_global_overlay_terms_are_reused_by_matching_source(tmp_path: Path) -> None:
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

    reused = service.load_for_session(session_id=second.session_id, user_id=42)
    assert reused is not None
    assert term.term_id in {item.term_id for item in reused.terms}

    context = SemanticContextBuilder(
        store=store,
        vector_store=None,
        catalog_service=service,
    ).build(session_id=second.session_id, user_id=42, query="net revenue")

    assert context.status == "ready"
    assert "Net sales" in context.prompt
    assert any(item["term_id"] == term.term_id for item in context.hints["terms"])


def test_csv_overlay_reuse_uses_content_hash_not_file_name(tmp_path: Path) -> None:
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

    reused = service.load_for_session(session_id=second.session_id, user_id=42)

    assert reused is not None
    assert reused.source_key == first_catalog.source_key
    assert rel.relationship_id in {item.relationship_id for item in reused.relationships}


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


def test_domain_seed_templates_are_ranked_by_matching_columns(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _ecommerce_snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())

    catalog = service.refresh(session_id=state.session_id, user_id=7)

    keys = {metric.key for metric in catalog.metrics}
    assert {"gmv", "refunds", "discounts"} <= keys
    assert "repeat_purchase_rate" not in keys


def test_semantic_context_includes_metric_and_catalog_hints(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.save_data_catalog(state.session_id, _orders_snapshot())
    vectors = _VectorStore()
    service = SemanticCatalogService(store=store, vector_store=vectors)
    catalog = service.refresh(session_id=state.session_id, user_id=7)
    vectors.results = [
        SemanticSearchResultItem(
            entity_type="metric",
            entity_id=catalog.metrics[0].metric_id,
            score=0.9,
        ),
    ]

    context = SemanticContextBuilder(store=store, vector_store=vectors, catalog_service=service).build(
        session_id=state.session_id,
        user_id=7,
        query="revenue by region",
    )

    assert "Metric:" in context.prompt
    assert context.hints["metrics"][0]["key"] == catalog.metrics[0].key
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
