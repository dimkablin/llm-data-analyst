from __future__ import annotations

from backend.data_access.semantic_context import build_semantic_hints
from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticDimension,
    SemanticEntity,
    SemanticFact,
    SemanticMetric,
    SemanticRelationship,
    SemanticSearchResultItem,
    SemanticTable,
)
from backend.data_access.semantic_query import (
    SemanticQuery,
    SemanticQueryCompiler,
    semantic_query_from_hints,
)


def _catalog() -> SemanticCatalog:
    return SemanticCatalog(
        catalog_id="cat",
        user_id=0,
        session_id="",
        source_key="source",
        version="2.0",
        tables=[
            SemanticTable(
                table_id="table:orders",
                qualified_name="orders",
                table_name="orders",
                source_kind="csv",
            )
        ],
        dimensions=[
            SemanticDimension(
                dimension_id="dimension:orders.region",
                name="region",
                table="orders",
                expr="region",
                type="categorical",
            ),
            SemanticDimension(
                dimension_id="dimension:orders.order_date",
                name="order_date",
                table="orders",
                expr="order_date",
                type="time",
                grains=["month"],
            ),
        ],
        facts=[
            SemanticFact(
                fact_id="fact:orders.amount",
                name="amount",
                table="orders",
                expr="amount",
            )
        ],
        metrics=[
            SemanticMetric(
                metric_id="metric:revenue",
                key="revenue",
                name="Revenue",
                type="simple",
                base_table="orders",
                expr="amount",
                agg="sum",
                default_time_dimension="order_date",
                allowed_dimensions=["region", "order_date"],
            ),
            SemanticMetric(
                metric_id="metric:orders",
                key="orders",
                name="Orders",
                type="simple",
                base_table="orders",
                expr="order_id",
                agg="count_distinct",
                allowed_dimensions=["region", "order_date"],
            ),
            SemanticMetric(
                metric_id="metric:aov",
                key="aov",
                name="AOV",
                type="ratio",
                base_table="orders",
                numerator="revenue",
                denominator="orders",
                allowed_dimensions=["region", "order_date"],
            ),
        ],
    )


def test_compiler_builds_single_table_metric_by_dimension() -> None:
    query = SemanticQuery(metrics=["revenue"], dimensions=["region"], limit=100)

    sql = SemanticQueryCompiler(_catalog(), dialect="duckdb").compile(query)

    assert sql == (
        'SELECT "region" AS "region", SUM("amount") AS "revenue" '
        'FROM "orders" GROUP BY "region" ORDER BY "revenue" DESC LIMIT 100'
    )


def test_compiler_builds_ratio_metric() -> None:
    query = SemanticQuery(metrics=["aov"], dimensions=["region"])

    sql = SemanticQueryCompiler(_catalog(), dialect="duckdb").compile(query)

    assert 'SUM("amount") / NULLIF(COUNT(DISTINCT "order_id"), 0) AS "aov"' in sql
    assert 'GROUP BY "region"' in sql


def test_context_hints_can_build_semantic_query_for_metric() -> None:
    catalog = _catalog()
    hints = build_semantic_hints(
        catalog,
        [SemanticSearchResultItem(entity_type="metric", entity_id="metric:revenue", score=0.9)],
    )

    query = semantic_query_from_hints(hints, question="revenue by region", catalog=catalog)

    assert query is not None
    assert query.metrics == ["revenue"]
    assert query.dimensions == ["region"]


def test_context_hints_keep_multiple_matched_metrics() -> None:
    catalog = _catalog()
    hints = build_semantic_hints(
        catalog,
        [
            SemanticSearchResultItem(entity_type="metric", entity_id="metric:revenue", score=0.9),
            SemanticSearchResultItem(entity_type="metric", entity_id="metric:orders", score=0.8),
        ],
    )

    query = semantic_query_from_hints(hints, question="revenue and orders by region", catalog=catalog)

    assert query is not None
    assert query.metrics == ["revenue", "orders"]
    sql = SemanticQueryCompiler(catalog, dialect="duckdb").compile(query)
    assert 'SUM("amount") AS "revenue"' in sql
    assert 'COUNT(DISTINCT "order_id") AS "orders"' in sql


def test_compiler_qualifies_filters_through_semantic_dimensions() -> None:
    query = SemanticQuery(
        metrics=["revenue"],
        dimensions=["region"],
        filters=[
            {"field": "region", "op": "=", "value": "EMEA"},
            {"field": "amount", "op": ">", "value": 100},
        ],
    )

    sql = SemanticQueryCompiler(_catalog(), dialect="duckdb").compile(query)

    assert 'WHERE "region" = ' in sql
    assert '"amount" > 100' in sql


def test_compiler_rejects_unknown_filter_field() -> None:
    query = SemanticQuery(metrics=["revenue"], filters=[{"field": "missing", "op": "=", "value": "x"}])

    try:
        SemanticQueryCompiler(_catalog(), dialect="duckdb").compile(query)
    except ValueError as exc:
        assert "Unknown semantic filter field" in str(exc)
    else:
        raise AssertionError("unknown semantic filter field was accepted")


def test_compiler_uses_many_to_one_relationship_for_dimension() -> None:
    catalog = _catalog()
    catalog.tables.append(
        SemanticTable(
            table_id="table:customers",
            qualified_name="customers",
            table_name="customers",
            source_kind="csv",
        )
    )
    catalog.dimensions.append(
        SemanticDimension(
            dimension_id="dimension:customers.customer_region",
            name="customer_region",
            table="customers",
            expr="region",
            type="categorical",
        )
    )
    catalog.relationships.append(
        SemanticRelationship(
            relationship_id="relationship:orders_customers",
            from_table="orders",
            from_column="customer_id",
            to_table="customers",
            to_column="customer_id",
            cardinality="many_to_one",
        )
    )
    catalog.entities.extend(
        [
            SemanticEntity(
                entity_id="entity:orders.customer",
                name="customer",
                table="orders",
                expr="customer_id",
                type="foreign",
            ),
            SemanticEntity(
                entity_id="entity:customers.customer",
                name="customer",
                table="customers",
                expr="customer_id",
                type="primary",
            ),
        ]
    )
    query = SemanticQuery(
        metrics=["revenue"],
        dimensions=["customer_region"],
        filters=[{"field": "customer_region", "op": "=", "value": "EMEA"}],
    )

    sql = SemanticQueryCompiler(catalog, dialect="duckdb").compile(query)

    assert 'FROM "orders" AS t0 LEFT JOIN "customers" AS t1 ON t0."customer_id" = t1."customer_id"' in sql
    assert 't1."region" AS "customer_region"' in sql
    assert 'SUM(t0."amount") AS "revenue"' in sql
    assert 'WHERE t1."region" = ' in sql
