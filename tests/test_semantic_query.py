from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import backend.tools.impl  # noqa: F401 - side-effect import avoids sql_table_service circular import
from backend.agent.graph.nodes.agent import _with_semantic_metric_footer
from backend.data_access.semantic_context import SemanticContextBuilder
from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticColumn,
    SemanticDimension,
    SemanticEntity,
    SemanticFact,
    SemanticMetric,
    SemanticRelationship,
    SemanticSearchResultItem,
    SemanticTable,
    SemanticTerm,
)
from backend.data_access.semantic_query import (
    SemanticQuery,
    SemanticQueryCompiler,
    SemanticQueryFilter,
)
from backend.data_access.semantic_vector_store import catalog_chunks
from backend.data_access.sql_table_service import SQLTableService
from backend.tools.artifact_references import attach_query_metadata
from backend.tools.impl.sql_tool import SQLTool
from backend.tools.sandbox import SessionSandbox


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
                grains=["month", "year"],
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


def _catalog_with_common_metric_filter() -> SemanticCatalog:
    catalog = _catalog()
    catalog.dimensions.append(
        SemanticDimension(
            dimension_id="dimension:orders.scope",
            name="scope",
            table="orders",
            expr="scope",
            type="categorical",
        )
    )
    catalog.columns.extend(
        [
            SemanticColumn(
                column_id="column:orders.scope",
                table="orders",
                name="scope",
                dtype="text",
            ),
            SemanticColumn(
                column_id="column:orders.amount",
                table="orders",
                name="amount",
                dtype="double",
            ),
        ]
    )
    fixed_filter = [{"field": "scope", "op": "=", "value": "company"}]
    catalog.metrics.extend(
        [
            SemanticMetric(
                metric_id="metric:actual_value",
                key="actual_value",
                name="Actual value",
                type="simple",
                base_table="orders",
                expr="amount",
                agg="sum",
                filters=fixed_filter,
                allowed_dimensions=["order_date", "scope"],
            ),
            SemanticMetric(
                metric_id="metric:target_value",
                key="target_value",
                name="Target value",
                type="simple",
                base_table="orders",
                expr="amount",
                agg="sum",
                filters=fixed_filter,
                allowed_dimensions=["order_date", "scope"],
            ),
            SemanticMetric(
                metric_id="metric:completion_index",
                key="completion_index",
                name="Completion index",
                type="derived",
                base_table="orders",
                formula="actual_value / NULLIF(target_value, 0)",
                allowed_dimensions=["order_date", "scope"],
            ),
        ]
    )
    return catalog


@pytest.mark.parametrize("op", ["in", "not_in"])
@pytest.mark.parametrize("value", ["north", []])
def test_set_filters_require_nonempty_collections(op: str, value: object) -> None:
    with pytest.raises(ValueError, match="requires a non-empty list or tuple"):
        SemanticQueryFilter(field="region", op=op, value=value)


@pytest.mark.parametrize("op", ["=", "starts_with"])
def test_scalar_filters_reject_collections(op: str) -> None:
    with pytest.raises(ValueError, match="requires a scalar value"):
        SemanticQueryFilter(field="region", op=op, value=["north"])


def test_set_filters_accept_tuples() -> None:
    item = SemanticQueryFilter(field="region", op="in", value=("north", "south"))

    assert item.value == ["north", "south"]


def test_compiler_builds_single_table_metric_by_dimension() -> None:
    query = SemanticQuery(metrics=["revenue"], dimensions=["region"], limit=100)

    sql = SemanticQueryCompiler(_catalog(), dialect="duckdb").compile(query)

    assert sql == (
        'SELECT "region" AS "region", SUM("amount") AS "revenue" '
        'FROM "orders" GROUP BY "region" ORDER BY "revenue" DESC LIMIT 100'
    )


def test_compiler_uses_shared_default_time_dimension_for_grain() -> None:
    query = SemanticQuery(
        metrics=["revenue"],
        time_grain="month",
        order_by=[{"field": "orders.order_date", "direction": "asc"}],
    )

    sql = SemanticQueryCompiler(_catalog(), dialect="duckdb").compile(query)

    assert "DATE_TRUNC('month'," in sql
    assert 'AS "order_date"' in sql
    assert 'ORDER BY "order_date" ASC' in sql


def test_compiler_treats_equivalent_default_time_refs_as_shared() -> None:
    catalog = _catalog()
    catalog.metrics[1] = catalog.metrics[1].model_copy(update={"default_time_dimension": "orders.order_date"})

    sql = SemanticQueryCompiler(catalog, dialect="duckdb").compile(
        SemanticQuery(metrics=["revenue", "orders"], time_grain="month")
    )

    assert "DATE_TRUNC('month'," in sql


@pytest.mark.parametrize(
    "time_ref",
    ["orders.order_date", "dimension:orders.order_date"],
)
def test_compiler_applies_grain_to_qualified_time_dimension(time_ref: str) -> None:
    sql = SemanticQueryCompiler(_catalog(), dialect="postgres").compile(
        SemanticQuery(
            metrics=["revenue"],
            time_dimension=time_ref,
            time_grain="month",
        )
    )

    assert """DATE_TRUNC('month', "order_date")""" in sql
    assert 'ORDER BY "order_date" ASC' in sql


def test_compiler_rejects_time_grain_for_non_time_dimension() -> None:
    catalog = _catalog()
    catalog.dimensions[0] = catalog.dimensions[0].model_copy(update={"type": "categorical"})

    with pytest.raises(ValueError, match="requires an active time dimension"):
        SemanticQueryCompiler(catalog).compile(
            SemanticQuery(
                metrics=["revenue"],
                time_dimension="region",
                time_grain="month",
            )
        )


def test_compiler_rejects_incompatible_metric_dimension() -> None:
    catalog = _catalog()
    catalog.metrics[1] = catalog.metrics[1].model_copy(update={"allowed_dimensions": ["region"]})

    with pytest.raises(ValueError, match="orders does not allow dimensions: order_date"):
        SemanticQueryCompiler(catalog).compile(
            SemanticQuery(
                metrics=["revenue", "orders"],
                time_dimension="order_date",
                time_grain="month",
            )
        )


def test_compiler_rejects_unknown_order_field() -> None:
    with pytest.raises(ValueError, match="Unknown semantic order field: missing"):
        SemanticQueryCompiler(_catalog()).compile(
            SemanticQuery(
                metrics=["revenue"],
                order_by=[{"field": "missing", "direction": "asc"}],
            )
        )


def test_clickhouse_compiler_uses_native_time_bucket() -> None:
    sql = SemanticQueryCompiler(_catalog(), dialect="clickhouse").compile(
        SemanticQuery(
            metrics=["revenue"],
            time_dimension="order_date",
            time_grain="month",
        )
    )

    assert 'toStartOfMonth("order_date") AS "order_date"' in sql


def test_derived_metric_can_reuse_another_metric_formula() -> None:
    catalog = _catalog()
    catalog.metrics.append(
        SemanticMetric(
            metric_id="metric:monthly_normalized_revenue",
            key="monthly_normalized_revenue",
            name="Monthly normalized revenue",
            type="derived",
            base_table="orders",
            formula="revenue / 12",
        )
    )

    sql = SemanticQueryCompiler(catalog, dialect="duckdb").compile(
        SemanticQuery(metrics=["monthly_normalized_revenue"])
    )

    assert '(SUM("amount")) / 12 AS "monthly_normalized_revenue"' in sql


def test_compiler_applies_metric_filter_inside_aggregation() -> None:
    catalog = _catalog()
    catalog.columns.append(
        SemanticColumn(
            column_id="column:orders.kind",
            table="orders",
            name="kind",
            dtype="text",
        )
    )
    catalog.metrics[0] = SemanticMetric.model_validate(
        {**catalog.metrics[0].model_dump(), "filters": [{"field": "kind", "op": "=", "value": "fact"}]}
    )

    sql = SemanticQueryCompiler(catalog, dialect="duckdb").compile(
        SemanticQuery(metrics=["revenue"], dimensions=["region"])
    )

    assert 'SUM(CASE WHEN "kind" = \'fact\' THEN "amount" END) AS "revenue"' in sql


def test_compiler_accepts_query_filter_matching_every_metric_leaf_filter() -> None:
    catalog = _catalog_with_common_metric_filter()
    catalog.metrics.append(
        SemanticMetric(
            metric_id="metric:monthly_index",
            key="monthly_index",
            name="Monthly index",
            type="derived",
            base_table="orders",
            formula="completion_index / 12",
            allowed_dimensions=["order_date", "scope"],
        )
    )
    sql = SemanticQueryCompiler(catalog).compile(
        SemanticQuery(
            metrics=["monthly_index"],
            time_dimension="order_date",
            time_grain="month",
            filters=[{"field": "scope", "op": "=", "value": "company"}],
        )
    )

    assert sql.count("\"scope\" = 'company'") == 2
    assert "WHERE \"scope\" = 'company'" not in sql


@pytest.mark.parametrize("partial_match", [False, True])
def test_compiler_combines_query_filter_with_fixed_metric_filter(
    partial_match: bool,
) -> None:
    catalog = _catalog_with_common_metric_filter()
    query_value = "branch"
    if partial_match:
        target = next(metric for metric in catalog.metrics if metric.key == "target_value")
        target.filters = []
        query_value = "company"

    sql = SemanticQueryCompiler(catalog).compile(
        SemanticQuery(
            metrics=["completion_index"],
            filters=[{"field": "scope", "op": "=", "value": query_value}],
        )
    )

    assert f"WHERE \"scope\" = '{query_value}'" in sql
    assert "CASE WHEN \"scope\" = 'company'" in sql


def test_compiler_allows_narrower_range_than_metric_filter() -> None:
    catalog = _catalog()
    catalog.metrics[0] = SemanticMetric.model_validate(
        {
            **catalog.metrics[0].model_dump(),
            "filters": [{"field": "order_date", "op": ">=", "value": "2020-01-01"}],
        }
    )

    sql = SemanticQueryCompiler(catalog).compile(
        SemanticQuery(
            metrics=[catalog.metrics[0].key],
            filters=[{"field": "order_date", "op": ">=", "value": "2025-01-01"}],
        )
    )

    assert "CASE WHEN \"order_date\" >= '2020-01-01'" in sql
    assert "WHERE \"order_date\" >= '2025-01-01'" in sql


@pytest.mark.parametrize(
    "formula",
    [
        "actual_value + SUM(orders.amount)",
        "actual_value + COUNT(*)",
        "actual_value + COUNT(1)",
    ],
)
def test_compiler_does_not_treat_dependency_filter_as_global_for_raw_derived_term(
    formula: str,
) -> None:
    catalog = _catalog_with_common_metric_filter()
    metric = next(metric for metric in catalog.metrics if metric.key == "completion_index")
    metric.formula = formula

    sql = SemanticQueryCompiler(catalog).compile(
        SemanticQuery(
            metrics=["completion_index"],
            filters=[{"field": "scope", "op": "=", "value": "company"}],
        )
    )

    assert "WHERE \"scope\" = 'company'" in sql


def test_compiler_does_not_inherit_fixed_filter_through_nested_raw_derived_term() -> None:
    catalog = _catalog_with_common_metric_filter()
    inner = next(metric for metric in catalog.metrics if metric.key == "completion_index")
    inner.formula = "actual_value + COUNT(*)"
    catalog.metrics.append(
        SemanticMetric(
            metric_id="metric:nested_index",
            key="nested_index",
            name="Nested index",
            type="derived",
            base_table="orders",
            formula="completion_index / 12",
            allowed_dimensions=["order_date", "scope"],
        )
    )

    sql = SemanticQueryCompiler(catalog).compile(
        SemanticQuery(
            metrics=["nested_index"],
            filters=[{"field": "scope", "op": "=", "value": "company"}],
        )
    )

    assert "WHERE \"scope\" = 'company'" in sql


def test_compiler_builds_ratio_metric() -> None:
    query = SemanticQuery(metrics=["aov"], dimensions=["region"])

    sql = SemanticQueryCompiler(_catalog(), dialect="duckdb").compile(query)

    assert 'SUM("amount") / NULLIF(COUNT(DISTINCT "order_id"), 0) AS "aov"' in sql
    assert 'GROUP BY "region"' in sql


def test_compiler_prefers_duplicate_dimensions_from_metric_table() -> None:
    catalog = _catalog()
    catalog.tables.append(
        SemanticTable(
            table_id="table:manual",
            qualified_name="manual",
            table_name="manual",
            source_kind="csv",
        )
    )
    catalog.dimensions.extend(
        [
            SemanticDimension(
                dimension_id="dimension:manual.order_date",
                name="order_date",
                table="manual",
                expr="order_date",
                type="time",
                grains=["month"],
            ),
            SemanticDimension(
                dimension_id="dimension:manual.region",
                name="region",
                table="manual",
                expr="region",
                type="categorical",
            ),
        ]
    )
    sql = SemanticQueryCompiler(catalog, dialect="postgresql").compile(
        SemanticQuery(
            metrics=["revenue"],
            dimensions=["region"],
            time_dimension="order_date",
            time_grain="month",
        )
    )

    assert 'FROM "orders"' in sql
    assert """DATE_TRUNC('month', "order_date")""" in sql
    assert '"region" AS "region"' in sql
    assert "manual" not in sql


def test_compiler_rejects_ambiguous_non_local_dimension_with_qualified_hint() -> None:
    catalog = _catalog()
    for table in ("customers", "products"):
        catalog.tables.append(
            SemanticTable(
                table_id=f"table:{table}",
                qualified_name=table,
                table_name=table,
                source_kind="csv",
            )
        )
        catalog.dimensions.append(
            SemanticDimension(
                dimension_id=f"dimension:{table}.category",
                name="category",
                table=table,
                expr="category",
                type="categorical",
            )
        )

    with pytest.raises(ValueError, match="Ambiguous semantic dimension: category"):
        SemanticQueryCompiler(catalog).compile(SemanticQuery(metrics=["revenue"], dimensions=["category"]))


def test_context_separates_retrieval_candidates_from_confirmed_metrics() -> None:
    catalog = _catalog()
    item = SemanticSearchResultItem(
        entity_type="metric",
        entity_id="metric:revenue",
        score=0.99,
    )

    candidate = SemanticContextBuilder(store=None).build_from_catalog(  # type: ignore[arg-type]
        catalog=catalog,
        query="Show business performance",
        items=[item],
    )
    exact = SemanticContextBuilder(store=None).build_from_catalog(  # type: ignore[arg-type]
        catalog=catalog,
        query="Show Revenue",
        items=[item],
    )

    assert candidate.hints["candidate_metric_keys"] == ["revenue"]
    assert candidate.hints["confirmed_metric_keys"] == []
    assert candidate.hints["definition_status"] == "candidates"
    assert "SUM(orders.amount)" not in candidate.prompt
    assert exact.hints["confirmed_metric_keys"] == ["revenue"]
    assert exact.hints["definition_status"] == "resolved"
    assert "SUM(orders.amount)" not in exact.prompt


def test_context_confirms_multiple_distinct_exact_metrics() -> None:
    context = SemanticContextBuilder(store=None).build_from_catalog(  # type: ignore[arg-type]
        catalog=_catalog(),
        query="Compare Revenue and Orders",
        items=[],
    )

    assert context.hints["definition_status"] == "resolved"
    assert context.hints["confirmed_metric_keys"] == ["revenue", "orders"]


def test_context_reports_missing_and_ambiguous_metric_definitions() -> None:
    catalog = _catalog()
    catalog.terms.append(
        SemanticTerm(
            term_id="term:service_index",
            name="Service index",
            entity_refs=["metric:service_index"],
        )
    )
    missing = SemanticContextBuilder(store=None).build_from_catalog(  # type: ignore[arg-type]
        catalog=catalog,
        query="Show Service index",
        items=[],
    )

    catalog.metrics[0] = catalog.metrics[0].model_copy(update={"synonyms": ["performance"]})
    catalog.metrics[1] = catalog.metrics[1].model_copy(update={"synonyms": ["performance"]})
    ambiguous = SemanticContextBuilder(store=None).build_from_catalog(  # type: ignore[arg-type]
        catalog=catalog,
        query="Show performance",
        items=[],
    )

    assert missing.hints["definition_status"] == "missing"
    assert missing.hints["confirmed_metric_keys"] == []
    assert ambiguous.hints["definition_status"] == "ambiguous"
    assert ambiguous.hints["confirmed_metric_keys"] == []


def test_context_does_not_execute_exact_metric_when_another_match_is_ambiguous() -> None:
    catalog = _catalog()
    catalog.metrics[0] = catalog.metrics[0].model_copy(update={"synonyms": ["performance"]})
    catalog.metrics[1] = catalog.metrics[1].model_copy(update={"synonyms": ["performance"]})

    context = SemanticContextBuilder(store=None).build_from_catalog(  # type: ignore[arg-type]
        catalog=catalog,
        query="Show Revenue and performance",
        items=[],
    )

    assert context.hints["definition_status"] == "ambiguous"
    assert "revenue" in context.hints["candidate_metric_keys"]
    assert context.hints["confirmed_metric_keys"] == []


def test_term_with_one_metric_and_other_refs_confirms_metric() -> None:
    catalog = _catalog()
    catalog.terms.append(
        SemanticTerm(
            term_id="term:performance",
            name="Business performance",
            entity_refs=["metric:revenue", "dimension:orders.region"],
        )
    )

    context = SemanticContextBuilder(store=None).build_from_catalog(  # type: ignore[arg-type]
        catalog=catalog,
        query="Show Business performance",
        items=[],
    )

    assert context.hints["definition_status"] == "resolved"
    assert context.hints["confirmed_metric_keys"] == ["revenue"]


def test_term_with_multiple_metric_refs_is_ambiguous() -> None:
    catalog = _catalog()
    catalog.terms.append(
        SemanticTerm(
            term_id="term:performance",
            name="Business performance",
            entity_refs=["metric:revenue", "metric:orders"],
        )
    )

    context = SemanticContextBuilder(store=None).build_from_catalog(  # type: ignore[arg-type]
        catalog=catalog,
        query="Show Business performance",
        items=[],
    )

    assert context.hints["definition_status"] == "ambiguous"
    assert context.hints["confirmed_metric_keys"] == []


def test_inactive_semantic_entities_are_not_retrieved_or_indexed() -> None:
    catalog = _catalog()
    catalog.metrics[0] = catalog.metrics[0].model_copy(update={"is_active": False})
    context = SemanticContextBuilder(store=None).build_from_catalog(  # type: ignore[arg-type]
        catalog=catalog,
        query="Show Revenue",
        items=[
            SemanticSearchResultItem(
                entity_type="metric",
                entity_id="metric:revenue",
                score=0.99,
            )
        ],
    )

    assert context.hints["candidate_metric_keys"] == []
    assert context.hints["confirmed_metric_keys"] == []
    assert all(chunk.payload["entity_id"] != "metric:revenue" for chunk in catalog_chunks(catalog))


def _semantic_service(context) -> SQLTableService:
    service = object.__new__(SQLTableService)
    service.semantic_hints = context.hints
    service.csv_loaded = False
    service.db_runtime_config = None
    return service


def test_typed_semantic_query_executes_only_confirmed_metric_and_keeps_provenance() -> None:
    context = SemanticContextBuilder(store=None).build_from_catalog(  # type: ignore[arg-type]
        catalog=_catalog(),
        query="Show Revenue",
        items=[],
    )
    service = _semantic_service(context)
    captured: dict[str, str] = {}

    def execute(sql: str, *, artifact_name: str | None = None, purpose: str | None = None):
        captured["sql"] = sql
        return {"items": {}, "meta": {}}

    service.execute_sql_artifact = execute  # type: ignore[method-assign]
    query = SemanticQuery(
        metrics=["revenue"],
        dimensions=["region"],
        time_dimension="order_date",
        time_grain="month",
        filters=[{"field": "order_date", "op": ">=", "value": "2025-01-01"}],
        limit=24,
    )

    payload = service.build_table_artifact(
        "Revenue by month and region",
        mode="semantic_query",
        semantic_query=query,
    )

    assert 'SUM("amount") AS "revenue"' in captured["sql"]
    assert """DATE_TRUNC('month', "order_date")""" in captured["sql"]
    assert payload["meta"]["semantic_metric"]["query"] == query.model_dump(mode="json")
    assert payload["meta"]["semantic_metric"]["metrics"][0]["formula"] == "SUM(orders.amount)"


def test_sql_tool_preserves_executed_metric_formula_for_final_answer() -> None:
    context = SemanticContextBuilder(store=None).build_from_catalog(  # type: ignore[arg-type]
        catalog=_catalog(),
        query="Show Revenue",
        items=[],
    )
    service = _semantic_service(context)
    service.execute_sql_artifact = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "items": {"monthly_revenue": pd.DataFrame({"revenue": [42.0]})},
        "meta": {},
    }
    tool = SQLTool(
        sandbox=SessionSandbox(),
    )
    tool._service = service

    tool_text, payload = tool._run(
        question="Show Revenue",
        metrics=["revenue"],
        artifact_name="monthly_revenue",
    )
    final_text = _with_semantic_metric_footer(
        "Monthly values are shown in the table.",
        [SimpleNamespace(meta=payload["meta"])],
    )

    assert "SEMANTIC METRICS EXECUTED: Revenue: SUM(orders.amount)" in tool_text
    assert "ROW_PREVIEW_FOR_LLM_CONTEXT" in tool_text
    assert "42.0" in tool_text
    assert final_text.endswith("Semantic metric: Revenue; Formula: SUM(orders.amount)")


def test_sql_tool_keeps_successive_unnamed_sql_results_in_sandbox() -> None:
    class StubService:
        def __init__(self) -> None:
            self.artifact_names: list[str] = []

        def build_table_artifact(self, _question: str, *, artifact_name: str, **_kwargs):
            self.artifact_names.append(artifact_name)
            frame = pd.DataFrame({"value": [1]})
            attach_query_metadata(frame, {"requested_sql": _question})
            return {
                "items": {artifact_name: frame},
                "meta": {},
            }

    sandbox = SessionSandbox()
    tool = SQLTool(
        sandbox=sandbox,
    )
    service = StubService()
    tool._service = service  # type: ignore[assignment]

    tool._run(sql="SELECT 1 AS value")
    tool._run(sql="SELECT 2 AS value")
    tool._run(sql="SELECT 1 AS value")
    tool._run(sql="SELECT 3 AS value", artifact_name="explicit_result")

    assert set(sandbox.get_user_scope()) == {"sql_result", "sql_result_2", "explicit_result"}
    assert service.artifact_names == [
        "sql_result",
        "sql_result_2",
        "sql_result",
        "explicit_result",
    ]


@pytest.mark.parametrize(
    ("metric_key", "expected_formula"),
    [
        ("aov", "revenue / NULLIF(orders, 0)"),
        (
            "revenue",
            "SUM(CASE WHEN orders.kind = 'fact' THEN orders.amount END)",
        ),
    ],
)
def test_semantic_artifact_reports_executed_metric_formula(
    metric_key: str,
    expected_formula: str,
) -> None:
    catalog = _catalog()
    catalog.columns.append(
        SemanticColumn(
            column_id="column:orders.kind",
            table="orders",
            name="kind",
            dtype="text",
        )
    )
    catalog.metrics[0] = SemanticMetric.model_validate(
        {
            **catalog.metrics[0].model_dump(),
            "filters": [{"field": "kind", "op": "=", "value": "fact"}],
        }
    )
    context = SemanticContextBuilder(store=None).build_from_catalog(  # type: ignore[arg-type]
        catalog=catalog,
        query=f"Show {metric_key}",
        items=[],
    )
    service = _semantic_service(context)
    service.execute_sql_artifact = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "items": {},
        "meta": {},
    }

    payload = service.build_table_artifact(
        f"Show {metric_key}",
        mode="semantic_query",
        semantic_query=SemanticQuery(metrics=[metric_key]),
    )

    assert payload["meta"]["semantic_metric"]["metrics"][0]["formula"] == expected_formula


def test_typed_semantic_query_does_not_mutate_retrieval_context() -> None:
    catalog = _catalog()
    context = SemanticContextBuilder(store=None).build_from_catalog(  # type: ignore[arg-type]
        catalog=catalog,
        query="Show business performance",
        items=[
            SemanticSearchResultItem(
                entity_type="metric",
                entity_id="metric:revenue",
                score=0.99,
            )
        ],
    )
    service = _semantic_service(context)
    service.execute_sql_artifact = lambda *_args, **_kwargs: {"items": {}, "meta": {}}  # type: ignore[method-assign]

    payload = service.build_table_artifact(
        "Show business performance",
        mode="semantic_query",
        semantic_query=SemanticQuery(metrics=["revenue"]),
    )

    assert payload["meta"]["semantic_metric"]["metric_keys"] == ["revenue"]
    assert service.semantic_hints["confirmed_metric_keys"] == []


def test_semantic_context_includes_all_retrieved_metric_candidates() -> None:
    catalog = _catalog()
    catalog.metrics.append(
        SemanticMetric(
            metric_id="metric:order_count",
            key="order_count",
            name="Order Count",
            type="simple",
            base_table="orders",
            expr="order_id",
            agg="count_distinct",
        )
    )
    context = SemanticContextBuilder(store=None).build_from_catalog(  # type: ignore[arg-type]
        catalog=catalog,
        query="Show business performance",
        items=[
            SemanticSearchResultItem(entity_type="metric", entity_id="metric:revenue", score=0.9),
            SemanticSearchResultItem(
                entity_type="metric",
                entity_id="metric:order_count",
                score=0.8,
            ),
        ],
    )

    assert "key=revenue" in context.prompt
    assert "key=order_count" in context.prompt


def test_catalog_metric_can_execute_without_runtime_confirmation() -> None:
    context = SemanticContextBuilder(store=None).build_from_catalog(  # type: ignore[arg-type]
        catalog=_catalog(),
        query="Show business performance",
        items=[
            SemanticSearchResultItem(
                entity_type="metric",
                entity_id="metric:revenue",
                score=0.99,
            )
        ],
    )
    service = _semantic_service(context)
    service.execute_sql_artifact = lambda *_args, **_kwargs: {"items": {}, "meta": {}}  # type: ignore[method-assign]

    payload = service.build_table_artifact(
        "Show Revenue",
        mode="semantic_query",
        semantic_query=SemanticQuery(metrics=["revenue"]),
    )

    assert payload["meta"]["semantic_metric"]["metric_keys"] == ["revenue"]


def test_confirmed_semantic_metric_does_not_block_regular_sql_modes() -> None:
    catalog = _catalog()
    catalog.metrics.append(
        SemanticMetric(
            metric_id="metric:weighted_revenue",
            key="weighted_revenue",
            name="Weighted Revenue",
            type="derived",
            base_table="orders",
            formula="revenue * 7.5",
            allowed_dimensions=["region", "order_date"],
        )
    )
    context = SemanticContextBuilder(store=None).build_from_catalog(  # type: ignore[arg-type]
        catalog=catalog,
        query="Show Weighted Revenue",
        items=[],
    )
    service = _semantic_service(context)
    captured: list[str] = []

    def execute(sql: str, *, artifact_name: str | None = None, purpose: str | None = None):
        captured.append(sql)
        return {"items": {}, "meta": {}}

    service.execute_sql_artifact = execute  # type: ignore[method-assign]
    service.build_table_artifact(
        "Use supporting SQL",
        mode="execute_sql",
        sql="SELECT AVG(amount) FROM orders",
    )

    assert captured == ["SELECT AVG(amount) FROM orders"]


def test_compiler_qualifies_filters_through_semantic_dimensions() -> None:
    query = SemanticQuery(
        metrics=["revenue"],
        dimensions=["region"],
        filters=[{"field": "region", "op": "=", "value": "EMEA"}],
    )

    sql = SemanticQueryCompiler(_catalog(), dialect="duckdb").compile(query)

    assert 'WHERE "region" = ' in sql


def test_compiler_rejects_query_filter_not_modeled_as_dimension() -> None:
    query = SemanticQuery(
        metrics=["revenue"],
        filters=[{"field": "amount", "op": ">", "value": 100}],
    )

    with pytest.raises(ValueError, match="require active semantic dimensions: amount"):
        SemanticQueryCompiler(_catalog(), dialect="duckdb").compile(query)


def test_compiler_rejects_unknown_filter_field() -> None:
    query = SemanticQuery(metrics=["revenue"], filters=[{"field": "missing", "op": "=", "value": "x"}])

    with pytest.raises(ValueError, match="require active semantic dimensions: missing"):
        SemanticQueryCompiler(_catalog(), dialect="duckdb").compile(query)


def test_compiler_rejects_filtering_by_fact_from_another_table() -> None:
    catalog = _catalog()
    catalog.facts.append(
        SemanticFact(
            fact_id="fact:customers.credit_limit",
            name="credit_limit",
            table="customers",
            expr="credit_limit",
        )
    )

    with pytest.raises(ValueError, match="require active semantic dimensions: credit_limit"):
        SemanticQueryCompiler(catalog).compile(
            SemanticQuery(
                metrics=["revenue"],
                filters=[{"field": "credit_limit", "op": ">", "value": 0}],
            )
        )


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
    catalog.metrics[0] = catalog.metrics[0].model_copy(
        update={
            "allowed_dimensions": [
                *catalog.metrics[0].allowed_dimensions,
                "customer_region",
            ]
        }
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

    filter_only_sql = SemanticQueryCompiler(catalog, dialect="duckdb").compile(
        SemanticQuery(
            metrics=["revenue"],
            filters=[{"field": "customer_region", "op": "=", "value": "EMEA"}],
        )
    )

    assert 'LEFT JOIN "customers" AS t1' in filter_only_sql
    assert 'WHERE t1."region" = ' in filter_only_sql
    assert "GROUP BY" not in filter_only_sql

    catalog.columns.append(
        SemanticColumn(
            column_id="column:orders.amount",
            table="orders",
            name="amount",
            dtype="double",
        )
    )
    catalog.metrics.append(
        SemanticMetric(
            metric_id="metric:qualified_revenue",
            key="qualified_revenue",
            name="Qualified revenue",
            type="derived",
            base_table="orders",
            formula="SUM(orders.amount) * 7.5",
            allowed_dimensions=["customer_region"],
        )
    )
    derived_sql = SemanticQueryCompiler(catalog, dialect="duckdb").compile(
        SemanticQuery(metrics=["qualified_revenue"], dimensions=["customer_region"])
    )

    assert 'SUM(t0."amount") * 7.5 AS "qualified_revenue"' in derived_sql
    assert "COUNT(DISTINCT" not in derived_sql


def test_compiler_supports_unicode_qualified_columns_in_derived_metric() -> None:
    catalog = SemanticCatalog(
        catalog_id="unicode",
        user_id=0,
        session_id="",
        source_key="source",
        version="2.0",
        tables=[
            SemanticTable(
                table_id="table:public.sales_ru",
                qualified_name="public.продажи",
                table_name="продажи",
                source_kind="postgres",
            )
        ],
        columns=[
            SemanticColumn(
                column_id="column:public.sales_ru.amount",
                table="public.продажи",
                name="сумма",
                dtype="double",
            )
        ],
        metrics=[
            SemanticMetric(
                metric_id="metric:adjusted_sales",
                key="adjusted_sales",
                name="Adjusted sales",
                type="derived",
                base_table="public.продажи",
                formula="SUM(public.продажи.сумма) * 7.5",
            )
        ],
    )

    sql = SemanticQueryCompiler(catalog, dialect="postgres").compile(
        SemanticQuery(metrics=["adjusted_sales"])
    )

    assert 'SUM("сумма") * 7.5 AS "adjusted_sales"' in sql
    assert 'FROM "public"."продажи"' in sql
