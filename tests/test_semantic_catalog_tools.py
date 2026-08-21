from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.runtime import Runtime

from backend.agent.tool_loop import _as_tool_messages, _build_tool_node
from backend.data_access.data_catalog import CatalogColumn, CatalogTable, DataCatalogSnapshot
from backend.data_access.semantic_catalog_service import SemanticCatalogService
from backend.data_access.semantic_generation_service import (
    SemanticCatalogGenerationResponse,
    SemanticCatalogGenerationSummary,
)
from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticContextResult,
    SemanticSearchResultItem,
    SemanticValidationResult,
)
from backend.data_access.semantic_query import SemanticQuery, SemanticQueryCompiler
from backend.tools.context import ToolBuildContext
from backend.tools.impl.factory import SemanticCatalogReadToolFactory
from backend.tools.impl.semantic_catalog_tool import (
    SemanticCatalogEditTool,
    SemanticCatalogGenerateTool,
    SemanticCatalogReadTool,
)
from backend.tools.registry import ToolRegistry
from tests.in_memory_semantic_store import SemanticSessionStore as SessionStore


class _Settings:
    tool_exec_timeout_sec = 30
    tool_cache_size = 8
    llm_base_url = ""
    llm_model = ""
    llm_api_key = ""
    llm_enable_thinking = False
    llm_chat_template_kwargs_enabled = False
    llm_provider = "openai"
    storage_dir = ""


class _VectorStore:
    @property
    def enabled(self) -> bool:
        return False

    def upsert_catalog(self, catalog) -> None:
        del catalog


class _GenerationService:
    def __init__(self) -> None:
        self.called = False

    def generate(self, **kwargs):
        self.called = True
        raise AssertionError(f"generate should not be called: {kwargs}")


def _snapshot() -> DataCatalogSnapshot:
    return DataCatalogSnapshot(
        built_at="2026-07-20T00:00:00+00:00",
        source_fingerprint="fp-tools",
        tables=[
            CatalogTable(
                qualified_name="sales",
                table_name="sales",
                source_kind="csv_session",
                columns=[
                    CatalogColumn(name="amount", dtype="numeric"),
                    CatalogColumn(name="region", dtype="string"),
                ],
            )
        ],
    )


def _service(tmp_path: Path) -> tuple[SemanticCatalogService, str]:
    store = SessionStore(str(tmp_path), ttl_days=1)
    state = store.create_session()
    store.bind_csv_source(state.session_id, filename="sales.csv", source_ref_id="sales-source")
    store.save_data_catalog(state.session_id, _snapshot())
    service = SemanticCatalogService(store=store, vector_store=_VectorStore())
    service.refresh(session_id=state.session_id, user_id=7)
    return service, state.session_id


def _create_fresh_session_for_same_source(service: SemanticCatalogService) -> str:
    state = service.store.create_session()
    service.store.bind_csv_source(state.session_id, filename="sales.csv", source_ref_id="sales-source")
    service.store.save_data_catalog(state.session_id, _snapshot())
    return state.session_id


def _invoke_edit_tool_call(edit_tool: SemanticCatalogEditTool, args: dict) -> str:
    node = _build_tool_node([edit_tool], tool_collector=None)
    messages = _as_tool_messages(
        node.invoke(
            [
                {
                    "name": "semantic_catalog_edit_tool",
                    "args": args,
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
            runtime=Runtime(),
        )
    )

    assert messages[0].status == "success"
    return str(messages[0].content)


def test_semantic_catalog_tools_are_langchain_tools_and_parallel_flags(tmp_path: Path) -> None:
    service, session_id = _service(tmp_path)
    registry = ToolRegistry.from_services()
    tools = registry.build_tools(
        ToolBuildContext(
            settings=_Settings(),  # type: ignore[arg-type]
            allowed_tool_keys={
                "semantic_catalog_read_tool",
                "semantic_catalog_edit_tool",
                "semantic_catalog_generate_tool",
            },
            semantic_catalog_service=service,
            semantic_generation_service=object(),
            trace_context={"session_id": session_id, "user_id": "7"},
        )
    )
    by_name = {tool.name: tool for tool in tools}

    assert isinstance(by_name["semantic_catalog_read_tool"], SemanticCatalogReadTool)
    assert isinstance(by_name["semantic_catalog_edit_tool"], SemanticCatalogEditTool)
    assert isinstance(by_name["semantic_catalog_generate_tool"], SemanticCatalogGenerateTool)
    assert by_name["semantic_catalog_read_tool"].__class__.parallel_safe is True
    assert by_name["semantic_catalog_edit_tool"].__class__.parallel_safe is False
    assert by_name["semantic_catalog_generate_tool"].__class__.parallel_safe is False


def test_semantic_catalog_read_tool_cannot_override_bound_connection() -> None:
    calls: list[tuple[str, str]] = []

    class CatalogService:
        def status_for_connection(self, *, connection_id: str):
            calls.append(("connection", connection_id))
            return {"status": "ready"}

    tool = SemanticCatalogReadTool(
        catalog_service=CatalogService(),
        session_id="session-1",
        user_id=7,
        connection_id="conn-A",
    )

    with pytest.raises(ValueError, match="connection_id"):
        tool.invoke({"action": "status", "connection_id": "conn-B"})

    assert calls == []


def test_semantic_catalog_read_factory_uses_resolved_db_runtime_only() -> None:
    calls: list[tuple[str, str]] = []

    class CatalogService:
        def status_for_connection(self, *, connection_id: str):
            calls.append(("connection", connection_id))
            return {"status": "ready"}

        def load_for_session(self, *, session_id: str, user_id: int):
            calls.append(("session", session_id))
            return None

    service = CatalogService()
    factory = SemanticCatalogReadToolFactory()
    base_context = dict(
        settings=_Settings(),
        semantic_catalog_service=service,
        trace_context={
            "session_id": "session-1",
            "user_id": 7,
            "db_connection_id": "untrusted-trace-id",
            "session_source": {"source_type": "csv", "source_ref_id": "csv-source"},
        },
    )

    csv_tool = factory.build(ToolBuildContext(**base_context))
    csv_tool.invoke({"action": "status"})
    db_tool = factory.build(
        ToolBuildContext(
            **base_context,
            tool_db_runtime=SimpleNamespace(connection_id="conn-A"),
        )
    )
    db_tool.invoke({"action": "status"})

    assert calls == [("session", "session-1"), ("connection", "conn-A")]


def test_semantic_catalog_read_and_edit_tools_use_existing_service(tmp_path: Path) -> None:
    service, session_id = _service(tmp_path)
    read_tool = SemanticCatalogReadTool(
        catalog_service=service,
        session_id=session_id,
        user_id=7,
    )
    edit_tool = SemanticCatalogEditTool(
        catalog_service=service,
        session_id=session_id,
        user_id=7,
    )

    tables = json.loads(read_tool.invoke({"action": "list_tables"}))
    created = json.loads(
        edit_tool.invoke(
            {
                "action": "create_metric",
                "metric": {
                    "key": "avg_amount",
                    "name": "Average amount",
                    "type": "simple",
                    "base_table": "sales",
                    "expr": "amount",
                    "agg": "avg",
                    "allowed_dimensions": ["region"],
                },
            }
        )
    )
    validation = json.loads(read_tool.invoke({"action": "validate"}))
    catalog: SemanticCatalog | None = service.load_for_session(session_id=session_id, user_id=7)

    assert tables[0]["qualified_name"] == "sales"
    assert created["key"] == "avg_amount"
    assert validation["errors"] == []
    assert catalog is not None
    assert any(metric.key == "avg_amount" for metric in catalog.metrics)


def test_semantic_catalog_resolve_returns_full_metric_contract(tmp_path: Path) -> None:
    service, session_id = _service(tmp_path)
    edit_tool = SemanticCatalogEditTool(
        catalog_service=service,
        session_id=session_id,
        user_id=7,
    )
    edit_tool.invoke(
        {
            "action": "create_metric",
            "metric": {
                "key": "avg_amount",
                "name": "Average amount",
                "type": "simple",
                "base_table": "sales",
                "expr": "amount",
                "agg": "avg",
            },
        }
    )
    read_tool = SemanticCatalogReadTool(
        catalog_service=service,
        session_id=session_id,
        user_id=7,
    )

    result = json.loads(read_tool.invoke({"action": "resolve", "query": "avg_amount"}))

    assert result["matched"] is True
    assert result["definition_status"] == "resolved"
    assert result["confirmed_metric_keys"] == ["avg_amount"]
    assert result["candidate_metric_keys"] == ["avg_amount"]
    assert result["execution_mode"] == "semantic_query"
    assert result["terms"] == []
    assert result["relationships"] == []
    assert result["candidates"] == []
    assert result["metrics"][0] == {
        "key": "avg_amount",
        "name": "Average amount",
        "type": "simple",
        "base_table": "sales",
        "expr": "amount",
        "agg": "avg",
        "numerator": None,
        "denominator": None,
        "formula": "AVG(sales.amount)",
        "format": "number",
        "default_time_dimension": None,
        "allowed_dimensions": [],
        "filters": [],
    }


def test_semantic_catalog_resolve_guides_cross_table_execution() -> None:
    class CatalogService:
        def search(self, **_kwargs) -> SemanticContextResult:
            metrics = [
                {"key": "order_total", "name": "Order total", "base_table": "orders"},
                {"key": "customer_score", "name": "Customer score", "base_table": "customers"},
            ]
            return SemanticContextResult(
                status="ready",
                hints={
                    "definition_status": "resolved",
                    "confirmed_metric_keys": ["order_total", "customer_score"],
                    "candidate_metric_keys": ["order_total", "customer_score"],
                    "metrics": metrics,
                    "terms": [],
                    "relationships": [],
                    "catalog": {
                        "relationships": [
                            {
                                "from_table": "orders",
                                "from_column": "customer_id",
                                "to_table": "customers",
                                "to_column": "customer_id",
                                "cardinality": "many_to_one",
                                "description": "Each order belongs to one customer.",
                            }
                        ]
                    },
                },
            )

    tool = SemanticCatalogReadTool(
        catalog_service=CatalogService(),
        session_id="session-1",
        user_id=7,
    )

    result = json.loads(tool.invoke({"action": "resolve", "query": "compare metrics"}))

    assert result["execution_mode"] == "execute_sql"
    assert list(result).index("execution_mode") < list(result).index("metrics")
    assert "allowed_dimensions support the requested join grain" in result["execution_note"]
    assert "Do not call semantic_query" in result["execution_note"]
    assert result["relationships"][0]["from_table"] == "orders"


def test_semantic_catalog_resolve_returns_term_and_relationship_definitions() -> None:
    class CatalogService:
        def search(self, **_kwargs) -> SemanticContextResult:
            return SemanticContextResult(
                status="ready",
                hints={
                    "definition_status": "not_found",
                    "metric_resolution_status": "not_found",
                    "term_resolution_status": "resolved",
                    "confirmed_metric_keys": [],
                    "candidate_metric_keys": [],
                    "metrics": [],
                    "terms": [
                        {
                            "name": "Turnover pulse",
                            "description": "Internal service indicator.",
                            "synonyms": ["pulse"],
                            "entity_refs": ["metric:turnover_pulse"],
                        }
                    ],
                    "relationships": [
                        {
                            "from_table": "orders",
                            "from_column": "customer_id",
                            "to_table": "customers",
                            "to_column": "customer_id",
                            "cardinality": "many_to_one",
                            "description": "Each order belongs to one customer.",
                        }
                    ],
                },
            )

    tool = SemanticCatalogReadTool(
        catalog_service=CatalogService(),
        session_id="session-1",
        user_id=7,
    )

    result = json.loads(tool.invoke({"action": "resolve", "query": "turnover pulse"}))

    assert result["matched"] is True
    assert result["metric_resolution_status"] == "not_found"
    assert result["term_resolution_status"] == "resolved"
    assert result["terms"] == [
        {
            "name": "Turnover pulse",
            "description": "Internal service indicator.",
            "synonyms": ["pulse"],
            "entity_refs": ["metric:turnover_pulse"],
        }
    ]
    assert result["relationships"] == [
        {
            "from_table": "orders",
            "from_column": "customer_id",
            "to_table": "customers",
            "to_column": "customer_id",
            "cardinality": "many_to_one",
            "description": "Each order belongs to one customer.",
        }
    ]


def test_semantic_catalog_search_returns_compact_ranked_candidates() -> None:
    class CatalogService:
        def search(self, **_kwargs) -> SemanticContextResult:
            return SemanticContextResult(
                status="ready",
                prompt="large internal prompt",
                items=[
                    SemanticSearchResultItem(
                        entity_type="term",
                        entity_id="term:turnover_pulse",
                        score=0.91,
                    ),
                    SemanticSearchResultItem(
                        entity_type="metric",
                        entity_id="metric:avg_amount",
                        score=0.72,
                    ),
                    SemanticSearchResultItem(
                        entity_type="dimension",
                        entity_id="dimension:sales.region",
                        score=0.65,
                    ),
                ],
                hints={
                    "catalog": {
                        "terms": [
                            {
                                "term_id": "term:turnover_pulse",
                                "name": "Turnover pulse",
                                "description": "Internal service indicator.",
                                "synonyms": ["pulse"],
                                "entity_refs": ["metric:turnover_pulse"],
                            }
                        ],
                        "metrics": [
                            {
                                "metric_id": "metric:avg_amount",
                                "key": "avg_amount",
                                "name": "Average amount",
                                "description": "Average observed amount.",
                                "type": "simple",
                                "base_table": "sales",
                                "formula": "AVG(sales.amount)",
                                "synonyms": [],
                            }
                        ],
                        "dimensions": [
                            {
                                "dimension_id": "dimension:sales.region",
                                "name": "Region",
                                "table": "sales",
                                "expr": "region",
                                "type": "categorical",
                                "grains": [],
                                "description": "Sales region.",
                                "synonyms": [],
                            }
                        ],
                    }
                },
            )

    tool = SemanticCatalogReadTool(
        catalog_service=CatalogService(),
        session_id="session-1",
        user_id=7,
    )

    result = json.loads(tool.invoke({"action": "search", "query": "turnover pulse"}))

    assert result == {
        "status": "ready",
        "matched": True,
        "query": "turnover pulse",
        "candidates": [
            {
                "entity_type": "term",
                "entity_id": "term:turnover_pulse",
                "score": 0.91,
                "name": "Turnover pulse",
                "description": "Internal service indicator.",
                "synonyms": ["pulse"],
                "entity_refs": ["metric:turnover_pulse"],
            },
            {
                "entity_type": "metric",
                "entity_id": "metric:avg_amount",
                "score": 0.72,
                "key": "avg_amount",
                "name": "Average amount",
                "description": "Average observed amount.",
                "type": "simple",
                "base_table": "sales",
            },
            {
                "entity_type": "dimension",
                "entity_id": "dimension:sales.region",
                "score": 0.65,
                "name": "Region",
                "table": "sales",
                "expr": "region",
                "type": "categorical",
                "description": "Sales region.",
            },
        ],
    }


def test_semantic_catalog_edit_tool_accepts_schema_metric_payload(tmp_path: Path) -> None:
    service, session_id = _service(tmp_path)
    edit_tool = SemanticCatalogEditTool(
        catalog_service=service,
        session_id=session_id,
        user_id=7,
    )

    created = json.loads(
        _invoke_edit_tool_call(
            edit_tool,
            {
                "action": "create_metric",
                "metric": {
                    "key": "avg_amount_from_agent",
                    "name": "Average amount from agent",
                    "type": "simple",
                    "base_table": "sales",
                    "expr": "amount",
                    "agg": "avg",
                    "allowed_dimensions": ["region"],
                },
            },
        )
    )
    fresh_session_id = _create_fresh_session_for_same_source(service)
    fresh_catalog = service.load_for_session(session_id=fresh_session_id, user_id=7)

    assert created["key"] == "avg_amount_from_agent"
    assert created["formula"] == "AVG(sales.amount)"
    assert fresh_catalog is not None
    assert any(metric.key == "avg_amount_from_agent" for metric in fresh_catalog.metrics)


def test_semantic_catalog_edit_tool_accepts_json_string_metric_payload(tmp_path: Path) -> None:
    service, session_id = _service(tmp_path)
    edit_tool = SemanticCatalogEditTool(
        catalog_service=service,
        session_id=session_id,
        user_id=7,
    )

    created = json.loads(
        _invoke_edit_tool_call(
            edit_tool,
            {
                "action": "create_metric",
                "metric": json.dumps(
                    {
                        "key": "avg_amount_from_json_string",
                        "name": "Average amount from JSON string",
                        "type": "simple",
                        "base_table": "sales",
                        "expr": "amount",
                        "agg": "avg",
                        "allowed_dimensions": ["region"],
                    }
                ),
            },
        )
    )
    fresh_session_id = _create_fresh_session_for_same_source(service)
    fresh_catalog = service.load_for_session(session_id=fresh_session_id, user_id=7)

    assert created["key"] == "avg_amount_from_json_string"
    assert created["formula"] == "AVG(sales.amount)"
    assert fresh_catalog is not None
    assert any(metric.key == "avg_amount_from_json_string" for metric in fresh_catalog.metrics)


def test_semantic_catalog_edit_tool_accepts_schema_formula_metric_and_compiles_in_fresh_session(
    tmp_path: Path,
) -> None:
    service, session_id = _service(tmp_path)
    edit_tool = SemanticCatalogEditTool(
        catalog_service=service,
        session_id=session_id,
        user_id=7,
    )

    created = json.loads(
        _invoke_edit_tool_call(
            edit_tool,
            {
                "action": "create_metric",
                "metric": {
                    "key": "amount_per_row_from_agent",
                    "name": "Amount per row from agent",
                    "type": "derived",
                    "base_table": "sales",
                    "formula": "SUM(amount) / NULLIF(COUNT(amount), 0)",
                    "allowed_dimensions": ["region"],
                },
            },
        )
    )
    fresh_session_id = _create_fresh_session_for_same_source(service)
    fresh_catalog = service.load_for_session(session_id=fresh_session_id, user_id=7)

    assert created["key"] == "amount_per_row_from_agent"
    assert fresh_catalog is not None
    metric_keys = {metric.key for metric in fresh_catalog.metrics}
    assert "amount_per_row_from_agent" in metric_keys
    sql = SemanticQueryCompiler(fresh_catalog, dialect="duckdb").compile(
        SemanticQuery(metrics=["amount_per_row_from_agent"], dimensions=["region"])
    )
    assert 'SUM(amount) / NULLIF(COUNT(amount), 0) AS "amount_per_row_from_agent"' in sql
    assert 'GROUP BY "region"' in sql


def test_semantic_catalog_edit_tool_raises_tool_errors(tmp_path: Path) -> None:
    service, session_id = _service(tmp_path)
    edit_tool = SemanticCatalogEditTool(
        catalog_service=service,
        session_id=session_id,
        user_id=7,
    )

    with pytest.raises(ValueError, match="metric is required"):
        edit_tool.invoke({"action": "create_metric"})


def test_semantic_catalog_edit_tool_errors_become_failed_tool_messages(tmp_path: Path) -> None:
    service, session_id = _service(tmp_path)
    edit_tool = SemanticCatalogEditTool(
        catalog_service=service,
        session_id=session_id,
        user_id=7,
    )
    node = _build_tool_node([edit_tool], tool_collector=None)

    messages = _as_tool_messages(
        node.invoke(
            [
                {
                    "name": "semantic_catalog_edit_tool",
                    "args": {"action": "create_metric"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
            runtime=Runtime(),
        )
    )

    assert messages[0].status == "error"
    assert "Tool error:" in messages[0].content
    assert "metric is required" in messages[0].content


def test_semantic_catalog_generate_tool_requires_apply_confirmation() -> None:
    generation_service = _GenerationService()
    tool = SemanticCatalogGenerateTool(
        generation_service=generation_service,
        session_id="session-1",
        user_id=7,
    )

    result = json.loads(tool.invoke({}))

    assert result["status"] == "confirmation_required"
    assert generation_service.called is False


def test_semantic_catalog_generate_tool_returns_compact_applied_result() -> None:
    class SuccessfulGenerationService:
        def generate(self, **kwargs):
            assert kwargs["request"].sample_rows == 0
            assert kwargs["request"].max_tables == 4
            assert kwargs["request"].ensure_metrics is True
            return SemanticCatalogGenerationResponse(
                catalog=SemanticCatalog(
                    catalog_id="cat-1",
                    source_key="db_connection:conn-1",
                    status="ready",
                    published_version=3,
                    validation=SemanticValidationResult(quality_score=0.8),
                ),
                summary=SemanticCatalogGenerationSummary(
                    tables_scanned=4,
                    metrics_added=2,
                ),
            )

    tool = SemanticCatalogGenerateTool(
        generation_service=SuccessfulGenerationService(),
        session_id="session-1",
        user_id=7,
    )

    result = json.loads(tool.invoke({"apply": True, "sample_rows": 0, "max_tables": 4}))

    assert result["status"] == "applied"
    assert result["applied"] is True
    assert result["published_version"] == 3
    assert result["summary"]["metrics_added"] == 2
    assert result["validation"]["quality_score"] == 0.8
    assert "catalog" not in result
