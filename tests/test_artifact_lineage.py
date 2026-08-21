from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from langchain_core.messages import ToolMessage

from backend.agent.callbacks import ToolCollector
from backend.artifacts.execution import ExecutionArtifact, ExecutionStore
from backend.sessions.session_store import SessionStore
from backend.tools.artifact_references import (
    QUERY_META_ATTR,
    attach_query_metadata,
    load_artifact_dataframe,
    materialize_artifact_inputs,
)
from backend.tools.impl.pandas_tool import PandasTool
from backend.tools.impl.plotly_tool import PlotlyTool
from backend.tools.sandbox import SessionSandbox


def test_execution_store_preserves_distinct_artifact_identity_and_provenance() -> None:
    store = ExecutionStore(session_id="session-1")
    first = store.put(
        ExecutionArtifact(
            id="first",
            name="result",
            producer_tool="pandas_tool",
            data=pd.DataFrame({"left": [1]}),
            meta={"lineage": {"source_artifact_ids": ["source-a"]}},
        )
    )
    second = store.put(
        ExecutionArtifact(
            id="second",
            name="result",
            producer_tool="pandas_tool",
            data=pd.DataFrame({"right": [1]}),
            meta={"lineage": {"source_artifact_ids": ["source-b"]}},
        )
    )

    assert first.id == "first"
    assert second.id == "second"
    assert second.version == 2
    assert first.schema is not None and first.schema.columns == ["left"]
    assert second.schema is not None and second.schema.columns == ["right"]
    assert first.parent_ids == ["source-a"]
    assert second.parent_ids == ["source-b"]


def test_pandas_merge_records_nonblocking_source_lineage() -> None:
    sandbox = SessionSandbox()
    sandbox.put("fact", pd.DataFrame({"project_id": [1], "fact": [100]}))
    sandbox.put("plan", pd.DataFrame({"project_id": [1], "plan": [80]}))
    tool = PandasTool(pd.DataFrame(), sandbox=sandbox, tool_cache_size=0)

    text, payload = tool._run(
        """
joined = fact.merge(plan, on="project_id", how="inner")
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"joined_plan_fact": joined},
}
tool_result
"""
    )
    collector = ToolCollector()
    collector.absorb_tool_message(
        "pandas_tool",
        ToolMessage(content=text, artifact=payload, tool_call_id="call-1"),
    )
    assert collector.artifacts[0].meta["lineage"]["source_table_names"] == [
        "fact",
        "plan",
    ]


def test_pandas_result_records_semantic_parent_artifact() -> None:
    source = pd.DataFrame({"month": ["2026-01"], "metric": [7.5]})
    sandbox = SessionSandbox()
    sandbox.put("semantic_result", source)
    store = ExecutionStore(session_id="session-1")
    parent = store.put(
        ExecutionArtifact(
            name="semantic_result",
            data=source,
            meta={"semantic_metric": {"metrics": [{"key": "service_index"}]}},
        )
    )
    tool = PandasTool(source, sandbox=sandbox, tool_cache_size=0)

    text, payload = tool._run(
        """
final_result = semantic_result.copy()
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"final_result": final_result},
}
"""
    )
    collector = ToolCollector(execution_store=store)
    collector.absorb_tool_message(
        "pandas_tool",
        ToolMessage(content=text, artifact=payload, tool_call_id="call-semantic"),
    )

    assert collector.artifacts[0].parent_ids == [parent.id]


def test_tool_collector_persists_each_completed_artifact_without_breaking_tool_result() -> None:
    persisted: list[str] = []

    def sink(artifacts: list[ExecutionArtifact]) -> None:
        persisted.extend(artifact.id for artifact in artifacts)

    collector = ToolCollector(artifact_sink=sink)
    collector.absorb_tool_message(
        "pandas_tool",
        ToolMessage(
            content="ok",
            artifact={"table": {"result": pd.DataFrame({"value": [1]})}},
            tool_call_id="call-persist",
        ),
    )

    assert persisted == [collector.artifacts[0].id]
    assert collector.persisted_artifact_ids == set(persisted)

    collector.artifact_sink = lambda _artifacts: (_ for _ in ()).throw(RuntimeError("storage down"))
    collector.absorb_tool_message(
        "pandas_tool",
        ToolMessage(
            content="ok",
            artifact={"table": {"second": pd.DataFrame({"value": [2]})}},
            tool_call_id="call-storage-error",
        ),
    )
    assert [artifact.name for artifact in collector.artifacts] == ["result", "second"]


def test_pandas_materializes_persisted_artifact_by_id_on_a_fresh_worker(
    tmp_path: Path,
) -> None:
    session_store = SessionStore(str(tmp_path), ttl_days=7)
    session = session_store.create_session("session-1")
    parent = ExecutionArtifact(
        id="artifact-1",
        name="semantic_result",
        data=pd.DataFrame({"value": [2, 3]}),
        meta={"semantic_metric": {"metrics": [{"key": "service_index"}]}},
    )
    session_store.add_artifacts(session.session_id, [parent])

    # A new store and sandbox simulate another Uvicorn worker or a backend restart.
    restarted_store = SessionStore(str(tmp_path), ttl_days=7)
    sandbox = SessionSandbox()
    sandbox.put("source", pd.DataFrame({"value": [99]}))
    current_store = ExecutionStore(session_id=session.session_id)
    tool = PandasTool(
        pd.DataFrame(),
        sandbox=sandbox,
        tool_cache_size=0,
        session_id=session.session_id,
        session_store=restarted_store,
        execution_store=current_store,
    )
    text, payload = tool._run(
        """
result = source.assign(doubled=source["value"] * 2)
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"derived": result},
}
""",
        input_artifacts={"source": parent.id},
    )
    collector = ToolCollector(execution_store=current_store)
    collector.absorb_tool_message(
        "pandas_tool",
        ToolMessage(content=text, artifact=payload, tool_call_id="call-restored"),
    )

    assert sandbox.get_user_scope()["derived"]["doubled"].tolist() == [4, 6]
    assert collector.artifacts[-1].parent_ids == [parent.id]
    assert current_store.get(parent.id) is not None


def test_pandas_uses_current_sandbox_artifact_when_input_reference_repeats_alias() -> None:
    sandbox = SessionSandbox()
    sandbox.put("source", pd.DataFrame({"value": [2, 3]}))
    tool = PandasTool(pd.DataFrame(), sandbox=sandbox, tool_cache_size=0)

    text, payload = tool._run(
        """
result = source.assign(doubled=source["value"] * 2)
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"derived": result},
}
""",
        input_artifacts={"source": "source"},
    )

    assert payload.get("status") != "error", text
    assert sandbox.get_user_scope()["derived"]["doubled"].tolist() == [4, 6]


def test_loading_persisted_artifact_restores_transitive_lineage(tmp_path: Path) -> None:
    session_store = SessionStore(str(tmp_path), ttl_days=7)
    session = session_store.create_session("session-1")
    root = ExecutionArtifact(
        id="root",
        name="semantic_source",
        data=pd.DataFrame({"value": [1]}),
        meta={"semantic_metric": {"metrics": [{"key": "service_index"}]}},
    )
    middle = ExecutionArtifact(
        id="middle",
        name="prepared",
        data=pd.DataFrame({"value": [2]}),
        parent_ids=[root.id],
    )
    child = ExecutionArtifact(
        id="child",
        name="final",
        data=pd.DataFrame({"value": [3]}),
        parent_ids=[middle.id],
    )
    session_store.add_artifacts(session.session_id, [root, middle, child])
    current_store = ExecutionStore(session_id=session.session_id)

    _dataframe, restored = load_artifact_dataframe(
        child.id,
        session_id=session.session_id,
        session_store=session_store,
        execution_store=current_store,
    )

    lineage = current_store.get_lineage(restored.id)
    assert [artifact.id for artifact in lineage] == ["root", "middle", "child"]
    assert lineage[0].meta["semantic_metric"]["metrics"][0]["key"] == "service_index"


def test_loading_artifact_reports_missing_or_cyclic_lineage_without_hanging(tmp_path: Path) -> None:
    session_store = SessionStore(str(tmp_path), ttl_days=7)
    session = session_store.create_session("session-1")
    first = ExecutionArtifact(
        id="first",
        name="first",
        data=pd.DataFrame({"value": [1]}),
        parent_ids=["second", "missing"],
    )
    second = ExecutionArtifact(
        id="second",
        name="second",
        data=pd.DataFrame({"value": [2]}),
        parent_ids=["first"],
    )
    session_store.add_artifacts(session.session_id, [first, second])

    _dataframe, restored = load_artifact_dataframe(
        first.id,
        session_id=session.session_id,
        session_store=session_store,
        execution_store=ExecutionStore(session_id=session.session_id),
    )

    assert restored.meta["lineage_incomplete"]["missing_parent_ids"] == ["first", "missing"]


def test_persisted_artifact_cannot_cross_session_boundary(tmp_path: Path) -> None:
    session_store = SessionStore(str(tmp_path), ttl_days=7)
    owner = session_store.create_session("owner-session")
    other = session_store.create_session("other-session")
    artifact = ExecutionArtifact(
        id="private-artifact",
        name="private_data",
        data=pd.DataFrame({"value": [42]}),
    )
    session_store.add_artifacts(owner.session_id, [artifact])
    sandbox = SessionSandbox()
    tool = PandasTool(
        pd.DataFrame(),
        sandbox=sandbox,
        tool_cache_size=0,
        session_id=other.session_id,
        session_store=session_store,
        execution_store=ExecutionStore(session_id=other.session_id),
    )

    text, payload = tool._run(
        'tool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"result": source}}',
        input_artifacts={"source": artifact.id},
    )

    assert "not found in the current session" in text
    assert payload["status"] == "error"
    assert "source" not in sandbox.get_user_scope()


def test_materialize_artifact_inputs_is_atomic_and_rejects_reserved_aliases(tmp_path: Path) -> None:
    session_store = SessionStore(str(tmp_path), ttl_days=7)
    session = session_store.create_session("session-1")
    artifact = ExecutionArtifact(
        id="artifact-1",
        name="source",
        data=pd.DataFrame({"value": [1]}),
    )
    session_store.add_artifacts(session.session_id, [artifact])
    sandbox = SessionSandbox()

    for inputs in (
        {"df": artifact.id},
        {"valid": artifact.id, "missing": "missing-artifact"},
    ):
        try:
            materialize_artifact_inputs(
                inputs,
                session_id=session.session_id,
                session_store=session_store,
                execution_store=ExecutionStore(session_id=session.session_id),
                sandbox=sandbox,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid artifact inputs must fail")

        assert sandbox.get_user_scope() == {}


def test_plotly_result_records_referenced_artifact_name() -> None:
    source = pd.DataFrame({"month": ["2026-01"], "metric": [7.5]})
    sandbox = SessionSandbox()
    sandbox.put("semantic_result", source)
    tool = PlotlyTool(source, sandbox=sandbox, tool_cache_size=0)

    hints = tool._merge_inferred_artifact_hints(
        {},
        code="figure = px.line(semantic_result, x='month', y='metric')",
        normalized_result={"metric_chart": go.Figure()},
    )

    assert hints["meta"]["lineage"]["source_artifact_names"] == ["semantic_result"]


def test_pandas_propagates_truncated_input_provenance() -> None:
    sandbox = SessionSandbox()
    limited = pd.DataFrame(
        {
            "month": ["2025-01", "2025-02"],
            "branch": ["A", "A"],
            "value": [10.0, 12.0],
        }
    )
    attach_query_metadata(
        limited,
        {
            "truncated": True,
            "max_rows": 200,
            "returned_rows": 200,
        },
    )
    sandbox.put("limited_source", limited)
    tool = PandasTool(pd.DataFrame(), sandbox=sandbox, tool_cache_size=0)

    text, payload = tool._run(
        """
result_df = limited_source.groupby("branch", as_index=False)["value"].sum()
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"derived_result": result_df},
}
"""
    )

    derived = sandbox.get_user_scope()["derived_result"]
    completeness = payload["meta"]["upstream_completeness"]
    assert "TRUNCATED_RESULT" in text
    assert completeness["truncated"] is True
    assert completeness["source_artifacts"] == ["limited_source"]
    assert derived.attrs[QUERY_META_ATTR]["truncated"] is True


def test_pandas_does_not_finish_from_a_bounded_input() -> None:
    sandbox = SessionSandbox()
    bounded = pd.DataFrame({"branch": ["A"], "value": [10.0]})
    attach_query_metadata(bounded, {"truncated": False, "has_more_rows": True})
    sandbox.put("bounded_source", bounded)
    tool = PandasTool(pd.DataFrame(), sandbox=sandbox, tool_cache_size=0)

    _text, payload = tool._run(
        """
result_df = bounded_source.groupby("branch", as_index=False)["value"].sum()
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"derived_result": result_df},
}
"""
    )

    completeness = payload["meta"]["upstream_completeness"]
    assert completeness["truncated"] is True
    assert completeness["source_artifacts"] == ["bounded_source"]


def test_pandas_output_name_does_not_inherit_stale_self_lineage() -> None:
    sandbox = SessionSandbox()
    stale = pd.DataFrame({"value": [1.0]})
    attach_query_metadata(stale, {"truncated": True})
    sandbox.put("result", stale)
    sandbox.put("complete_source", pd.DataFrame({"value": [2.0]}))
    tool = PandasTool(pd.DataFrame(), sandbox=sandbox, tool_cache_size=0)

    text, payload = tool._run(
        """
result = complete_source.copy()
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"result": result},
}
"""
    )

    published = sandbox.get_user_scope()["result"]
    assert "TRUNCATED_RESULT" not in text
    assert "upstream_completeness" not in payload.get("meta", {})
    assert published.attrs.get(QUERY_META_ATTR, {}).get("truncated") is not True
