from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import plotly.graph_objects as go
import pytest

from backend.artifacts.execution import ExecutionArtifact, ExecutionStore
from backend.core.config import Settings
from backend.mcp.models import (
    MCPServerConfig,
    MCPServerTransport,
    MCPToolCallResult,
    MCPToolDescriptor,
)
from backend.sessions.session_store import SessionStore
from backend.tools.artifact_references import QUERY_META_ATTR
from backend.tools.context import ToolBuildContext
from backend.tools.impl.factory import SQLToolFactory
from backend.tools.impl.mcp_tool import MCPToolFactory
from backend.tools.impl.sql_tool import SQLTool, SQLToolArgs
from backend.tools.sandbox import SessionSandbox


class _RecordingMCPProvider:
    def __init__(self, result: str | MCPToolCallResult = "ok") -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result

    def call_tool(
        self,
        *,
        config: MCPServerConfig,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str | MCPToolCallResult:
        self.calls.append(
            {
                "server_id": config.server_id,
                "tool_name": tool_name,
                "arguments": arguments,
            }
        )
        return self.result


class _StaticSQLService:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def build_table_artifact(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return self.payload


def _mcp_tool(
    sandbox: SessionSandbox,
    provider: _RecordingMCPProvider,
    *,
    session_id: str = "",
    session_store: object | None = None,
    execution_store: object | None = None,
):
    config = MCPServerConfig(
        server_id="chronos",
        name="Chronos",
        transport=MCPServerTransport.streamable_http,
        url="http://127.0.0.1:8765/mcp",
    )
    descriptor = MCPToolDescriptor.from_mcp_tool(
        server_id=config.server_id,
        tool_name="forecast",
        description="Forecast a series prepared by sql_tool.",
        input_schema={
            "type": "object",
            "properties": {
                "series": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "dt": {"type": "string"},
                            "y": {"type": "number"},
                        },
                        "required": ["dt", "y"],
                        "additionalProperties": False,
                    },
                },
                "horizon": {"type": "integer"},
            },
            "required": ["series", "horizon"],
        },
    )
    return MCPToolFactory(
        config=config,
        descriptor=descriptor,
        provider=provider,
    ).build(
        type(
            "_Context",
            (),
            {
                "sandbox": sandbox,
                "trace_context": {"session_id": session_id},
                "session_store": session_store,
                "execution_store": execution_store,
            },
        )()
    )


def test_mcp_tool_expands_dataframe_artifact_before_provider_call() -> None:
    sandbox = SessionSandbox()
    sandbox.put(
        "forecast_history",
        pd.DataFrame(
            {
                "dt": pd.to_datetime(["2026-01-01", "2026-02-01"]),
                "y": [10.5, 12.0],
            }
        ),
    )
    provider = _RecordingMCPProvider()
    tool = _mcp_tool(sandbox, provider)

    assert '{"$artifact": "artifact_id"}' in tool.description
    result = tool.invoke(
        {
            "series": {"$artifact": "forecast_history"},
            "horizon": 2,
        }
    )

    assert result == "ok"
    assert provider.calls == [
        {
            "server_id": "chronos",
            "tool_name": "forecast",
            "arguments": {
                "series": [
                    {"dt": "2026-01-01T00:00:00.000", "y": 10.5},
                    {"dt": "2026-02-01T00:00:00.000", "y": 12.0},
                ],
                "horizon": 2,
            },
        }
    ]


def test_mcp_tool_loads_persisted_artifact_id_on_a_fresh_worker(tmp_path: Path) -> None:
    session_store = SessionStore(str(tmp_path), ttl_days=7)
    session = session_store.create_session("session-1")
    artifact = ExecutionArtifact(
        id="forecast-artifact",
        name="forecast_history",
        data=pd.DataFrame({"dt": ["2026-01-01"], "y": [10.5]}),
    )
    session_store.add_artifacts(session.session_id, [artifact])
    provider = _RecordingMCPProvider()
    tool = _mcp_tool(
        SessionSandbox(),
        provider,
        session_id=session.session_id,
        session_store=SessionStore(str(tmp_path), ttl_days=7),
        execution_store=ExecutionStore(session_id=session.session_id),
    )

    result = tool.invoke(
        {
            "series": {"$artifact": artifact.id},
            "horizon": 1,
        }
    )

    assert result == "ok"
    assert provider.calls[0]["arguments"]["series"] == [{"dt": "2026-01-01", "y": 10.5}]


def test_mcp_tool_publishes_structured_rows_and_figure_to_sandbox() -> None:
    sandbox = SessionSandbox()
    sandbox.put("forecast_history", pd.DataFrame({"dt": ["2026-07-01"], "y": [11.0]}))
    provider = _RecordingMCPProvider(
        MCPToolCallResult(
            structured_content={
                "rows": [{"ds": "2026-08-01", "prediction": 12.5}],
                "plot": {
                    "figure": {
                        "data": [{"type": "scatter", "x": ["2026-08-01"], "y": [12.5]}],
                        "layout": {"title": {"text": "Forecast"}},
                    }
                },
            }
        )
    )

    content, artifact = _mcp_tool(sandbox, provider)._run(
        series={"$artifact": "forecast_history"},
        horizon=1,
    )

    scope = sandbox.get_user_scope()
    assert scope["mcp__chronos__forecast"]["rows"][0]["prediction"] == 12.5
    rows = scope["mcp__chronos__forecast_rows"]
    assert pd.api.types.is_datetime64_any_dtype(rows["ds"])
    assert rows["ds"].dt.tz is None
    assert rows["ds"].dt.strftime("%Y-%m-%d").tolist() == ["2026-08-01"]
    assert rows["prediction"].tolist() == [12.5]
    assert isinstance(scope["mcp__chronos__forecast_plot_figure"], go.Figure)
    assert artifact is not None
    assert list(artifact["table"]) == ["mcp__chronos__forecast_rows"]
    assert list(artifact["plot"]) == ["mcp__chronos__forecast_plot_figure"]
    assert artifact["meta"]["lineage"]["source_artifact_names"] == ["forecast_history"]
    assert "`mcp__chronos__forecast_rows`" in content
    assert "`mcp__chronos__forecast_plot_figure`" in content
    assert "do not copy their rows into code" in content


def test_mcp_tool_rejects_missing_or_truncated_artifact() -> None:
    sandbox = SessionSandbox()
    truncated = pd.DataFrame({"dt": ["2026-01-01"], "y": [10.0]})
    truncated.attrs[QUERY_META_ATTR] = {"truncated": True}
    sandbox.put("truncated_history", truncated)
    provider = _RecordingMCPProvider()
    tool = _mcp_tool(sandbox, provider)

    with pytest.raises(ValueError, match="not found"):
        tool.invoke({"series": {"$artifact": "missing"}, "horizon": 1})
    with pytest.raises(ValueError, match="truncated"):
        tool.invoke(
            {
                "series": {"$artifact": "truncated_history"},
                "horizon": 1,
            }
        )

    assert provider.calls == []


@pytest.mark.parametrize(
    ("name", "artifact", "message"),
    [
        ("not_tabular", {"dt": ["2026-01-01"], "y": [10.0]}, "not a DataFrame"),
        (
            "too_many_rows",
            pd.DataFrame({"dt": range(1_001), "y": range(1_001)}),
            "exceeds 1000 rows",
        ),
    ],
)
def test_mcp_tool_rejects_unsafe_artifact_payloads(
    name: str,
    artifact: object,
    message: str,
) -> None:
    sandbox = SessionSandbox()
    sandbox.put(name, artifact)
    provider = _RecordingMCPProvider()

    with pytest.raises(ValueError, match=message):
        _mcp_tool(sandbox, provider).invoke({"series": {"$artifact": name}, "horizon": 1})

    assert provider.calls == []


def test_sql_tool_preserves_query_metadata_on_sandbox_dataframe() -> None:
    sandbox = SessionSandbox()
    tool = SQLTool(
        sandbox=sandbox,
    )
    tool._service = _StaticSQLService(  # type: ignore[assignment]
        {
            "items": {"forecast_history": pd.DataFrame({"dt": ["2026-01-01"], "y": [10.0]})},
            "source": {"source_type": "db_connection"},
            "recipe": [],
            "meta": {
                "query": {
                    "executed_sql": "SELECT dt, y FROM monthly_sales",
                    "truncated": False,
                }
            },
        }
    )

    tool._run_query(
        SQLToolArgs(
            mode="execute_sql",
            sql="SELECT dt, y FROM monthly_sales",
            artifact_name="forecast_history",
        )
    )
    stored = sandbox.get_user_scope()["forecast_history"]

    assert isinstance(stored, pd.DataFrame)
    assert stored.attrs[QUERY_META_ATTR] == {
        "executed_sql": "SELECT dt, y FROM monthly_sales",
        "truncated": False,
    }


def test_sql_tool_returns_complete_result_without_runtime_completion_metadata() -> None:
    tool = SQLTool(
        sandbox=SessionSandbox(),
    )
    tool._service = _StaticSQLService(  # type: ignore[assignment]
        {
            "items": {"deviation_ranking": pd.DataFrame({"branch": ["A"], "gap_pct": [5.0]})},
            "source": {"source_type": "db_connection"},
            "recipe": [],
            "meta": {},
        }
    )

    _, payload = tool._run_query(
        SQLToolArgs(
            mode="execute_sql",
            sql="SELECT 'A' AS branch, 5.0 AS gap_pct",
            artifact_name="deviation_ranking",
        )
    )

    assert "answer_artifacts" not in payload["meta"]
    assert "terminal_answer" not in payload["meta"]


def test_sql_tool_factory_uses_service_row_ceiling() -> None:
    tool = SQLToolFactory().build(ToolBuildContext(settings=Settings()))

    assert tool._service.max_rows == 1000


def test_sql_tool_does_not_present_truncated_preview_as_analysis_ready() -> None:
    sandbox = SessionSandbox()
    tool = SQLTool(
        sandbox=sandbox,
    )
    tool._service = _StaticSQLService(  # type: ignore[assignment]
        {
            "items": {"limited_rows": pd.DataFrame({"dt": ["2026-01-01"], "y": [10.0]})},
            "source": {"source_type": "db_connection"},
            "recipe": [],
            "meta": {
                "query": {
                    "executed_sql": "SELECT dt, y FROM monthly_sales",
                    "returned_rows": 200,
                    "max_rows": 200,
                    "truncated": True,
                }
            },
        }
    )

    text, _ = tool._run_query(
        SQLToolArgs(
            mode="execute_sql",
            sql="SELECT dt, y FROM monthly_sales",
            artifact_name="limited_rows",
        )
    )

    assert "TRUNCATED_RESULT" in text
    assert "Используй эти имена напрямую" not in text
    assert "complete non-overlapping partitions" in text


def test_sql_tool_marks_binding_explicit_limit_as_bounded() -> None:
    sandbox = SessionSandbox()
    tool = SQLTool(
        sandbox=sandbox,
    )
    tool._service = _StaticSQLService(  # type: ignore[assignment]
        {
            "items": {"top_rows": pd.DataFrame({"dt": ["2026-01-01", "2026-02-01"], "y": [10.0, 20.0]})},
            "source": {"source_type": "db_connection"},
            "recipe": [],
            "meta": {
                "query": {
                    "executed_sql": "SELECT dt, y FROM monthly_sales LIMIT 3",
                    "requested_limit": 2,
                    "returned_rows": 2,
                    "truncated": False,
                    "has_more_rows": True,
                }
            },
        }
    )

    text, _ = tool._run_query(
        SQLToolArgs(
            mode="execute_sql",
            sql="SELECT dt, y FROM monthly_sales LIMIT 2",
            artifact_name="top_rows",
        )
    )

    assert "BOUNDED_RESULT" in text
    assert "exact top-N" in text
    assert "Do not increase LIMIT incrementally" in text


def test_sql_tool_error_includes_bounded_source_context() -> None:
    tool = SQLTool(
        sandbox=SessionSandbox(),
    )
    tool._service = Mock()  # type: ignore[assignment]
    tool._service.build_table_artifact.side_effect = RuntimeError(
        'column "branch" does not exist\nLINE 5: branch,'
    )
    sql = """WITH complaints AS (
SELECT
  date,
  topic,
  branch,
  value
FROM mart.complaints
)
SELECT * FROM complaints"""

    text, payload = tool._run(mode="execute_sql", sql=sql)

    assert payload["status"] == "error"
    assert "SQL_CONTEXT around reported line 5" in text
    assert ">    5:   branch," in text
    assert "     7: FROM mart.complaints" in text
    assert "SELECT * FROM complaints" not in text
    assert payload["error"] == 'column "branch" does not exist\nLINE 5: branch,'
    assert payload["sql_excerpt"]
    assert len(text) < 900
