from __future__ import annotations

import anyio
import pytest
from starlette.testclient import TestClient

from chronos_mcp import __main__
from chronos_mcp.application import ModelForecast, ModelMetadata, ModelPoint
from chronos_mcp.server import ServerSettings, create_server, run


class FakeRuntime:
    def predict(self, request):
        context = request.contexts[0]
        return ModelForecast(
            points=[
                ModelPoint(
                    series_id=context.series_id,
                    target=context.target,
                    timestamp=context.future_timestamps[0],
                    prediction=12.0,
                    quantiles={0.1: 10.0, 0.5: 12.0, 0.9: 14.0},
                )
            ],
            metadata=ModelMetadata(
                library_version="test",
                model_alias="fake",
                model_id="fake/model",
                model_revision="test-revision",
                family="fake",
                capabilities=["quantiles"],
                inference_ms=1,
                cached=True,
            ),
        )

    def capabilities(self):
        return []


def call(awaitable):
    return anyio.run(lambda: awaitable)


def forecast_arguments(rows):
    return {
        "source": {"kind": "inline", "rows": rows},
        "time_column": "ts",
        "targets": [{"name": "sales", "column": "y", "aggregation": "none"}],
        "horizon": 1,
        "frequency": "month_start",
        "missing_policy": "error",
    }


def test_console_entrypoint_calls_server_run() -> None:
    assert __main__.main is run


def test_server_exposes_typed_forecast_backtest_and_capabilities_tools() -> None:
    server = create_server(runtime=FakeRuntime(), settings=ServerSettings(transport="stdio"))

    tools = {tool.name: tool for tool in call(server.list_tools())}

    assert set(tools) == {"forecast", "backtest", "capabilities"}
    assert "source" in tools["forecast"].inputSchema["properties"]
    source_schema = tools["forecast"].inputSchema["properties"]["source"]
    assert source_schema == {"$ref": "#/$defs/InlineSource"}
    target_schema = tools["forecast"].inputSchema["$defs"]["Target"]["properties"]
    assert "source row field" in target_schema["column"]["description"]
    assert target_schema["column"]["examples"] == ["y"]
    assert "question" not in tools["forecast"].inputSchema["properties"]
    assert "sql" not in tools["forecast"].inputSchema["properties"]
    assert tools["forecast"].outputSchema is not None
    assert "every request to predict or forecast future time-series values" in tools["forecast"].description
    assert "aggregate to the requested frequency" in tools["forecast"].description
    assert "source completeness" in tools["forecast"].description
    assert "ready-to-use Plotly figure" in tools["forecast"].description
    assert "Do not " not in tools["forecast"].description
    assert "Never " not in tools["forecast"].description


def test_server_returns_structured_success_and_typed_tool_error() -> None:
    server = create_server(runtime=FakeRuntime(), settings=ServerSettings(transport="stdio"))

    success = call(
        server.call_tool(
            "forecast",
            forecast_arguments(
                [
                    {"ts": "2026-01-01", "y": 10.0},
                    {"ts": "2026-02-01", "y": 11.0},
                ]
            ),
        )
    )
    failure = call(
        server.call_tool(
            "forecast",
            forecast_arguments(
                [
                    {"ts": "2026-01-01", "y": 10.0},
                    {"ts": "2026-03-01", "y": 11.0},
                ]
            ),
        )
    )

    assert success.isError is False
    assert success.structuredContent["status"] == "ok"
    assert len(success.structuredContent["rows"]) == 1
    assert failure.isError is True
    assert failure.structuredContent["status"] == "error"
    assert failure.structuredContent["error"]["code"] == "MISSING_PERIODS"


def test_http_transport_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHRONOS_MCP_TRANSPORT", "streamable-http")
    monkeypatch.delenv("CHRONOS_MCP_API_KEY", raising=False)
    monkeypatch.delenv("CHRONOS_MCP_API_KEY_FILE", raising=False)

    with pytest.raises(ValueError, match="CHRONOS_MCP_API_KEY"):
        ServerSettings.from_env()


def test_http_transport_reads_api_key_from_docker_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    secret_file = tmp_path / "chronos_key"
    secret_file.write_text("docker-secret\n", encoding="utf-8")
    monkeypatch.setenv("CHRONOS_MCP_TRANSPORT", "streamable-http")
    monkeypatch.setenv("CHRONOS_MCP_API_KEY_FILE", str(secret_file))

    settings = ServerSettings.from_env()

    assert settings.api_key == "docker-secret"


def test_http_transport_accepts_only_its_bearer_key() -> None:
    server = create_server(
        runtime=FakeRuntime(),
        settings=ServerSettings(
            transport="streamable-http",
            api_key="chronos-secret-key",
            public_url="http://127.0.0.1:8810/mcp",
        ),
    )
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"},
        },
    }
    headers = {"Accept": "application/json, text/event-stream"}

    with TestClient(server.streamable_http_app(), base_url="http://127.0.0.1:8810") as client:
        missing = client.post("/mcp", json=request, headers=headers)
        wrong = client.post(
            "/mcp",
            json=request,
            headers={**headers, "Authorization": "Bearer wrong-key"},
        )
        accepted = client.post(
            "/mcp",
            json=request,
            headers={**headers, "Authorization": "Bearer chronos-secret-key"},
        )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert accepted.status_code == 200
