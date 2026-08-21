from __future__ import annotations

import json

import pandas as pd

from chronos_mcp.adapters import (
    ChronosRuntime,
    ChronosSettings,
    DataGatewaySettings,
    HttpDataGateway,
)
from chronos_mcp.application import ChronosApplication
from chronos_mcp.contracts import (
    Aggregation,
    Covariates,
    ForecastRequest,
    Frequency,
    InlineSource,
    MissingPolicy,
    Target,
)
from chronos_mcp.preparation import TableReadPlan


class FakePipeline:
    def __init__(self) -> None:
        self.context = None
        self.arguments = None

    def predict_df(self, context, **arguments):
        self.context = context
        self.arguments = arguments
        return pd.DataFrame(
            [
                {
                    "id": "series-0",
                    "timestamp": "2026-03-01",
                    "predictions": 12.0,
                    "0.1": 10.0,
                    "0.5": 12.0,
                    "0.9": 14.0,
                }
            ]
        )


class FakeResponse:
    def __init__(self, payload) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def read(self):
        return json.dumps(self.payload).encode()


def test_chronos_adapter_calls_predict_df_and_preserves_horizon_one_shape() -> None:
    pipeline = FakePipeline()
    runtime = ChronosRuntime(
        ChronosSettings(model_revision="pinned"),
        pipeline_loader=lambda _settings: pipeline,
    )
    application = ChronosApplication(runtime=runtime)
    request = ForecastRequest(
        source=InlineSource(
            rows=[
                {"ts": "2026-01-01", "y": 10.0, "promotion": 0},
                {"ts": "2026-02-01", "y": 11.0, "promotion": 1},
                {"ts": "2026-03-01", "y": None, "promotion": 1},
            ]
        ),
        time_column="ts",
        targets=[Target(name="sales", column="y", aggregation=Aggregation.none)],
        horizon=1,
        frequency=Frequency.month_start,
        missing_policy=MissingPolicy.error,
        covariates=Covariates(future_columns=["promotion"]),
    )

    response = application.forecast(request)

    assert len(response.rows) == 1
    assert response.rows[0].prediction == 12.0
    assert pipeline.arguments["prediction_length"] == 1
    assert pipeline.arguments["quantile_levels"] == [0.1, 0.5, 0.9]
    assert list(pipeline.context.columns) == ["id", "timestamp", "target", "promotion"]
    assert pd.api.types.is_datetime64_dtype(pipeline.context["timestamp"])
    assert pipeline.context["timestamp"].dt.tz is None
    assert pipeline.arguments["future_df"].to_dict(orient="records") == [
        {
            "id": "series-0",
            "timestamp": pd.Timestamp("2026-03-01"),
            "promotion": 1,
        }
    ]


def test_data_gateway_receives_typed_plan_without_sql_or_credentials() -> None:
    captured = {}

    def opener(request, *, timeout):
        captured["body"] = json.loads(request.data)
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "rows": [{"ts": "2026-01-01", "y": 10}],
                "query": {
                    "sql": "SELECT date, value FROM allowed_table",
                    "parameter_count": 0,
                    "fingerprint": "sha256:test",
                },
            }
        )

    gateway = HttpDataGateway(
        DataGatewaySettings(
            base_url="https://gateway.internal",
            token="service-token",
            timeout_seconds=5,
        ),
        opener=opener,
    )
    response = gateway(
        TableReadPlan(
            connection_id="demo",
            schema_name="analytics",
            table="sales",
            columns=("ts", "y"),
            filter_payload={"column": "region", "op": "eq", "value": "Moscow"},
            history_start=None,
            history_end=None,
            horizon=3,
            frequency=Frequency.month_start,
            future_columns=(),
            max_rows=100,
        )
    )

    assert response.rows == [{"ts": "2026-01-01", "y": 10}]
    assert captured["body"]["connection_id"] == "demo"
    assert captured["body"]["frequency"] == "month_start"
    assert "sql" not in captured["body"]
    assert "password" not in captured["body"]
    assert captured["headers"]["Authorization"] == "Bearer service-token"
