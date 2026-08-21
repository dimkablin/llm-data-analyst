from __future__ import annotations

from unittest.mock import patch

from backend.integrations import (
    ForecastConfig,
    ForecastIntegrationService,
    ForecastQueryResult,
)
from backend.tools.impl.forecast_tool import ForecastTool


def _service() -> ForecastIntegrationService:
    return ForecastIntegrationService(
        ForecastConfig(
            enabled=True,
            base_url="https://forecast.example",
            predict_endpoint="/v1/forecast",
            timeout_sec=60.0,
            horizon_default=12,
            backend_api_url="http://backend:8000/v1",
            llm_base_url="http://llm.example",
            llm_api_key="test-key",
            llm_model="test-model",
        )
    )


def test_forecast_tool_passes_explicit_horizon_and_returns_typed_artifacts() -> None:
    service = _service()
    tool = ForecastTool(forecast_service=service)
    result = ForecastQueryResult(
        question="forecast attrition",
        horizon=2,
        model_name="chronos",
        summary=None,
        forecast_rows=[
            {"ts": "2026-08-01", "yhat": 21, "lower": 18, "upper": 24},
            {"ts": "2026-09-01", "yhat": 23, "lower": 19, "upper": 27},
        ],
        plotly_figure=None,
        warnings=[],
        request_params={"message": "forecast attrition", "fh": 2},
    )

    with patch.object(service, "run_forecast", return_value=result) as run_forecast:
        text, payload = tool._run(
            question="forecast attrition",
            horizon=2,
            artifact_name="attrition_forecast",
            plot_artifact_name="attrition_forecast_chart",
        )

    prepared_question = run_forecast.call_args.args[0]
    assert prepared_question.startswith("forecast attrition")
    assert run_forecast.call_args.kwargs == {
        "db_runtime_config": None,
        "csv_session_id": None,
        "horizon": 2,
    }
    assert list(payload["table"]["attrition_forecast"]["yhat"]) == [21, 23]
    assert "attrition_forecast_chart" in payload["plot"]
    assert "44" not in text


def test_forecast_tool_has_structured_args_instead_of_python_code() -> None:
    schema = ForecastTool(forecast_service=_service()).args_schema.model_json_schema()

    assert set(schema["properties"]) == {
        "question",
        "horizon",
        "artifact_name",
        "plot_artifact_name",
    }
    assert "code" not in schema["properties"]
