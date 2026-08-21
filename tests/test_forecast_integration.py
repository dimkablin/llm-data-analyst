from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.integrations import (
    ForecastConfig,
    ForecastIntegrationError,
    ForecastIntegrationService,
    ForecastQueryResult,
)

_BASE_CONFIG = dict(
    enabled=True,
    base_url="https://forecast.example",
    predict_endpoint="/v1/forecast",
    timeout_sec=60.0,
    horizon_default=3,
    backend_api_url="http://backend:8000/v1",
    llm_base_url="http://llm.example",
    llm_api_key="test-key",
    llm_model="test-model",
)


class ForecastIntegrationTests(unittest.TestCase):
    def test_config_from_env_detects_enabled_source(self) -> None:
        config = ForecastConfig.from_env(
            env={
                "FORECAST_ENABLED": "true",
                "FORECAST_BACKEND_URL": "https://forecast.example",
                "FORECAST_PREDICT_ENDPOINT": "/v1/forecast",
                "FORECAST_TIMEOUT_SEC": "75",
                "FORECAST_HORIZON_DEFAULT": "6",
            }
        )
        self.assertTrue(config.enabled)
        self.assertEqual(config.base_url, "https://forecast.example")
        self.assertEqual(config.timeout_sec, 75.0)
        self.assertEqual(config.horizon_default, 6)

    def test_normalize_row_maps_aliases_correctly(self) -> None:
        normalize = ForecastIntegrationService._normalize_row
        row = normalize({"ts": "2025-04-01", "yhat": 18, "lower": 16, "upper": 20})
        self.assertEqual(row["ts"], "2025-04-01")
        self.assertEqual(row["yhat"], 18)
        self.assertEqual(row["lower"], 16)
        self.assertEqual(row["upper"], 20)
        row = normalize({"date": "2025-05-01", "yhat": 19.5})
        self.assertEqual(row["ts"], "2025-05-01")
        self.assertEqual(row["yhat"], 19.5)
        self.assertIsNone(normalize({"yhat": 18}))
        self.assertIsNone(normalize("not a dict"))

    def test_service_passes_horizon_and_normalizes_backend_response(self) -> None:
        fake_response = {
            "model_name": "chronos",
            "summary": "Two-step forecast",
            "forecast": [
                {"ts": "2025-04-01", "yhat": 18, "lower": 16, "upper": 20},
                {"date": "2025-05-01", "yhat": 19.5, "lower": 17.0, "upper": 22.0},
            ],
        }
        service = ForecastIntegrationService(ForecastConfig(**_BASE_CONFIG))
        with patch("backend.integrations.forecast.post_json", return_value=fake_response):
            result = service.run_forecast(
                "forecast revenue for two months",
                horizon=2,
                csv_session_id="test-session",
            )
        self.assertEqual(result.horizon, 2)
        self.assertEqual(result.model_name, "chronos")
        self.assertEqual(result.summary, "Two-step forecast")
        self.assertEqual(len(result.forecast_rows), 2)
        self.assertEqual(result.forecast_rows[1]["yhat"], 19.5)

    def test_service_rejects_response_without_forecast_rows(self) -> None:
        service = ForecastIntegrationService(ForecastConfig(**_BASE_CONFIG))
        with patch("backend.integrations.forecast.post_json", return_value={}):
            with self.assertRaisesRegex(ForecastIntegrationError, "returned no forecast rows"):
                service.run_forecast("forecast attrition", csv_session_id="test-session")

    def test_chart_contract_matches_confidence_columns(self) -> None:
        service = ForecastIntegrationService(ForecastConfig(**_BASE_CONFIG))
        with_ci = ForecastQueryResult(
            question="forecast",
            horizon=2,
            model_name="chronos",
            summary=None,
            forecast_rows=[
                {"ts": "2026-08-01", "yhat": 21, "lower": 18, "upper": 24},
                {"ts": "2026-09-01", "yhat": 23, "lower": 19, "upper": 27},
            ],
            plotly_figure=None,
            warnings=[],
            request_params={"message": "forecast", "fh": 2},
        )
        without_ci = ForecastQueryResult(
            question="forecast",
            horizon=2,
            model_name="chronos",
            summary=None,
            forecast_rows=[
                {"ts": "2026-08-01", "yhat": 21, "lower": None, "upper": None},
                {"ts": "2026-09-01", "yhat": 23, "lower": None, "upper": None},
            ],
            plotly_figure=None,
            warnings=[],
            request_params={"message": "forecast", "fh": 2},
        )
        ci_figure = service.build_artifact_payload(with_ci)["plot"]["forecast_chart"]
        no_ci_figure = service.build_artifact_payload(without_ci)["plot"]["forecast_chart"]
        ci_names = [str(trace.get("name", "")) for trace in ci_figure["data"]]
        no_ci_names = [str(trace.get("name", "")) for trace in no_ci_figure["data"]]
        self.assertIn("Forecast", ci_names)
        self.assertIn("Confidence interval", ci_names)
        self.assertIn("Forecast", no_ci_names)
        self.assertNotIn("Confidence interval", no_ci_names)

    def test_payload_builder_keeps_provenance_and_forecast_metadata(self) -> None:
        result = ForecastQueryResult(
            question="forecast revenue",
            horizon=3,
            model_name="test-model",
            summary="Projected growth",
            forecast_rows=[
                {"ts": "2025-04-01", "yhat": 18, "lower": None, "upper": None},
                {"ts": "2025-05-01", "yhat": 20, "lower": None, "upper": None},
            ],
            plotly_figure=None,
            warnings=[],
            request_params={"message": "forecast revenue", "fh": 3},
        )
        service = ForecastIntegrationService(ForecastConfig(**_BASE_CONFIG))
        payload = service.build_artifact_payload(
            result,
            artifact_name="revenue_forecast",
            tool_name="forecast_tool",
        )
        self.assertEqual(payload["artifact_name"], "revenue_forecast")
        self.assertEqual(payload["source"]["source_type"], "forecast")
        self.assertTrue(payload["recipe"])
        self.assertEqual(payload["meta"]["forecast"]["row_count"], 2)


if __name__ == "__main__":
    unittest.main()
