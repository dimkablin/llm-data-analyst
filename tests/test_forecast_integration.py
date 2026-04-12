from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.integrations import (
    ForecastConfig,
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
                "FORECAST_SOURCE_LABEL": "Predict",
            }
        )

        self.assertTrue(config.enabled)
        self.assertEqual(config.base_url, "https://forecast.example")
        self.assertEqual(config.predict_endpoint, "/v1/forecast")
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

    def test_service_normalizes_backend_response(self) -> None:
        fake_response = {
            "model_name": "chronos",
            "summary": "Three-step forecast",
            "forecast": [
                {"ts": "2025-04-01", "yhat": 18, "lower": 16, "upper": 20},
                {"date": "2025-05-01", "yhat": 19.5, "lower": 17.0, "upper": 22.0},
            ],
        }

        service = ForecastIntegrationService(ForecastConfig(**_BASE_CONFIG))

        with patch("backend.integrations.forecast.post_json", return_value=fake_response):
            result = service.run_forecast("прогноз выручки на 2 месяца", horizon=2, csv_session_id="test-session")

        self.assertEqual(result.horizon, 2)
        self.assertEqual(result.model_name, "chronos")
        self.assertEqual(result.summary, "Three-step forecast")
        self.assertEqual(len(result.forecast_rows), 2)
        self.assertEqual(result.forecast_rows[0]["ts"], "2025-04-01")
        self.assertEqual(result.forecast_rows[1]["yhat"], 19.5)

    def test_payload_builder_keeps_source_and_recipe_consistent(self) -> None:
        result = ForecastQueryResult(
            question="прогноз выручки",
            horizon=3,
            model_name="test-model",
            summary="Projected growth",
            forecast_rows=[
                {"ts": "2025-04-01", "yhat": 18, "lower": None, "upper": None},
                {"ts": "2025-05-01", "yhat": 20, "lower": None, "upper": None},
            ],
            plotly_figure=None,
            warnings=[],
            request_params={"message": "прогноз выручки", "fh": 3},
        )

        service = ForecastIntegrationService(
            ForecastConfig(**{**_BASE_CONFIG, "source_label": "Forecast"})
        )
        payload = service.build_artifact_payload(
            result,
            artifact_name="revenue_forecast",
            tool_name="forecast_tool",
        )

        self.assertEqual(payload["artifact_name"], "revenue_forecast")
        self.assertEqual(payload["source"]["source_type"], "forecast")
        self.assertEqual(payload["recipe"][0]["kind"], "model_inference")
        self.assertEqual(payload["recipe"][0]["model_name"], "test-model")
        self.assertIn("forecast", payload["meta"])
        self.assertEqual(payload["meta"]["forecast"]["row_count"], 2)


if __name__ == "__main__":
    unittest.main()
