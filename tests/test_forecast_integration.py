from __future__ import annotations

import unittest

from backend.artifacts import build_artifact_meta
from backend.integrations import (
    ForecastConfig,
    ForecastIntegrationError,
    ForecastIntegrationService,
)


class ForecastIntegrationTests(unittest.TestCase):
    def test_config_from_env_detects_enabled_source(self) -> None:
        config = ForecastConfig.from_env(
            {
                "FORECAST_ENABLED": "true",
                "FORECAST_BACKEND_URL": "https://forecast.example",
                "FORECAST_PREDICT_ENDPOINT": "/v1/forecast",
                "FORECAST_TIMEOUT_SEC": "75",
                "FORECAST_HORIZON_DEFAULT": "6",
                "FORECAST_SOURCE_LABEL": "Predict",
            }
        )

        self.assertTrue(config.enabled)
        self.assertTrue(config.available)
        self.assertEqual(config.base_url, "https://forecast.example")
        self.assertEqual(config.predict_endpoint, "/v1/forecast")
        self.assertEqual(config.timeout_sec, 75.0)
        self.assertEqual(config.horizon_default, 6)
        self.assertEqual(config.source_label, "Predict")

    def test_normalize_series_input_enforces_first_contract(self) -> None:
        service = ForecastIntegrationService(
            ForecastConfig(
                enabled=True,
                base_url="https://forecast.example",
                predict_endpoint="/v1/forecast",
                timeout_sec=60.0,
                horizon_default=3,
            ),
            transport=lambda url, payload, timeout: {},
        )

        rows = service.normalize_series_input(
            [
                {"month": "2025-01-01", "revenue": 10},
                {"month": "2025-02-01", "revenue": "12.5"},
                {"month": "2025-03-01", "revenue": 15},
            ],
            time_col="month",
            value_col="revenue",
        )

        self.assertEqual(
            rows,
            [
                {"ts": "2025-01-01", "y": 10.0},
                {"ts": "2025-02-01", "y": 12.5},
                {"ts": "2025-03-01", "y": 15.0},
            ],
        )

        with self.assertRaises(ForecastIntegrationError):
            service.normalize_series_input(
                [{"month": "2025-01-01", "revenue": 10}],
                time_col="month",
                value_col="revenue",
            )

        with self.assertRaises(ForecastIntegrationError):
            service.normalize_series_input(
                [
                    {"month": "2025-01-01", "revenue": 10},
                    {"month": "2025-02-01", "revenue": "oops"},
                    {"month": "2025-03-01", "revenue": 15},
                ],
                time_col="month",
                value_col="revenue",
            )

    def test_service_normalizes_backend_response(self) -> None:
        captured: dict[str, object] = {}

        def fake_transport(url: str, payload: dict[str, object], timeout_sec: float) -> dict[str, object]:
            captured["url"] = url
            captured["payload"] = dict(payload)
            captured["timeout_sec"] = timeout_sec
            return {
                "model_name": "chronos",
                "summary": "Three-step forecast",
                "forecast": [
                    {"ts": "2025-04-01", "yhat": 18, "lower": 16, "upper": 20},
                    {
                        "date": "2025-05-01",
                        "prediction": 19.5,
                        "lower_bound": 17.0,
                        "upper_bound": 22.0,
                    },
                ],
            }

        service = ForecastIntegrationService(
            ForecastConfig(
                enabled=True,
                base_url="https://forecast.example",
                predict_endpoint="/v1/forecast",
                timeout_sec=45.0,
                horizon_default=3,
            ),
            transport=fake_transport,
        )

        result = service.run_forecast(
            [
                {"month": "2025-01-01", "revenue": 10},
                {"month": "2025-02-01", "revenue": 12},
                {"month": "2025-03-01", "revenue": 15},
            ],
            time_col="month",
            value_col="revenue",
            horizon=2,
            frequency="month",
            target_name="revenue",
        )

        self.assertEqual(captured["url"], "https://forecast.example/v1/forecast")
        self.assertEqual(captured["timeout_sec"], 45.0)
        self.assertEqual(result.horizon, 2)
        self.assertEqual(result.input_point_count, 3)
        self.assertEqual(result.model_name, "chronos")
        self.assertEqual(result.summary, "Three-step forecast")
        self.assertEqual(len(result.forecast_rows), 2)
        self.assertEqual(result.forecast_rows[0]["ts"], "2025-04-01")
        self.assertEqual(result.forecast_rows[1]["yhat"], 19.5)

    def test_payload_builder_keeps_source_and_recipe_consistent(self) -> None:
        service = ForecastIntegrationService(
            ForecastConfig(
                enabled=True,
                base_url="https://forecast.example",
                predict_endpoint="/v1/forecast",
                timeout_sec=45.0,
                horizon_default=3,
                source_label="Forecast",
            ),
            transport=lambda url, payload, timeout: {
                "model": "test-model",
                "forecast_summary": "Projected growth",
                "predictions": [
                    {"period": "2025-04-01", "value": 18},
                    {"period": "2025-05-01", "value": 20},
                ],
            },
        )

        result = service.run_forecast(
            [
                {"month": "2025-01-01", "revenue": 10},
                {"month": "2025-02-01", "revenue": 12},
                {"month": "2025-03-01", "revenue": 15},
            ],
            time_col="month",
            value_col="revenue",
        )
        payload = service.build_artifact_payload(
            result,
            artifact_name="revenue_forecast",
            tool_name="forecast_tool",
        )

        self.assertEqual(payload["artifact_name"], "revenue_forecast")
        self.assertEqual(payload["source"]["source_type"], "forecast")
        self.assertEqual(payload["recipe"][0]["kind"], "model_inference")
        self.assertEqual(payload["recipe"][0]["title"], "Forecast run")
        self.assertEqual(payload["recipe"][0]["model_name"], "test-model")
        self.assertEqual(payload["meta"]["forecast"]["horizon"], 3)
        self.assertEqual(payload["meta"]["forecast"]["input_point_count"], 3)

        meta = build_artifact_meta(
            tool_name="forecast_tool",
            source_context={
                "source_type": "csv",
                "source_ref_id": "legacy.csv",
                "source_label": "Legacy CSV",
            },
            artifact_hints=payload,
        )

        self.assertEqual(meta["source"]["source_type"], "forecast")
        self.assertEqual(meta["provenance"]["source"], meta["source"])
        self.assertEqual(meta["provenance"]["recipe"], meta["recipe"])
        self.assertEqual(meta["recipe"][0]["kind"], "model_inference")


if __name__ == "__main__":
    unittest.main()
