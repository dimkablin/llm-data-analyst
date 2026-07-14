from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.integrations import (
    AnomalyPlanfactConfig,
    AnomalyPlanfactIntegrationService,
    AnomalyPlanfactQueryResult,
)

_BASE_CONFIG = dict(
    enabled=True,
    base_url="https://anomaly.example",
    analyze_endpoint="/v1/anomaly",
    timeout_sec=60.0,
    backend_api_url="http://backend:8000/v1",
    llm_base_url="http://llm.example",
    llm_api_key="test-key",
    llm_model="test-model",
)


class AnomalyPlanfactIntegrationTests(unittest.TestCase):
    def test_config_from_env_detects_enabled_source(self) -> None:
        config = AnomalyPlanfactConfig.from_env(
            env={
                "ANOMALY_PLANFACT_ENABLED": "true",
                "ANOMALY_PLANFACT_BACKEND_URL": "https://anomaly.example",
                "ANOMALY_PLANFACT_ANALYZE_ENDPOINT": "/v1/anomaly",
                "ANOMALY_PLANFACT_TIMEOUT_SEC": "75",
                "ANOMALY_PLANFACT_SOURCE_LABEL": "Plan-Fact",
            }
        )

        self.assertTrue(config.enabled)
        self.assertEqual(config.base_url, "https://anomaly.example")
        self.assertEqual(config.analyze_endpoint, "/v1/anomaly")
        self.assertEqual(config.timeout_sec, 75.0)

    def test_normalize_row_maps_aliases_correctly(self) -> None:
        normalize = AnomalyPlanfactIntegrationService._normalize_row

        row = normalize({"ts": "2025-01-01", "y": 9.0, "yhat": 10.0})
        self.assertEqual(row["ts"], "2025-01-01")
        self.assertEqual(row["y"], 9.0)
        self.assertEqual(row["yhat"], 10.0)

        row = normalize({"date": "2025-02-01", "fact": 15.0, "plan": 12.5})
        self.assertEqual(row["ts"], "2025-02-01")
        self.assertEqual(row["y"], 15.0)
        self.assertEqual(row["yhat"], 12.5)

        self.assertIsNone(normalize({"y": 9.0, "yhat": 10.0}))
        self.assertIsNone(normalize("not a dict"))

    def test_service_normalizes_backend_response(self) -> None:
        fake_response = {
            "model_name": "planfact-detector",
            "summary": "Detected deviations in two periods.",
            "anomalies": [
                {"date": "2025-02-01", "fact": 130, "plan": 100, "anomaly_score": 0.91},
                {"period": "2025-03-01", "fact": 90, "plan": 105},
            ],
        }

        service = AnomalyPlanfactIntegrationService(AnomalyPlanfactConfig(**_BASE_CONFIG))

        with patch("backend.integrations.anomaly_planfact.post_json", return_value=fake_response):
            result = service.run_analysis("аномалии по выручке", csv_session_id="test-session")

        self.assertEqual(result.model_name, "planfact-detector")
        self.assertEqual(result.summary, "Detected deviations in two periods.")
        self.assertEqual(len(result.anomaly_rows), 2)
        self.assertEqual(result.anomaly_rows[0]["ts"], "2025-02-01")
        self.assertEqual(result.anomaly_rows[1]["y"], 90)

    def test_payload_builder_keeps_source_and_recipe_consistent(self) -> None:
        result = AnomalyPlanfactQueryResult(
            question="аномалии по выручке",
            model_name="planfact-test",
            summary="One material deviation detected.",
            anomaly_rows=[
                {
                    "ts": "2025-02-01",
                    "y": 140,
                    "yhat": 100,
                    "lower": None,
                    "upper": None,
                    "severity": 0.98,
                    "direction": "high",
                }
            ],
            plotly_figure=None,
            warnings=[],
            request_params={"message": "аномалии по выручке", "model": "PlanFact", "fraction": 0.2, "top_k": 50},
        )

        service = AnomalyPlanfactIntegrationService(
            AnomalyPlanfactConfig(**{**_BASE_CONFIG, "source_label": "Anomaly"})
        )
        payload = service.build_artifact_payload(
            result,
            artifact_name="revenue_planfact",
            tool_name="anomaly_planfact_tool",
        )

        self.assertEqual(payload["artifact_name"], "revenue_planfact")
        self.assertEqual(payload["source"]["source_type"], "anomaly_planfact")
        self.assertEqual(payload["recipe"][0]["kind"], "model_inference")
        self.assertEqual(payload["recipe"][0]["model_name"], "planfact-test")
        self.assertIn("anomaly_planfact", payload["meta"])
        self.assertEqual(payload["meta"]["anomaly_planfact"]["row_count"], 1)
