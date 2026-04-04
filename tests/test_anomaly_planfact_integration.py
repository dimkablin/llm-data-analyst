from __future__ import annotations

import unittest

from backend.artifacts import build_artifact_meta
from backend.integrations import (
    AnomalyPlanfactConfig,
    AnomalyPlanfactIntegrationError,
    AnomalyPlanfactIntegrationService,
)


class AnomalyPlanfactIntegrationTests(unittest.TestCase):
    def test_config_from_env_detects_enabled_source(self) -> None:
        config = AnomalyPlanfactConfig.from_env(
            {
                "ANOMALY_PLANFACT_ENABLED": "true",
                "ANOMALY_PLANFACT_BACKEND_URL": "https://anomaly.example",
                "ANOMALY_PLANFACT_ANALYZE_ENDPOINT": "/v1/anomaly",
                "ANOMALY_PLANFACT_TIMEOUT_SEC": "75",
                "ANOMALY_PLANFACT_SOURCE_LABEL": "Plan-Fact",
            }
        )

        self.assertTrue(config.enabled)
        self.assertTrue(config.available)
        self.assertEqual(config.base_url, "https://anomaly.example")
        self.assertEqual(config.analyze_endpoint, "/v1/anomaly")
        self.assertEqual(config.timeout_sec, 75.0)
        self.assertEqual(config.source_label, "Plan-Fact")

    def test_normalize_series_input_enforces_first_contract(self) -> None:
        service = AnomalyPlanfactIntegrationService(
            AnomalyPlanfactConfig(
                enabled=True,
                base_url="https://anomaly.example",
                analyze_endpoint="/v1/anomaly",
                timeout_sec=60.0,
            ),
            transport=lambda url, payload, timeout: {},
        )

        rows = service.normalize_series_input(
            [
                {"month": "2025-01-01", "plan": 10, "fact": 9},
                {"month": "2025-02-01", "plan": "12.5", "fact": 15},
            ],
            time_col="month",
            plan_col="plan",
            fact_col="fact",
        )

        self.assertEqual(
            rows,
            [
                {"ts": "2025-01-01", "plan": 10.0, "fact": 9.0},
                {"ts": "2025-02-01", "plan": 12.5, "fact": 15.0},
            ],
        )

        with self.assertRaises(AnomalyPlanfactIntegrationError):
            service.normalize_series_input(
                [{"month": "2025-01-01", "plan": 10, "fact": 9}],
                time_col="month",
                plan_col="plan",
                fact_col="fact",
            )

        with self.assertRaises(AnomalyPlanfactIntegrationError):
            service.normalize_series_input(
                [
                    {"month": "2025-01-01", "plan": 10, "fact": 9},
                    {"month": "2025-02-01", "plan": "oops", "fact": 15},
                ],
                time_col="month",
                plan_col="plan",
                fact_col="fact",
            )

    def test_service_normalizes_backend_response(self) -> None:
        captured: dict[str, object] = {}

        def fake_transport(
            url: str,
            payload: dict[str, object],
            timeout_sec: float,
        ) -> dict[str, object]:
            captured["url"] = url
            captured["payload"] = dict(payload)
            captured["timeout_sec"] = timeout_sec
            return {
                "model_name": "planfact-detector",
                "summary": "Detected deviations in two periods.",
                "anomalies": [
                    {
                        "date": "2025-02-01",
                        "expected": 100,
                        "actual": 130,
                        "absolute_deviation": 30,
                        "relative_deviation": 30.0,
                        "score": 0.91,
                        "anomaly": True,
                        "reason": "Revenue materially exceeded plan.",
                    },
                    {
                        "period": "2025-03-01",
                        "plan": 105,
                        "fact": 90,
                        "abs_diff": -15,
                        "pct_diff": -14.29,
                        "is_anomaly": False,
                    },
                ],
            }

        service = AnomalyPlanfactIntegrationService(
            AnomalyPlanfactConfig(
                enabled=True,
                base_url="https://anomaly.example",
                analyze_endpoint="/v1/anomaly",
                timeout_sec=45.0,
            ),
            transport=fake_transport,
        )

        result = service.run_analysis(
            [
                {"month": "2025-01-01", "plan_revenue": 95, "fact_revenue": 97},
                {"month": "2025-02-01", "plan_revenue": 100, "fact_revenue": 130},
                {"month": "2025-03-01", "plan_revenue": 105, "fact_revenue": 90},
            ],
            time_col="month",
            plan_col="plan_revenue",
            fact_col="fact_revenue",
            target_name="revenue",
        )

        self.assertEqual(captured["url"], "https://anomaly.example/v1/anomaly")
        self.assertEqual(captured["timeout_sec"], 45.0)
        self.assertEqual(result.input_point_count, 3)
        self.assertEqual(result.model_name, "planfact-detector")
        self.assertEqual(result.summary, "Detected deviations in two periods.")
        self.assertEqual(len(result.analysis_rows), 2)
        self.assertEqual(result.analysis_rows[0]["ts"], "2025-02-01")
        self.assertEqual(result.analysis_rows[0]["is_anomaly"], True)
        self.assertEqual(result.analysis_rows[1]["fact"], 90.0)

    def test_payload_builder_keeps_source_and_recipe_consistent(self) -> None:
        service = AnomalyPlanfactIntegrationService(
            AnomalyPlanfactConfig(
                enabled=True,
                base_url="https://anomaly.example",
                analyze_endpoint="/v1/anomaly",
                timeout_sec=45.0,
                source_label="Anomaly",
            ),
            transport=lambda url, payload, timeout: {
                "model": "planfact-test",
                "analysis_summary": "One material deviation detected.",
                "rows": [
                    {
                        "ts": "2025-02-01",
                        "plan": 100,
                        "fact": 140,
                        "delta_abs": 40,
                        "delta_pct": 40.0,
                        "anomaly_score": 0.98,
                        "is_anomaly": True,
                    }
                ],
            },
        )

        result = service.run_analysis(
            [
                {"month": "2025-01-01", "plan": 95, "fact": 97},
                {"month": "2025-02-01", "plan": 100, "fact": 140},
            ],
            time_col="month",
            plan_col="plan",
            fact_col="fact",
            target_name="revenue",
        )
        payload = service.build_artifact_payload(
            result,
            artifact_name="revenue_planfact",
            tool_name="anomaly_planfact_tool",
        )

        self.assertEqual(payload["artifact_name"], "revenue_planfact")
        self.assertEqual(payload["source"]["source_type"], "anomaly_planfact")
        self.assertEqual(payload["recipe"][0]["kind"], "model_inference")
        self.assertEqual(payload["recipe"][0]["title"], "Anomaly / plan-fact analysis")
        self.assertEqual(payload["recipe"][0]["model_name"], "planfact-test")
        self.assertEqual(payload["meta"]["anomaly_planfact"]["anomaly_count"], 1)

        meta = build_artifact_meta(
            tool_name="anomaly_planfact_tool",
            source_context={
                "source_type": "csv",
                "source_ref_id": "legacy.csv",
                "source_label": "Legacy CSV",
            },
            artifact_hints=payload,
        )

        self.assertEqual(meta["source"]["source_type"], "anomaly_planfact")
        self.assertEqual(meta["provenance"]["source"], meta["source"])
        self.assertEqual(meta["provenance"]["recipe"], meta["recipe"])
        self.assertEqual(meta["recipe"][0]["kind"], "model_inference")


if __name__ == "__main__":
    unittest.main()
