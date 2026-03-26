from __future__ import annotations

import unittest

from backend.integrations import (
    AnomalyPlanfactConfig,
    AnomalyPlanfactIntegrationError,
    AnomalyPlanfactIntegrationService,
    ForecastConfig,
    ForecastIntegrationError,
    ForecastIntegrationService,
    RAGConfig,
    RAGIntegrationError,
    RAGService,
    SearchIntegrationConfig,
    SearchIntegrationError,
    SearchIntegrationService,
)


class IntegrationLayerConsistencyTests(unittest.TestCase):
    def test_source_descriptors_follow_minimal_catalog_contract(self) -> None:
        search = SearchIntegrationService(
            SearchIntegrationConfig(
                enabled=True,
                base_url="https://search.example",
                search_endpoint="/api/v1/search/",
                fetch_endpoint="/api/v1/fetch/",
                timeout_sec=10.0,
                max_results_default=5,
                fetch_top_n_default=3,
            )
        )
        rag = RAGService(
            RAGConfig(
                enabled=True,
                base_url="https://rag.example",
                query_endpoint="/query",
                stream_endpoint="/query/stream",
                timeout_sec=25.0,
                verify_ssl=False,
                query_mode_default="hybrid",
                top_k_default=5,
            )
        )
        forecast = ForecastIntegrationService(
            ForecastConfig(
                enabled=True,
                base_url="https://forecast.example",
                predict_endpoint="/forecast",
                timeout_sec=45.0,
                horizon_default=3,
            )
        )
        anomaly = AnomalyPlanfactIntegrationService(
            AnomalyPlanfactConfig(
                enabled=True,
                base_url="https://anomaly.example",
                analyze_endpoint="/anomaly",
                timeout_sec=50.0,
            )
        )

        descriptors = [
            search.source_descriptor(),
            rag.source_descriptor(),
            forecast.source_descriptor(),
            anomaly.source_descriptor(),
        ]

        for descriptor in descriptors:
            self.assertIn(descriptor["status"], {"available", "disabled", "misconfigured"})
            self.assertIn("requires_session_data", descriptor)
            self.assertIn("timeout_hint_sec", descriptor)
            self.assertIsInstance(descriptor["capabilities"], list)
            self.assertIn("display_name_ru", descriptor)
            self.assertIn("description_ru", descriptor)

        self.assertFalse(search.source_descriptor()["requires_session_data"])
        self.assertFalse(rag.source_descriptor()["requires_session_data"])
        self.assertTrue(forecast.source_descriptor()["requires_session_data"])
        self.assertTrue(anomaly.source_descriptor()["requires_session_data"])
        self.assertEqual(search.source_descriptor()["status"], "available")
        self.assertTrue(search.source_descriptor()["display_name_ru"])
        self.assertIn("search", search.source_descriptor()["capabilities"])

    def test_operational_meta_contains_common_fields(self) -> None:
        search = SearchIntegrationService(
            SearchIntegrationConfig(
                enabled=True,
                base_url="https://search.example",
                search_endpoint="/api/v1/search/",
                fetch_endpoint="/api/v1/fetch/",
                timeout_sec=10.0,
                max_results_default=5,
                fetch_top_n_default=3,
            ),
            transport=lambda url, payload, timeout: {
                "results": [{"title": "Doc", "url": "https://example.com/doc"}]
            },
        )
        rag = RAGService(
            RAGConfig(
                enabled=True,
                base_url="https://rag.example",
                query_endpoint="/query",
                stream_endpoint="/query/stream",
                timeout_sec=25.0,
                verify_ssl=False,
                query_mode_default="hybrid",
                top_k_default=5,
            ),
            transport=lambda url, payload, timeout, verify_ssl: {
                "response": "Use access token",
                "references": ["https://docs.example/auth"],
            },
        )
        forecast = ForecastIntegrationService(
            ForecastConfig(
                enabled=True,
                base_url="https://forecast.example",
                predict_endpoint="/forecast",
                timeout_sec=45.0,
                horizon_default=3,
            ),
            transport=lambda url, payload, timeout: {
                "forecast": [{"ts": "2025-04-01", "yhat": 18}]
            },
        )
        anomaly = AnomalyPlanfactIntegrationService(
            AnomalyPlanfactConfig(
                enabled=True,
                base_url="https://anomaly.example",
                analyze_endpoint="/anomaly",
                timeout_sec=50.0,
            ),
            transport=lambda url, payload, timeout: {
                "rows": [{"ts": "2025-01-01", "plan": 10, "fact": 12, "is_anomaly": True}]
            },
        )

        payloads = [
            search.build_artifact_payload(search.search("agents")),
            forecast.build_artifact_payload(
                forecast.run_forecast(
                    [
                        {"month": "2025-01-01", "revenue": 10},
                        {"month": "2025-02-01", "revenue": 12},
                        {"month": "2025-03-01", "revenue": 15},
                    ],
                    time_col="month",
                    value_col="revenue",
                )
            ),
            anomaly.build_artifact_payload(
                anomaly.run_analysis(
                    [
                        {"month": "2025-01-01", "plan": 10, "fact": 9},
                        {"month": "2025-02-01", "plan": 12, "fact": 15},
                    ],
                    time_col="month",
                    plan_col="plan",
                    fact_col="fact",
                )
            ),
        ]

        rag_result = rag.search(query="auth docs")
        self.assertEqual(rag_result.answer, "Use access token")
        self.assertEqual(rag_result.references, ["https://docs.example/auth"])

        for payload in payloads:
            meta_section = next(iter(payload["meta"].values()))
            self.assertIn("status", meta_section)
            self.assertIn("warnings", meta_section)
            self.assertIn("request_params", meta_section)
            self.assertIn("timeout_sec", meta_section)

    def test_timeout_errors_are_normalized_across_integrations(self) -> None:
        search = SearchIntegrationService(
            SearchIntegrationConfig(
                enabled=True,
                base_url="https://search.example",
                search_endpoint="/api/v1/search/",
                fetch_endpoint="/api/v1/fetch/",
                timeout_sec=10.0,
                max_results_default=5,
                fetch_top_n_default=3,
            ),
            transport=lambda url, payload, timeout: (_ for _ in ()).throw(TimeoutError()),
        )
        with self.assertRaises(SearchIntegrationError) as search_exc:
            search.search("agents")
        self.assertIn("request timed out", str(search_exc.exception).lower())

        rag = RAGService(
            RAGConfig(
                enabled=True,
                base_url="https://rag.example",
                query_endpoint="/query",
                stream_endpoint="/query/stream",
                timeout_sec=20.0,
                verify_ssl=False,
                query_mode_default="hybrid",
                top_k_default=5,
            ),
            transport=lambda url, payload, timeout, verify_ssl: (_ for _ in ()).throw(TimeoutError()),
        )
        with self.assertRaises(RAGIntegrationError) as rag_exc:
            rag.search(query="agents")
        self.assertIn("request timed out", str(rag_exc.exception).lower())

        forecast = ForecastIntegrationService(
            ForecastConfig(
                enabled=True,
                base_url="https://forecast.example",
                predict_endpoint="/forecast",
                timeout_sec=45.0,
                horizon_default=3,
            ),
            transport=lambda url, payload, timeout: (_ for _ in ()).throw(TimeoutError()),
        )
        with self.assertRaises(ForecastIntegrationError) as forecast_exc:
            forecast.run_forecast(
                [
                    {"month": "2025-01-01", "revenue": 10},
                    {"month": "2025-02-01", "revenue": 12},
                    {"month": "2025-03-01", "revenue": 15},
                ],
                time_col="month",
                value_col="revenue",
            )
        self.assertIn("request timed out", str(forecast_exc.exception).lower())

        anomaly = AnomalyPlanfactIntegrationService(
            AnomalyPlanfactConfig(
                enabled=True,
                base_url="https://anomaly.example",
                analyze_endpoint="/anomaly",
                timeout_sec=50.0,
            ),
            transport=lambda url, payload, timeout: (_ for _ in ()).throw(TimeoutError()),
        )
        with self.assertRaises(AnomalyPlanfactIntegrationError) as anomaly_exc:
            anomaly.run_analysis(
                [
                    {"month": "2025-01-01", "plan": 10, "fact": 9},
                    {"month": "2025-02-01", "plan": 12, "fact": 15},
                ],
                time_col="month",
                plan_col="plan",
                fact_col="fact",
            )
        self.assertIn("request timed out", str(anomaly_exc.exception).lower())


if __name__ == "__main__":
    unittest.main()
