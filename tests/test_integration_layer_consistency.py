from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.integrations import (
    AnomalyPlanfactConfig,
    AnomalyPlanfactIntegrationService,
    AnomalyPlanfactQueryResult,
    ForecastConfig,
    ForecastIntegrationService,
    ForecastQueryResult,
    RAGConfig,
    RAGIntegrationError,
    RAGService,
    SearchIntegrationConfig,
    SearchIntegrationError,
    SearchIntegrationService,
)
from backend.integrations.predict_common import PredictIntegrationError

_FORECAST_CONFIG = dict(
    enabled=True,
    base_url="https://forecast.example",
    predict_endpoint="/forecast",
    timeout_sec=45.0,
    horizon_default=3,
    backend_api_url="http://backend:8000/v1",
    llm_base_url="http://llm.example",
    llm_api_key="test-key",
    llm_model="test-model",
)

_ANOMALY_CONFIG = dict(
    enabled=True,
    base_url="https://anomaly.example",
    analyze_endpoint="/anomaly",
    timeout_sec=50.0,
    backend_api_url="http://backend:8000/v1",
    llm_base_url="http://llm.example",
    llm_api_key="test-key",
    llm_model="test-model",
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
        forecast = ForecastIntegrationService(ForecastConfig(**_FORECAST_CONFIG))
        anomaly = AnomalyPlanfactIntegrationService(AnomalyPlanfactConfig(**_ANOMALY_CONFIG))

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

        forecast_result = ForecastQueryResult(
            question="прогноз выручки",
            horizon=3,
            model_name="chronos",
            summary=None,
            forecast_rows=[{"ts": "2025-04-01", "yhat": 18, "lower": None, "upper": None}],
            plotly_figure=None,
            warnings=[],
            request_params={"message": "прогноз выручки", "fh": 3},
        )
        anomaly_result = AnomalyPlanfactQueryResult(
            question="аномалии",
            model_name="PlanFact",
            summary=None,
            anomaly_rows=[{"ts": "2025-01-01", "y": 12, "yhat": 10, "lower": None, "upper": None, "severity": None, "direction": None}],
            plotly_figure=None,
            warnings=[],
            request_params={"message": "аномалии", "model": "PlanFact", "fraction": 0.2, "top_k": 50},
        )

        forecast_svc = ForecastIntegrationService(ForecastConfig(**_FORECAST_CONFIG))
        anomaly_svc = AnomalyPlanfactIntegrationService(AnomalyPlanfactConfig(**_ANOMALY_CONFIG))

        payloads = [
            search.build_artifact_payload(search.search("agents")),
            forecast_svc.build_artifact_payload(forecast_result),
            anomaly_svc.build_artifact_payload(anomaly_result),
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

        forecast = ForecastIntegrationService(ForecastConfig(**_FORECAST_CONFIG))
        with patch(
            "backend.integrations.forecast.post_json",
            side_effect=PredictIntegrationError("request timed out"),
        ):
            with self.assertRaises(PredictIntegrationError) as forecast_exc:
                forecast.run_forecast("прогноз выручки", csv_session_id="test-session")
        self.assertIn("request timed out", str(forecast_exc.exception).lower())

        anomaly = AnomalyPlanfactIntegrationService(AnomalyPlanfactConfig(**_ANOMALY_CONFIG))
        with patch(
            "backend.integrations.anomaly_planfact.post_json",
            side_effect=PredictIntegrationError("request timed out"),
        ):
            with self.assertRaises(PredictIntegrationError) as anomaly_exc:
                anomaly.run_analysis("аномалии по выручке", csv_session_id="test-session")
        self.assertIn("request timed out", str(anomaly_exc.exception).lower())
