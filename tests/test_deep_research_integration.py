from __future__ import annotations

import unittest

from backend.artifact_meta import build_artifact_meta
from backend.deep_research_integration import (
    DeepResearchConfig,
    DeepResearchIntegrationService,
)


class DeepResearchIntegrationTests(unittest.TestCase):
    def test_config_from_env_detects_enabled_source(self) -> None:
        config = DeepResearchConfig.from_env(
            {
                "DEEP_RESEARCH_ENABLED": "true",
                "DEEP_RESEARCH_BACKEND_URL": "https://research.example",
                "DEEP_RESEARCH_CREATE_ENDPOINT": "/v1/research/",
                "DEEP_RESEARCH_EXECUTE_ENDPOINT": "/v1/research/{id}/run",
                "DEEP_RESEARCH_DETAIL_ENDPOINT": "/v1/research/{id}",
                "DEEP_RESEARCH_CREATE_TIMEOUT_SEC": "25",
                "DEEP_RESEARCH_EXECUTE_TIMEOUT_SEC": "180",
                "DEEP_RESEARCH_POLL_TIMEOUT_SEC": "90",
                "DEEP_RESEARCH_POLL_INTERVAL_SEC": "1.5",
                "DEEP_RESEARCH_MAX_ITERATIONS_DEFAULT": "4",
                "DEEP_RESEARCH_LANGUAGE_DEFAULT": "en",
                "DEEP_RESEARCH_SOURCE_LABEL": "Research",
            }
        )

        self.assertTrue(config.enabled)
        self.assertTrue(config.available)
        self.assertEqual(config.base_url, "https://research.example")
        self.assertEqual(config.create_endpoint, "/v1/research/")
        self.assertEqual(config.execute_endpoint, "/v1/research/{id}/run")
        self.assertEqual(config.detail_endpoint, "/v1/research/{id}")
        self.assertEqual(config.create_timeout_sec, 25.0)
        self.assertEqual(config.execute_timeout_sec, 180.0)
        self.assertEqual(config.poll_timeout_sec, 90.0)
        self.assertEqual(config.poll_interval_sec, 1.5)
        self.assertEqual(config.max_iterations_default, 4)
        self.assertEqual(config.language_default, "en")
        self.assertEqual(config.source_label, "Research")

    def test_service_runs_workflow_and_normalizes_report(self) -> None:
        calls: list[tuple[str, str, dict[str, object] | None, float]] = []

        def fake_transport(
            method: str,
            url: str,
            payload: dict[str, object] | None,
            timeout_sec: float,
        ) -> dict[str, object]:
            calls.append((method, url, payload, timeout_sec))
            if url.endswith("/api/v1/research/"):
                return {"id": "r-123", "status": "created"}
            if url.endswith("/api/v1/research/r-123/execute"):
                return {"status": "running"}
            if url.endswith("/api/v1/research/r-123"):
                return {
                    "status": "completed",
                    "report": {
                        "summary": "Detailed synthesis",
                        "content": "Long-form report body",
                        "sections": [
                            {"title": "Overview", "content": "Main overview"},
                            {"title": "Risks", "content": "Key risks"},
                        ],
                        "sources": [
                            {
                                "title": "Paper A",
                                "url": "https://example.com/paper-a",
                                "snippet": "Alpha",
                                "source": "arxiv",
                            }
                        ],
                    },
                }
            raise AssertionError(f"Unexpected URL: {url}")

        service = DeepResearchIntegrationService(
            DeepResearchConfig(
                enabled=True,
                base_url="https://research.example",
                create_endpoint="/api/v1/research/",
                execute_endpoint="/api/v1/research/{id}/execute",
                detail_endpoint="/api/v1/research/{id}",
                create_timeout_sec=20.0,
                execute_timeout_sec=60.0,
                poll_timeout_sec=5.0,
                poll_interval_sec=0.1,
                max_iterations_default=3,
                language_default="ru",
            ),
            transport=fake_transport,
        )

        result = service.run_research(
            "detailed research on observability",
            max_iterations=2,
            language="en",
        )

        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][0], "POST")
        self.assertEqual(calls[1][0], "POST")
        self.assertEqual(calls[2][0], "GET")
        self.assertEqual(
            calls[0][2],
            {
                "query": "detailed research on observability",
                "max_iterations": 2,
                "language": "en",
            },
        )
        self.assertEqual(result.research_id, "r-123")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.summary, "Detailed synthesis")
        self.assertEqual(result.report_text, "Long-form report body")
        self.assertEqual(result.sources, ["https://example.com/paper-a"])
        self.assertEqual(result.rows[0]["kind"], "section")
        self.assertEqual(result.rows[0]["title"], "Overview")

    def test_payload_builder_keeps_source_and_recipe_consistent(self) -> None:
        def fake_transport(
            method: str,
            url: str,
            payload: dict[str, object] | None,
            timeout_sec: float,
        ) -> dict[str, object]:
            _ = (method, url, payload, timeout_sec)
            if url.endswith("/research/"):
                return {"research_id": "dr-7", "status": "created"}
            if url.endswith("/research/dr-7/execute"):
                return {
                    "status": "completed",
                    "final_report": {
                        "summary": "Structured deep research",
                        "findings": [
                            "Finding one",
                            {"title": "Finding two", "content": "More detail"},
                        ],
                        "sources": ["https://example.com/a", "https://example.com/b"],
                    },
                }
            raise AssertionError(f"Unexpected URL: {url}")

        service = DeepResearchIntegrationService(
            DeepResearchConfig(
                enabled=True,
                base_url="https://research.example",
                create_endpoint="/research/",
                execute_endpoint="/research/{id}/execute",
                detail_endpoint="/research/{id}",
                create_timeout_sec=20.0,
                execute_timeout_sec=60.0,
                poll_timeout_sec=1.0,
                poll_interval_sec=0.1,
                max_iterations_default=3,
                language_default="ru",
                source_label="Deep Research",
            ),
            transport=fake_transport,
        )

        result = service.run_research("prepare a broad market research")
        payload = service.build_artifact_payload(
            result,
            artifact_name="market_research",
            tool_name="deep_research_tool",
        )

        self.assertEqual(payload["artifact_name"], "market_research")
        self.assertEqual(payload["source"]["source_type"], "deep_research")
        self.assertEqual(payload["recipe"][0]["kind"], "source_query")
        self.assertEqual(payload["recipe"][0]["title"], "Deep Research Query")
        self.assertEqual(
            payload["meta"]["deep_research"]["research_id"],
            "dr-7",
        )
        self.assertEqual(payload["meta"]["deep_research"]["result_count"], 2)
        self.assertEqual(payload["rows"][0]["kind"], "finding")

        meta = build_artifact_meta(
            tool_name="deep_research_tool",
            source_context={
                "source_type": "csv",
                "source_ref_id": "legacy.csv",
                "source_label": "Legacy CSV",
            },
            artifact_hints=payload,
        )

        self.assertEqual(meta["source"]["source_type"], "deep_research")
        self.assertEqual(meta["provenance"]["source"], meta["source"])
        self.assertEqual(meta["provenance"]["recipe"], meta["recipe"])
        self.assertEqual(meta["recipe"][0]["kind"], "source_query")


if __name__ == "__main__":
    unittest.main()
