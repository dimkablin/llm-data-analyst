from __future__ import annotations

import unittest

import pandas as pd

from backend.agent.tools.search_tool import SearchTool
from backend.artifacts import build_artifact_meta
from backend.integrations import SearchIntegrationConfig, SearchIntegrationService


class SearchIntegrationTests(unittest.TestCase):
    def test_config_from_env_detects_enabled_search_source(self) -> None:
        config = SearchIntegrationConfig.from_env(
            {
                "SEARCH_ENABLED": "true",
                "SEARCH_BACKEND_URL": "https://search.example",
                "SEARCH_ENDPOINT": "/v1/find",
                "SEARCH_TIMEOUT_SEC": "12",
                "SEARCH_MAX_RESULTS_DEFAULT": "7",
                "SEARCH_FETCH_TOP_N_DEFAULT": "4",
                "SEARCH_SOURCE_LABEL": "Web Search",
            }
        )

        self.assertTrue(config.enabled)
        self.assertTrue(config.available)
        self.assertEqual(config.base_url, "https://search.example")
        self.assertEqual(config.search_endpoint, "/v1/find")
        self.assertEqual(config.timeout_sec, 12.0)
        self.assertEqual(config.max_results_default, 7)
        self.assertEqual(config.fetch_top_n_default, 4)
        self.assertEqual(config.source_label, "Web Search")

    def test_service_normalizes_backend_response(self) -> None:
        captured: dict[str, object] = {}

        def fake_transport(url: str, payload: dict[str, object], timeout_sec: float) -> dict[str, object]:
            captured["url"] = url
            captured["payload"] = dict(payload)
            captured["timeout_sec"] = timeout_sec
            return {
                "answer": "Concise synthesis",
                "sources": ["https://docs.example/root"],
                "results": [
                    {
                        "title": "Doc 1",
                        "url": "https://docs.example/a",
                        "snippet": "Alpha",
                        "source": "engine-a",
                        "published_at": "2026-03-19",
                    },
                    {
                        "name": "Doc 2",
                        "link": "https://docs.example/b",
                        "description": "Beta",
                    },
                ],
            }

        service = SearchIntegrationService(
            SearchIntegrationConfig(
                enabled=True,
                base_url="https://search.example",
                search_endpoint="/api/v1/search/",
                fetch_endpoint="/api/v1/fetch/",
                timeout_sec=9.0,
                max_results_default=5,
                fetch_top_n_default=3,
            ),
            transport=fake_transport,
        )

        result = service.search(
            "agent observability",
            max_results=2,
            fetch_top_n=1,
            language="en",
        )

        self.assertEqual(captured["url"], "https://search.example/api/v1/search/")
        self.assertEqual(
            captured["payload"],
            {
                "query": "agent observability",
                "max_search_results": 2,
                "fetch_top_n": 1,
                "language": "en",
            },
        )
        self.assertEqual(captured["timeout_sec"], 9.0)
        self.assertEqual(result.answer, "Concise synthesis")
        self.assertEqual(result.result_count, 2)
        self.assertEqual(result.sources[0], "https://docs.example/root")
        self.assertEqual(result.results[0].title, "Doc 1")
        self.assertEqual(result.results[1].title, "Doc 2")
        self.assertEqual(result.to_rows()[0]["rank"], 1)

    def test_search_payload_builder_keeps_source_and_recipe_consistent(self) -> None:
        def fake_transport(url: str, payload: dict[str, object], timeout_sec: float) -> dict[str, object]:
            _ = (url, payload, timeout_sec)
            return {
                "answer": "Latest materials collected",
                "results": [
                    {
                        "title": "Article A",
                        "url": "https://example.com/a",
                        "snippet": "Alpha",
                    },
                    {
                        "title": "Article B",
                        "url": "https://example.com/b",
                        "snippet": "Beta",
                    },
                ],
            }

        service = SearchIntegrationService(
            SearchIntegrationConfig(
                enabled=True,
                base_url="https://search.example",
                search_endpoint="/api/v1/search/",
                fetch_endpoint="/api/v1/fetch/",
                timeout_sec=9.0,
                max_results_default=5,
                fetch_top_n_default=3,
                source_label="Search",
            ),
            transport=fake_transport,
        )
        result = service.search(
            "fresh materials about agents",
            max_results=2,
        )
        payload = service.build_artifact_payload(
            result,
            artifact_name="agent_search",
            tool_name="search_tool",
        )

        self.assertEqual(payload["artifact_name"], "agent_search")
        self.assertEqual(len(payload["rows"]), 2)
        self.assertEqual(
            list(payload["rows"][0].keys()),
            ["rank", "title", "url", "snippet", "source_name", "published_at"],
        )
        self.assertEqual(payload["source"]["source_type"], "search")
        self.assertEqual(payload["recipe"][0]["kind"], "source_query")
        self.assertEqual(payload["recipe"][0]["query_text"], "fresh materials about agents")
        self.assertEqual(payload["meta"]["search"]["result_count"], 2)

        meta = build_artifact_meta(
            tool_name="search_tool",
            source_context={
                "source_type": "csv",
                "source_ref_id": "legacy.csv",
                "source_label": "Legacy CSV",
            },
            artifact_hints=payload,
        )

        self.assertEqual(meta["source"]["source_type"], "search")
        self.assertEqual(meta["source"]["source_ref_id"], "search")
        self.assertEqual(meta["provenance"]["source"], meta["source"])
        self.assertEqual(meta["provenance"]["recipe"], meta["recipe"])
        self.assertEqual(meta["recipe"][0]["kind"], "source_query")

    def test_service_synthesizes_answer_when_bridge_returns_null_answer(self) -> None:
        """search_service sets answer=null; integration builds a snippet summary for the agent."""

        def fake_transport(url: str, payload: dict[str, object], timeout_sec: float) -> dict[str, object]:
            _ = (url, payload, timeout_sec)
            return {
                "answer": None,
                "results": [
                    {
                        "title": "Paris - Wikipedia",
                        "url": "https://en.wikipedia.org/wiki/Paris",
                        "snippet": "Paris is the capital and largest city of France.",
                    },
                ],
                "sources": ["https://en.wikipedia.org/wiki/Paris"],
            }

        service = SearchIntegrationService(
            SearchIntegrationConfig(
                enabled=True,
                base_url="https://search.example",
                search_endpoint="/api/v1/search/",
                fetch_endpoint="/api/v1/fetch/",
                timeout_sec=9.0,
                max_results_default=5,
                fetch_top_n_default=0,
            ),
            transport=fake_transport,
        )
        result = service.search("capital of France", max_results=2, fetch_top_n=0)
        self.assertIsNotNone(result.answer)
        assert result.answer is not None
        self.assertIn("Paris", result.answer)
        self.assertIn("capital", result.answer.lower())


class SearchToolRunDirectPayloadTests(unittest.TestCase):
    """Dict-style tool input must expose payload['table'] for ToolCollector."""

    def test_run_direct_uses_table_key_not_items_envelope(self) -> None:
        def fake_transport(url: str, payload: dict[str, object], timeout_sec: float) -> dict[str, object]:
            _ = (url, payload, timeout_sec)
            return {
                "answer": "Synth",
                "results": [
                    {"title": "T1", "url": "https://x/1", "snippet": "S1"},
                ],
            }

        service = SearchIntegrationService(
            SearchIntegrationConfig(
                enabled=True,
                base_url="https://search.example",
                search_endpoint="/api/v1/search/",
                fetch_endpoint="/api/v1/fetch/",
                timeout_sec=9.0,
                max_results_default=5,
                fetch_top_n_default=0,
            ),
            transport=fake_transport,
        )
        tool = SearchTool(pd.DataFrame(), search_service=service)
        _text, payload = tool._run_direct({"query": "q"})

        self.assertIn("table", payload)
        self.assertIsInstance(payload["table"], dict)
        self.assertEqual(len(payload["table"]), 1)
        df = next(iter(payload["table"].values()))
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 1)
        # Raw tool-result envelope must not sit at top level (breaks artifact extraction)
        self.assertNotIn("items", payload)

    def test_run_accepts_query_kwarg_like_langchain(self) -> None:
        """Structured tool calls pass query=... as kwargs; _run must not raise TypeError."""

        def fake_transport(url: str, payload: dict[str, object], timeout_sec: float) -> dict[str, object]:
            _ = (url, payload, timeout_sec)
            return {
                "answer": "Synth",
                "results": [
                    {"title": "T1", "url": "https://x/1", "snippet": "S1"},
                ],
            }

        service = SearchIntegrationService(
            SearchIntegrationConfig(
                enabled=True,
                base_url="https://search.example",
                search_endpoint="/api/v1/search/",
                fetch_endpoint="/api/v1/fetch/",
                timeout_sec=9.0,
                max_results_default=5,
                fetch_top_n_default=0,
            ),
            transport=fake_transport,
        )
        tool = SearchTool(pd.DataFrame(), search_service=service)
        _text, payload = tool._run(query="capital of France", run_manager=None)

        self.assertIn("table", payload)
        self.assertIsInstance(payload["table"], dict)

    def test_run_accepts_queries_list_kwarg(self) -> None:
        def fake_transport(url: str, payload: dict[str, object], timeout_sec: float) -> dict[str, object]:
            _ = (url, payload, timeout_sec)
            return {
                "answer": "Synth",
                "results": [
                    {"title": "T1", "url": "https://x/1", "snippet": "S1"},
                ],
            }

        service = SearchIntegrationService(
            SearchIntegrationConfig(
                enabled=True,
                base_url="https://search.example",
                search_endpoint="/api/v1/search/",
                fetch_endpoint="/api/v1/fetch/",
                timeout_sec=9.0,
                max_results_default=5,
                fetch_top_n_default=0,
            ),
            transport=fake_transport,
        )
        tool = SearchTool(pd.DataFrame(), search_service=service)
        _text, payload = tool._run(queries=["one", "two"], run_manager=None)

        self.assertIn("table", payload)


if __name__ == "__main__":
    unittest.main()
