from __future__ import annotations

import json
import unittest

from backend.integrations import RAGConfig, RAGService


class RAGServiceTests(unittest.TestCase):
    def test_config_from_env_detects_enabled_source(self) -> None:
        config = RAGConfig.from_env(
            {
                "RAG_ENABLED": "true",
                "RAG_URL": "https://rag.example",
                "RAG_QUERY_ENDPOINT": "/v1/query",
                "RAG_STREAM_ENDPOINT": "/v1/query/stream",
                "RAG_TIMEOUT_SEC": "18",
                "RAG_VERIFY_SSL": "true",
                "RAG_QUERY_MODE": "local",
                "RAG_TOP_K": "9",
                "RAG_SOURCE_LABEL": "Knowledge Base",
            }
        )

        self.assertTrue(config.enabled)
        self.assertTrue(config.available)
        self.assertEqual(config.base_url, "https://rag.example")
        self.assertEqual(config.query_endpoint, "/v1/query")
        self.assertEqual(config.stream_endpoint, "/v1/query/stream")
        self.assertEqual(config.timeout_sec, 18.0)
        self.assertTrue(config.verify_ssl)
        self.assertEqual(config.query_mode_default, "local")
        self.assertEqual(config.top_k_default, 9)
        self.assertEqual(config.source_label, "Knowledge Base")

    def test_service_normalizes_query_and_stream(self) -> None:
        captured: dict[str, object] = {}

        def fake_transport(
            url: str,
            payload: dict[str, object],
            timeout_sec: float,
            verify_ssl: bool,
        ) -> dict[str, object]:
            captured["url"] = url
            captured["payload"] = dict(payload)
            captured["timeout_sec"] = timeout_sec
            captured["verify_ssl"] = verify_ssl
            return {
                "response": "Use the auth token from the login endpoint.",
                "references": [
                    {"url": "https://docs.example/auth"},
                    "https://docs.example/login",
                ],
            }

        def fake_stream_transport(
            url: str,
            payload: dict[str, object],
            timeout_sec: float,
            verify_ssl: bool,
        ) -> list[str]:
            captured["stream_url"] = url
            captured["stream_payload"] = dict(payload)
            captured["stream_timeout_sec"] = timeout_sec
            captured["stream_verify_ssl"] = verify_ssl
            return [
                json.dumps({"response": "Use the auth token "}),
                json.dumps({"response": "from the login endpoint."}),
            ]

        service = RAGService(
            RAGConfig(
                enabled=True,
                base_url="https://rag.example",
                query_endpoint="/query",
                stream_endpoint="/query/stream",
                timeout_sec=14.0,
                verify_ssl=False,
                query_mode_default="hybrid",
                top_k_default=5,
            ),
            transport=fake_transport,
            stream_transport=fake_stream_transport,
        )

        result = service.search(
            query="how does auth work?",
            mode="global",
            top_k=4,
            include_references=True,
        )
        chunks = list(
            service.stream_search(
                query="how does auth work?",
                mode="global",
                top_k=4,
            )
        )

        self.assertEqual(captured["url"], "https://rag.example/query")
        self.assertEqual(
            captured["payload"],
            {
                "query": "how does auth work?",
                "mode": "global",
                "top_k": 4,
                "include_references": True,
            },
        )
        self.assertEqual(captured["timeout_sec"], 14.0)
        self.assertFalse(captured["verify_ssl"])
        self.assertEqual(result.answer, "Use the auth token from the login endpoint.")
        self.assertEqual(
            result.references,
            ["https://docs.example/auth", "https://docs.example/login"],
        )
        self.assertEqual(chunks, ["Use the auth token ", "from the login endpoint."])
        self.assertEqual(captured["stream_url"], "https://rag.example/query/stream")
        self.assertEqual(
            captured["stream_payload"],
            {
                "query": "how does auth work?",
                "mode": "global",
                "top_k": 4,
                "include_references": False,
                "stream": True,
            },
        )

    def test_format_for_user_handles_empty_answer(self) -> None:
        service = RAGService(
            RAGConfig(
                enabled=True,
                base_url="https://rag.example",
                query_endpoint="/query",
                stream_endpoint="/query/stream",
                timeout_sec=10.0,
                verify_ssl=False,
                query_mode_default="hybrid",
                top_k_default=5,
            ),
            transport=lambda *_args: {"references": ["https://docs.example/no-answer"]},
        )

        result = service.search(query="empty answer case")
        self.assertIn("RAG", service.format_for_user(result))


if __name__ == "__main__":
    unittest.main()
