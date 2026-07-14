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

    def test_retrieve_sends_only_need_context(self) -> None:
        captured: dict[str, object] = {}

        def fake_transport(
            url: str,
            payload: dict[str, object],
            timeout_sec: float,
            verify_ssl: bool,
        ) -> dict[str, object]:
            captured["url"] = url
            captured["payload"] = dict(payload)
            return {"data": "Chunk A\n-----\nChunk B"}

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
            transport=fake_transport,
        )

        result = service.retrieve(query="what is X?", mode="local", top_k=3)

        self.assertEqual(captured["payload"]["only_need_context"], True)
        self.assertNotIn("include_references", captured["payload"])
        self.assertEqual(result.answer, "Chunk A\n-----\nChunk B")
        self.assertEqual(result.references, [])

    def test_upload_document_uses_lightrag_multipart_endpoint(self) -> None:
        captured: dict[str, object] = {}

        def fake_upload_transport(
            url: str,
            file_name: str,
            content: bytes,
            content_type: str,
            timeout_sec: float,
            verify_ssl: bool,
        ) -> dict[str, object]:
            captured["url"] = url
            captured["file_name"] = file_name
            captured["content"] = content
            captured["content_type"] = content_type
            captured["timeout_sec"] = timeout_sec
            captured["verify_ssl"] = verify_ssl
            return {
                "status": "success",
                "message": "uploaded",
                "track_id": "upload_123",
            }

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
            upload_transport=fake_upload_transport,
        )

        result = service.upload_document(
            file_name="policy.txt",
            content=b"policy text",
            content_type="text/plain",
        )

        self.assertEqual(captured["url"], "https://rag.example/documents/upload")
        self.assertEqual(captured["file_name"], "policy.txt")
        self.assertEqual(captured["content"], b"policy text")
        self.assertEqual(captured["content_type"], "text/plain")
        self.assertEqual(captured["timeout_sec"], 10.0)
        self.assertFalse(captured["verify_ssl"])
        self.assertEqual(result["track_id"], "upload_123")

    def test_track_status_normalizes_lightrag_document_status(self) -> None:
        def fake_get_transport(
            url: str,
            timeout_sec: float,
            verify_ssl: bool,
        ) -> dict[str, object]:
            self.assertEqual(
                url,
                "https://rag.example/documents/track_status/upload_123",
            )
            self.assertEqual(timeout_sec, 10.0)
            self.assertFalse(verify_ssl)
            return {
                "track_id": "upload_123",
                "status_summary": {"DocStatus.PROCESSED": 1},
                "documents": [
                    {
                        "id": "doc-1",
                        "file_path": "policy.txt",
                        "status": "processed",
                        "chunks_count": 2,
                    }
                ],
            }

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
            get_transport=fake_get_transport,
        )

        result = service.get_track_status("upload_123")

        self.assertEqual(result["track_id"], "upload_123")
        self.assertEqual(result["status_summary"], {"processed": 1})
        self.assertEqual(result["documents"][0]["status"], "processed")
        self.assertEqual(result["documents"][0]["file_path"], "policy.txt")

    def test_list_documents_flattens_lightrag_status_buckets(self) -> None:
        def fake_get_transport(
            url: str,
            timeout_sec: float,
            verify_ssl: bool,
        ) -> dict[str, object]:
            self.assertEqual(url, "https://rag.example/documents")
            _ = (timeout_sec, verify_ssl)
            return {
                "statuses": {
                    "processed": [
                        {"id": "doc-1", "file_path": "a.txt", "status": "processed"}
                    ],
                    "failed": [
                        {"id": "doc-2", "file_path": "b.txt", "error_msg": "boom"}
                    ],
                }
            }

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
            get_transport=fake_get_transport,
        )

        result = service.list_documents()

        self.assertEqual(len(result["documents"]), 2)
        self.assertEqual(result["documents"][0]["status"], "processed")
        self.assertEqual(result["documents"][1]["status"], "failed")

    def test_delete_document_uses_lightrag_delete_endpoint(self) -> None:
        captured: dict[str, object] = {}

        def fake_delete_transport(
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
                "status": "deletion_started",
                "message": "Document deletion started",
                "doc_id": "doc-123",
            }

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
            delete_transport=fake_delete_transport,
        )

        result = service.delete_document("doc-123")

        self.assertEqual(captured["url"], "https://rag.example/documents/delete_document")
        self.assertEqual(
            captured["payload"],
            {
                "doc_ids": ["doc-123"],
                "delete_file": False,
                "delete_llm_cache": False,
            },
        )
        self.assertEqual(captured["timeout_sec"], 10.0)
        self.assertFalse(captured["verify_ssl"])
        self.assertEqual(result["status"], "deletion_started")
        self.assertEqual(result["document_id"], "doc-123")

    def test_delete_document_rejects_empty_document_id(self) -> None:
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
        )

        with self.assertRaisesRegex(Exception, "document_id"):
            service.delete_document(" ")

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
