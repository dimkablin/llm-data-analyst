from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import deps
from backend.api.routes import rag_documents
from backend.auth.auth_db import AuthUser


@dataclass
class _Session:
    session_id: str
    source_type: str | None = None
    source_ref_id: str | None = None
    source_label: str | None = None
    source_mode: str | None = None


class _FakeAuthDB:
    def get_user_by_token(self, token: str) -> AuthUser | None:
        if token != "token":
            return None
        return AuthUser(id=7, username="analyst", is_admin=False, created_at="now")

    def is_session_owner(self, session_id: str, user_id: int) -> bool:
        return session_id == "sid" and user_id == 7


class _FakeStore:
    def __init__(self) -> None:
        self.session = _Session(session_id="sid")

    def load_session(self, session_id: str) -> _Session | None:
        if session_id != "sid":
            return None
        return self.session

    def set_source(
        self,
        session_id: str,
        *,
        source_type: str | None,
        source_ref_id: str | None,
        source_label: str | None,
        source_mode: str | None = None,
    ) -> None:
        assert session_id == "sid"
        self.session.source_type = source_type
        self.session.source_ref_id = source_ref_id
        self.session.source_label = source_label
        self.session.source_mode = source_mode


class _FakeRAGService:
    def __init__(self) -> None:
        self.uploaded: dict[str, object] | None = None
        self.deleted_document_id: str | None = None

    @property
    def is_enabled(self) -> bool:
        return True

    def upload_document(
        self,
        *,
        file_name: str,
        content: bytes,
        content_type: str,
    ) -> dict[str, object]:
        self.uploaded = {
            "file_name": file_name,
            "content": content,
            "content_type": content_type,
        }
        return {
            "status": "success",
            "message": "uploaded",
            "track_id": "upload_123",
        }

    def get_track_status(self, track_id: str) -> dict[str, object]:
        assert track_id == "upload_123"
        return {
            "track_id": track_id,
            "status_summary": {"processed": 1},
            "documents": [
                {
                    "id": "doc-1",
                    "file_path": "policy.txt",
                    "status": "processed",
                    "chunks_count": 1,
                }
            ],
        }

    def list_documents(self) -> dict[str, object]:
        return {
            "documents": [
                {
                    "id": "doc-1",
                    "file_path": "policy.txt",
                    "status": "processed",
                }
            ]
        }

    def delete_document(self, document_id: str) -> dict[str, object]:
        self.deleted_document_id = document_id
        return {
            "status": "deletion_started",
            "message": "Document deletion started",
            "document_id": document_id,
        }


def _client() -> tuple[TestClient, _FakeStore, _FakeRAGService]:
    auth_db = _FakeAuthDB()
    store = _FakeStore()
    rag_service = _FakeRAGService()
    deps.set_auth_db(auth_db)
    rag_documents.setup(
        auth_db=auth_db,
        store=store,
        rag_service=rag_service,
    )
    app = FastAPI()
    app.include_router(rag_documents.router)
    return TestClient(app), store, rag_service


def test_upload_rag_document_requires_owned_session_and_returns_track_id() -> None:
    client, _store, rag_service = _client()

    response = client.post(
        "/sessions/sid/rag/documents",
        headers={"Authorization": "Bearer token"},
        files={"file": ("policy.txt", b"policy text", "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["track_id"] == "upload_123"
    assert payload["status"] == "success"
    assert rag_service.uploaded == {
        "file_name": "policy.txt",
        "content": b"policy text",
        "content_type": "text/plain",
    }


def test_rag_document_status_and_list_are_proxied() -> None:
    client, _store, _rag_service = _client()

    status_response = client.get(
        "/sessions/sid/rag/uploads/upload_123",
        headers={"Authorization": "Bearer token"},
    )
    list_response = client.get(
        "/sessions/sid/rag/documents",
        headers={"Authorization": "Bearer token"},
    )

    assert status_response.status_code == 200
    assert status_response.json()["documents"][0]["file_path"] == "policy.txt"
    assert list_response.status_code == 200
    assert list_response.json()["documents"][0]["status"] == "processed"


def test_delete_rag_document_requires_owned_session_and_proxies_delete() -> None:
    client, _store, rag_service = _client()

    response = client.delete(
        "/sessions/sid/rag/documents/doc-1",
        headers={"Authorization": "Bearer token"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "deletion_started",
        "message": "Document deletion started",
        "document_id": "doc-1",
    }
    assert rag_service.deleted_document_id == "doc-1"


def test_bind_rag_source_sets_active_knowledge_base() -> None:
    client, store, _rag_service = _client()

    response = client.post(
        "/sessions/sid/source/rag",
        headers={"Authorization": "Bearer token"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "source_type": "rag",
        "source_ref_id": "rag",
        "source_label": "База знаний",
        "source_mode": "lightrag",
    }
    assert store.session.source_type == "rag"
