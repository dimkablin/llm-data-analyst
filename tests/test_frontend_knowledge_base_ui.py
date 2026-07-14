from pathlib import Path


def test_frontend_exposes_knowledge_base_source_contract() -> None:
    dashboard = Path("frontend/src/app/components/workspace/DashboardPanel.tsx").read_text(
        encoding="utf-8"
    )
    api = Path("frontend/src/app/lib/backend-api.ts").read_text(encoding="utf-8")
    types = Path("frontend/src/app/lib/backend-types.ts").read_text(encoding="utf-8")

    assert '"rag"' in types
    assert "uploadRagDocument" in api
    assert "getRagUploadStatus" in api
    assert "listRagDocuments" in api
    assert "deleteRagDocument" in api
    assert "bindRagSource" in api
    assert "handleDeleteRagDocument" in dashboard
    assert "База знаний" in dashboard
    assert "Загрузить документы" in dashboard
