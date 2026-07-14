from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import data


@dataclass(frozen=True)
class _CSVInfo:
    session_id: str
    expires_at: int


class _FakeCSVRuntime:
    def get_session_info(self, session_id: str) -> _CSVInfo:
        assert session_id == "csv-runtime-session"
        return _CSVInfo(session_id=session_id, expires_at=1_800_000_000)

    def list_tables(self, session_id: str) -> list[dict[str, str]]:
        assert session_id == "csv-runtime-session"
        return [{"table_name": "sales"}]

    def describe_table(self, session_id: str, table_name: str) -> list[dict[str, object]]:
        assert session_id == "csv-runtime-session"
        assert table_name == "sales"
        return [
            {
                "column_name": "date",
                "data_type": "DATE",
                "is_nullable": True,
                "ordinal_position": 1,
            },
            {
                "column_name": "revenue",
                "data_type": "DOUBLE",
                "is_nullable": True,
                "ordinal_position": 2,
            },
        ]


def test_csv_schema_is_available_to_predict_service_without_auth() -> None:
    data.setup(
        auth_db=None,  # type: ignore[arg-type]
        store=None,  # type: ignore[arg-type]
        csv_runtime=_FakeCSVRuntime(),  # type: ignore[arg-type]
        manifest_store=None,  # type: ignore[arg-type]
        notebook_orchestrator=None,  # type: ignore[arg-type]
    )

    app = FastAPI()
    app.include_router(data.router)
    client = TestClient(app)

    response = client.get(
        "/csv/schema",
        params={"session_id": "csv-runtime-session"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "csv-runtime-session"
    assert payload["table_name"] == "sales"
    assert [column["name"] for column in payload["columns"]] == [
        "date",
        "revenue",
    ]
