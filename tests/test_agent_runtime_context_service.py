import pytest

from backend.agent.services.runtime_context import (
    build_runtime_metadata,
    extract_db_connection_id,
    normalize_session_source_for_sql_mode,
    resolve_csv_runtime_state,
    resolve_tool_db_runtime_config,
)


class _FakeDBRuntimeService:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def get_runtime_config(self, *, user_id: int, connection_id: str) -> dict[str, object]:
        self.calls.append((user_id, connection_id))
        return {"user_id": user_id, "connection_id": connection_id}


def test_build_runtime_metadata_preserves_trace_contract_fields() -> None:
    metadata = build_runtime_metadata(
        {
            "session_id": "s1",
            "user_id": 42,
            "username": "analyst",
            "request_kind": "query",
            "db_connection_id": "db1",
            "csv_duckdb_loaded": True,
            "csv_session_id": "csv1",
        }
    )

    assert metadata == {
        "session_id": "s1",
        "thread_id": "s1",
        "conversation_id": "s1",
        "user_id": "42",
        "username": "analyst",
        "request_kind": "query",
        "db_connection_id": "db1",
        "data_source": "db_connection",
        "csv_session_id": "csv1",
    }


def test_extract_db_connection_id_prefers_session_source() -> None:
    assert (
        extract_db_connection_id(
            {"source_type": "db_connection", "source_ref_id": "source-db"},
            {"db_connection_id": "trace-db"},
        )
        == "source-db"
    )


def test_resolve_csv_runtime_state_uses_trace_fallback() -> None:
    assert resolve_csv_runtime_state({}, {"csv_duckdb_loaded": True, "session_id": "s1"}) == (
        True,
        "s1",
    )


def test_normalize_session_source_for_sql_mode_marks_csv_duckdb_source() -> None:
    normalized = normalize_session_source_for_sql_mode(
        {},
        {"csv_duckdb_loaded": True, "csv_session_id": "csv1"},
    )

    assert normalized == {
        "source_type": "db_connection",
        "source_ref_id": "csv1",
        "source_label": "CSV DuckDB session csv1",
        "source_mode": "read_only",
        "csv_loaded": True,
        "csv_session_id": "csv1",
    }


def test_resolve_tool_db_runtime_config_uses_explicit_service_contract() -> None:
    service = _FakeDBRuntimeService()

    runtime = resolve_tool_db_runtime_config(
        db_runtime_service=service,
        session_source={"source_type": "db_connection", "source_ref_id": "db1"},
        trace_context={"user_id": "7"},
    )

    assert runtime == {"user_id": 7, "connection_id": "db1"}
    assert service.calls == [(7, "db1")]


def test_resolve_tool_db_runtime_config_requires_user_for_db_source() -> None:
    with pytest.raises(RuntimeError, match=r"trace_context\.user_id"):
        resolve_tool_db_runtime_config(
            db_runtime_service=_FakeDBRuntimeService(),
            session_source={"source_type": "db_connection", "source_ref_id": "db1"},
            trace_context={},
        )
