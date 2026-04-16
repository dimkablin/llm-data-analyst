from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.sessions.session_store import SessionStore
from backend.sessions.session_memory import (
    SessionArtifactRef,
    StructuredSessionMemory,
)


def _make_store(tmp_path: Path) -> SessionStore:
    return SessionStore(str(tmp_path), ttl_days=7)


def _make_ref(**kwargs) -> SessionArtifactRef:
    defaults = dict(
        id="art-1",
        name="my_table",
        type="table",
        turn_index=0,
        schema={"col_a": "str", "col_b": "int"},
        row_count=100,
        summary="Sample table",
    )
    defaults.update(kwargs)
    return SessionArtifactRef(**defaults)


# ---------------------------------------------------------------------------
# 1. Old session (only session_memory string, no new fields) loads cleanly
# ---------------------------------------------------------------------------

def test_get_structured_memory_old_session(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    session = store.create_session()
    sid = session.session_id

    # Manually patch the state file to simulate an old-format session
    # (no artifact_index_json / key_findings / session_turn_count keys)
    state_path = tmp_path / sid / "state.json"
    raw = json.loads(state_path.read_text())
    raw["session_memory"] = "old notes content"
    raw.pop("artifact_index_json", None)
    raw.pop("key_findings", None)
    raw.pop("session_turn_count", None)
    state_path.write_text(json.dumps(raw))

    mem = store.get_structured_memory(sid)

    assert mem.notes == "old notes content"
    assert mem.artifact_index == []
    assert mem.key_findings == []
    assert mem.turn_count == 0


# ---------------------------------------------------------------------------
# 2. Session not found → returns empty StructuredSessionMemory
# ---------------------------------------------------------------------------

def test_get_structured_memory_session_not_found(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    mem = store.get_structured_memory("nonexistent-session-id")

    assert isinstance(mem, StructuredSessionMemory)
    assert mem.notes == ""
    assert mem.artifact_index == []
    assert mem.key_findings == []
    assert mem.turn_count == 0


# ---------------------------------------------------------------------------
# 3. Full round-trip: set then get — all fields preserved
# ---------------------------------------------------------------------------

def test_set_and_get_structured_memory_roundtrip(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    session = store.create_session()
    sid = session.session_id

    ref = _make_ref(id="art-42", name="revenue_table", type="table", turn_index=3)
    memory = StructuredSessionMemory(
        notes="Session round-trip notes",
        artifact_index=[ref],
        key_findings=["Revenue up 10%", "Churn stable"],
        turn_count=7,
    )
    store.set_structured_memory(sid, memory)

    loaded = store.get_structured_memory(sid)

    assert loaded.notes == "Session round-trip notes"
    assert loaded.turn_count == 7
    assert loaded.key_findings == ["Revenue up 10%", "Churn stable"]
    assert len(loaded.artifact_index) == 1
    loaded_ref = loaded.artifact_index[0]
    assert loaded_ref.id == "art-42"
    assert loaded_ref.name == "revenue_table"
    assert loaded_ref.type == "table"
    assert loaded_ref.turn_index == 3


# ---------------------------------------------------------------------------
# 4. SessionArtifactRef with all fields serializes/deserializes intact
# ---------------------------------------------------------------------------

def test_artifact_index_json_roundtrip(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    session = store.create_session()
    sid = session.session_id

    ref = SessionArtifactRef(
        id="art-99",
        name="complex_table",
        type="table",
        turn_index=5,
        schema={"order_id": "int", "amount": "float", "region": "str"},
        row_count=2500,
        summary="Complex orders table",
    )
    memory = StructuredSessionMemory(artifact_index=[ref])
    store.set_structured_memory(sid, memory)

    loaded = store.get_structured_memory(sid)

    assert len(loaded.artifact_index) == 1
    r = loaded.artifact_index[0]
    assert r.id == "art-99"
    assert r.name == "complex_table"
    assert r.type == "table"
    assert r.turn_index == 5
    assert r.schema == {"order_id": "int", "amount": "float", "region": "str"}
    assert r.row_count == 2500
    assert r.summary == "Complex orders table"


# ---------------------------------------------------------------------------
# 5. Malformed artifact_index_json → returns empty list, no exception
# ---------------------------------------------------------------------------

def test_artifact_index_malformed_json_recovers(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    session = store.create_session()
    sid = session.session_id

    # Corrupt the artifact_index_json directly in the state file
    state_path = tmp_path / sid / "state.json"
    raw = json.loads(state_path.read_text())
    raw["artifact_index_json"] = "THIS IS NOT JSON {{{"
    state_path.write_text(json.dumps(raw))

    mem = store.get_structured_memory(sid)

    assert mem.artifact_index == []  # gracefully falls back


# ---------------------------------------------------------------------------
# 6. key_findings list of strings persists correctly
# ---------------------------------------------------------------------------

def test_key_findings_roundtrip(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    session = store.create_session()
    sid = session.session_id

    findings = ["Finding A", "Finding B", "Finding C with special chars: 2+2=4"]
    memory = StructuredSessionMemory(key_findings=findings)
    store.set_structured_memory(sid, memory)

    loaded = store.get_structured_memory(sid)

    assert loaded.key_findings == findings


# ---------------------------------------------------------------------------
# 7. turn_count=5 survives round-trip
# ---------------------------------------------------------------------------

def test_turn_count_persists(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    session = store.create_session()
    sid = session.session_id

    memory = StructuredSessionMemory(turn_count=5)
    store.set_structured_memory(sid, memory)

    loaded = store.get_structured_memory(sid)

    assert loaded.turn_count == 5


# ---------------------------------------------------------------------------
# 8. Backward compat: set_session_memory and append_session_memory still work
# ---------------------------------------------------------------------------

def test_existing_session_memory_methods_still_work(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    session = store.create_session()
    sid = session.session_id

    store.set_session_memory(sid, "Initial memory content")
    state = store._load_state(sid)
    assert state is not None
    assert state.session_memory == "Initial memory content"

    store.append_session_memory(sid, "appended note")
    state = store._load_state(sid)
    assert state is not None
    assert "appended note" in state.session_memory
    assert "Initial memory content" in state.session_memory

    # Also verify get_structured_memory reads from session_memory for notes
    mem = store.get_structured_memory(sid)
    assert "appended note" in mem.notes
    assert "Initial memory content" in mem.notes
