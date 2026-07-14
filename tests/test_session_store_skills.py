from __future__ import annotations

from pathlib import Path

from backend.sessions.session_store import SessionStore


def test_session_store_persists_selected_skill_ids(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=7)
    session = store.create_session()

    store.set_selected_skill_ids(session.session_id, ["cohort_analysis", "forecasting", "cohort_analysis"])
    reloaded = store.load_session(session.session_id)

    assert reloaded is not None
    assert reloaded.selected_skill_ids == ["cohort_analysis", "forecasting"]


def test_session_store_persists_context_usage_snapshot(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=7)
    session = store.create_session()

    store.set_context_usage(
        session.session_id,
        {
            "input_tokens": 100,
            "reserved_response_tokens": 20,
            "used_tokens": 120,
            "max_context_tokens": 200,
            "remaining_tokens": 80,
            "usage_ratio": 0.6,
            "usage_percent": 60,
            "overflow": False,
            "status": "normal",
            "context_window_source": "settings",
        },
    )
    reloaded = store.load_session(session.session_id)

    assert reloaded is not None
    assert reloaded.context_usage["usage_percent"] == 60
