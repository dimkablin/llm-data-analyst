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
