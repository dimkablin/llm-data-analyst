from types import SimpleNamespace

from backend.api.routes import sessions


def test_session_cleanup_preserves_shared_csv_and_clears_owned_catalog(monkeypatch) -> None:
    class Store:
        states = {
            "csv-a": SimpleNamespace(source_type="csv", source_ref_id="same"),
            "csv-b": SimpleNamespace(source_type="csv", source_ref_id="same"),
            "planfact": SimpleNamespace(source_type="planfact", source_ref_id="planfact"),
            "database": SimpleNamespace(source_type="db_connection", source_ref_id="connection"),
        }

        def load_session(self, session_id: str):
            return self.states.get(session_id)

    class Auth:
        session_ids = ["csv-a", "csv-b", "planfact", "database"]

        def list_sessions(self, _user_id: int) -> list[dict[str, str]]:
            return [{"session_id": session_id} for session_id in self.session_ids]

    class SemanticCatalog:
        def __init__(self) -> None:
            self.cleared: list[str] = []

        def clear_for_session(self, *, session_id: str, user_id: int) -> None:
            assert user_id == 7
            self.cleared.append(session_id)

    auth = Auth()
    semantic_catalog = SemanticCatalog()
    monkeypatch.setattr(sessions, "_store", Store())
    monkeypatch.setattr(sessions, "_auth_db", auth)
    monkeypatch.setattr(sessions, "_semantic_catalog_service", semantic_catalog)

    sessions._clear_session_semantic_catalog("csv-a", 7, preserve_shared_csv=True)
    sessions._clear_session_semantic_catalog("database", 7, preserve_shared_csv=False)
    sessions._clear_session_semantic_catalog("planfact", 7, preserve_shared_csv=True)
    auth.session_ids.remove("csv-a")
    sessions._clear_session_semantic_catalog("csv-b", 7, preserve_shared_csv=True)

    assert semantic_catalog.cleared == ["planfact", "csv-b"]
