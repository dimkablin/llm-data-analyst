from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.sessions.postgres_session_store import PostgresSessionStore
from backend.sessions.session_store import SessionBusyError, SessionStore


def test_file_store_query_lease_is_nonblocking_and_session_scoped(tmp_path) -> None:
    store = SessionStore(str(tmp_path), ttl_days=7)
    first = store.acquire_query_lease("session-a")
    other = store.acquire_query_lease("session-b")

    with pytest.raises(SessionBusyError):
        store.acquire_query_lease("session-a")

    first.release()
    store.acquire_query_lease("session-a").release()
    other.release()


@dataclass
class _SharedAdvisoryState:
    locked: bool = False


class _Result:
    def __init__(self, row: dict[str, bool]) -> None:
        self._row = row

    def fetchone(self) -> dict[str, bool]:
        return self._row


class _Connection:
    def __init__(self, state: _SharedAdvisoryState) -> None:
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, _params: tuple[object, ...] = ()) -> _Result:
        if "pg_try_advisory_lock" in query:
            acquired = not self.state.locked
            self.state.locked = True if acquired else self.state.locked
            return _Result({"acquired": acquired})
        if "pg_advisory_unlock" in query:
            self.state.locked = False
        return _Result({"acquired": False})


class _AppDataStore:
    def __init__(self) -> None:
        self.state = _SharedAdvisoryState()

    def connect(self) -> _Connection:
        return _Connection(self.state)


def test_postgres_query_lease_is_shared_between_store_instances(tmp_path) -> None:
    app_data = _AppDataStore()
    first_store = PostgresSessionStore(str(tmp_path / "a"), app_data_store=app_data)  # type: ignore[arg-type]
    second_store = PostgresSessionStore(str(tmp_path / "b"), app_data_store=app_data)  # type: ignore[arg-type]

    lease = first_store.acquire_query_lease("shared-session")
    with pytest.raises(SessionBusyError):
        second_store.acquire_query_lease("shared-session")

    lease.release()
    second_store.acquire_query_lease("shared-session").release()
    assert app_data.state.locked is False
