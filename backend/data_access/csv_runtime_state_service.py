from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.session_source import SessionSource
from backend.sessions.session_store import SessionState, SessionStore


class CSVRuntimeStateError(RuntimeError):
    """Raised when CSV runtime state cannot be initialized for a session."""


class CSVRuntimeStateService:
    """Ensure the session DuckDB runtime is available and in sync with sources."""

    def __init__(
        self,
        *,
        store: SessionStore,
        csv_runtime: CSVSessionRuntime,
        manifest_store: ManifestStore,
        storage_dir: str | Path,
    ) -> None:
        self._store = store
        self._csv_runtime = csv_runtime
        self._manifest_store = manifest_store
        self._storage_dir = Path(storage_dir)

    def ensure_csv_runtime(
        self,
        *,
        session_id: str,
        ttl_seconds: int | None = None,
    ) -> SessionState:
        state = self._store.load_session(session_id)
        if state is None:
            raise CSVRuntimeStateError("Session not found")

        if str(state.source_type or "").strip().lower() != "csv":
            return state

        if self._is_runtime_valid(state):
            return state

        tables = self._load_manifest_tables(session_id)
        if not tables:
            tables = self._load_legacy_table(session_id, state)
        if not tables:
            raise CSVRuntimeStateError("CSV dataset is not attached to this session")

        info = self._csv_runtime.register_dataframes(
            session_id=session_id,
            tables=tables,
            ttl_seconds=ttl_seconds,
        )
        self._store.set_csv_runtime_state(
            session_id,
            csv_loaded=True,
            csv_session_id=info.session_id,
            csv_table_names=list(info.table_names),
            csv_expires_at=info.expires_at,
        )
        refreshed = self._store.load_session(session_id)
        if refreshed is None:
            raise CSVRuntimeStateError("Session not found")
        return refreshed

    def _is_runtime_valid(self, state: SessionState) -> bool:
        if not state.csv_loaded or not state.csv_session_id:
            return False
        if not self._csv_runtime.db_exists(state.csv_session_id):
            return False
        if state.csv_expires_at is not None and state.csv_expires_at <= int(time.time()) + 60:
            return False
        return True

    def _load_manifest_tables(self, session_id: str) -> dict[str, pd.DataFrame]:
        manifest = self._manifest_store.load(session_id)
        tables: dict[str, pd.DataFrame] = {}
        for source in manifest.sources:
            if source.source_type != "csv":
                continue
            loaded = self._load_source_dataframe(session_id, source)
            if loaded is None:
                continue
            table_name = self._table_name_for_source(source, tables)
            tables[table_name] = loaded
        return tables

    def _load_source_dataframe(
        self,
        session_id: str,
        source: SessionSource,
    ) -> pd.DataFrame | None:
        if not source.parquet_path:
            return None
        path = Path(source.parquet_path)
        if not path.is_absolute():
            path = self._storage_dir / "sessions" / session_id / path
        if not path.is_file():
            return None
        return pd.read_parquet(path, engine="pyarrow")

    @staticmethod
    def _table_name_for_source(source: SessionSource, existing: dict[str, pd.DataFrame]) -> str:
        if source.csv_table_names:
            candidate = str(source.csv_table_names[0]).strip()
            if candidate and candidate not in existing:
                return candidate
        return CSVSessionRuntime.unique_table_name(
            source.file_name or source.display_name or source.alias,
            existing.keys(),
        )

    def _load_legacy_table(self, session_id: str, state: SessionState) -> dict[str, pd.DataFrame]:
        if not state.df_path:
            return {}
        df = self._store.get_dataframe(session_id)
        if df is None:
            return {}
        table_name = CSVSessionRuntime.sanitize_table_name(state.dataset_name or "uploaded.csv")
        return {table_name: df}
