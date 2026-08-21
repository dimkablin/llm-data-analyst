from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path

import pandas as pd

from backend.auth.blob_store import PostgresBlobStore
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.data_access.tabular_preprocessing import (
    TabularPreprocessingOptions,
    read_tabular_dataframe,
)
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.session_source import SessionSource, is_duckdb_source_type
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
        blob_store: PostgresBlobStore | None = None,
    ) -> None:
        self._store = store
        self._csv_runtime = csv_runtime
        self._manifest_store = manifest_store
        self._storage_dir = Path(storage_dir)
        self._blob_store = blob_store

    def ensure_csv_runtime(
        self,
        *,
        session_id: str,
        ttl_seconds: int | None = None,
    ) -> SessionState:
        state = self._store.load_session(session_id)
        if state is None:
            raise CSVRuntimeStateError("Session not found")

        if not is_duckdb_source_type(state.source_type):
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
            if not is_duckdb_source_type(source.source_type):
                continue
            loaded_tables = self._load_source_tables(session_id, source)
            if loaded_tables:
                tables.update(
                    (name, df)
                    for name, df in loaded_tables.items()
                    if name not in tables
                )
                continue
            loaded = self._load_source_dataframe(session_id, source)
            if loaded is None:
                continue
            table_name = self._table_name_for_source(source, tables)
            tables[table_name] = loaded
        return tables

    def _load_source_tables(
        self,
        session_id: str,
        source: SessionSource,
    ) -> dict[str, pd.DataFrame]:
        raw_paths = (source.preprocessing_summary or {}).get("duckdb_table_paths")
        if not isinstance(raw_paths, dict):
            return {}
        tables: dict[str, pd.DataFrame] = {}
        raw_blob_ids = (source.preprocessing_summary or {}).get("duckdb_table_blob_ids")
        blob_ids = raw_blob_ids if isinstance(raw_blob_ids, dict) else {}
        for table_name, raw_path in raw_paths.items():
            clean_name = str(table_name or "").strip()
            if not clean_name:
                continue
            path = self._resolve_session_path(session_id, raw_path)
            if path is not None:
                tables[clean_name] = pd.read_parquet(path, engine="pyarrow")
                continue
            blob_id = str(blob_ids.get(clean_name) or "").strip()
            restored = self._load_parquet_blob(session_id, blob_id)
            if restored is None:
                continue
            tables[clean_name] = restored
            self._materialize_parquet(session_id, raw_path, restored)
        return tables

    def _load_source_dataframe(
        self,
        session_id: str,
        source: SessionSource,
    ) -> pd.DataFrame | None:
        if not source.parquet_path:
            return None
        path = self._resolve_session_path(session_id, source.parquet_path)
        if path is not None:
            return pd.read_parquet(path, engine="pyarrow")
        if self._blob_store is None or not source.blob_id or not source.file_name:
            return None
        blob = self._blob_store.get_for_session(
            session_id=session_id,
            blob_id=source.blob_id,
            kind="source_upload",
        )
        if blob is None:
            return None
        suffix = Path(source.file_name).suffix.lower().lstrip(".")
        if suffix not in {"csv", "xlsx"}:
            return None
        options = TabularPreprocessingOptions.model_validate(
            (source.preprocessing_summary or {}).get("preprocessing_options") or {}
        )
        dataframe = read_tabular_dataframe(
            blob.content,
            file_format=suffix,
            options=options,
        ).dataframe
        self._materialize_parquet(session_id, source.parquet_path, dataframe)
        return dataframe

    def _load_parquet_blob(self, session_id: str, blob_id: str) -> pd.DataFrame | None:
        if self._blob_store is None or not blob_id:
            return None
        blob = self._blob_store.get_for_session(
            session_id=session_id,
            blob_id=blob_id,
            kind="runtime_snapshot",
        )
        if blob is None:
            return None
        return pd.read_parquet(BytesIO(blob.content), engine="pyarrow")

    def _materialize_parquet(
        self,
        session_id: str,
        raw_path: object,
        dataframe: pd.DataFrame,
    ) -> None:
        path_text = str(raw_path or "").strip()
        if not path_text:
            return
        path = Path(path_text)
        if not path.is_absolute():
            path = self._storage_dir / "sessions" / session_id / path
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(path, engine="pyarrow")

    def _resolve_session_path(self, session_id: str, raw_path: object) -> Path | None:
        path_text = str(raw_path or "").strip()
        if not path_text:
            return None
        path = Path(path_text)
        if not path.is_absolute():
            path = self._storage_dir / "sessions" / session_id / path
        if not path.is_file():
            return None
        return path

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
