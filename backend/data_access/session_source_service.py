from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backend.data_access.catalog_refresh import refresh_session_catalog
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.orchestrator import NotebookOrchestrator
from backend.notebook.session_source import SessionSource
from backend.sessions.session_store import SessionStore


class SessionSourceError(RuntimeError):
    """Raised when a session source cannot be changed consistently."""


@dataclass
class SessionSourceService:
    store: SessionStore
    csv_runtime: CSVSessionRuntime
    manifest_store: ManifestStore
    notebook_orchestrator: NotebookOrchestrator
    storage_dir: str | Path
    db_runtime: Any | None = None

    def remove_source(self, *, session_id: str, alias: str) -> SessionSource:
        clean_alias = str(alias or "").strip()
        if not clean_alias:
            raise SessionSourceError("Source alias must not be empty")

        manifest = self.manifest_store.load(session_id)
        removed = manifest.remove_source(clean_alias)
        if removed is None:
            raise SessionSourceError(f"Source '{clean_alias}' not found")

        if removed.source_type == "csv":
            self._remove_csv_runtime_tables(session_id, removed)
            self._remove_parquet_file(session_id, removed)

        self.manifest_store.save(session_id, manifest)
        self._remove_notebook_binding(session_id, clean_alias)
        self._sync_session_state_after_removal(session_id, removed)
        self._refresh_catalog(session_id)
        return removed

    def _remove_csv_runtime_tables(self, session_id: str, source: SessionSource) -> None:
        table_names = [
            str(table_name).strip()
            for table_name in source.csv_table_names
            if str(table_name).strip()
        ]
        if table_names:
            self.csv_runtime.unregister_tables(session_id, table_names)

    def _remove_parquet_file(self, session_id: str, source: SessionSource) -> None:
        path = self._source_parquet_path(session_id, source)
        if path is None:
            return
        with suppress(FileNotFoundError):
            path.unlink()

    def _source_parquet_path(self, session_id: str, source: SessionSource) -> Path | None:
        if not source.parquet_path:
            return None
        path = Path(source.parquet_path)
        if path.is_absolute():
            return path
        return Path(self.storage_dir) / "sessions" / session_id / path

    def _remove_notebook_binding(self, session_id: str, alias: str) -> None:
        result = self.notebook_orchestrator.remove_source_binding(session_id, alias)
        if not result.ok:
            raise SessionSourceError(result.error or "Failed to remove source binding cell")

    def _sync_session_state_after_removal(
        self,
        session_id: str,
        removed: SessionSource,
    ) -> None:
        state = self.store.load_session(session_id)
        if state is None:
            raise SessionSourceError("Session not found")

        if removed.source_type == "db_connection":
            active_db = str(state.source_type or "").strip().lower() == "db_connection"
            if active_db and state.source_ref_id == removed.connection_id:
                self.store.set_source(
                    session_id,
                    source_type=None,
                    source_ref_id=None,
                    source_label=None,
                    source_mode=None,
                )
            return

        csv_sources = self._remaining_csv_sources(session_id)
        if csv_sources:
            self._sync_remaining_csv_sources(session_id, csv_sources, state.source_type)
            return

        self.csv_runtime.unregister_tables(session_id, state.csv_table_names or [])
        self.store.clear_csv_runtime_state(session_id)
        self.store.clear_dataframe(session_id)
        if str(state.source_type or "").strip().lower() == "csv":
            self.store.set_source(
                session_id,
                source_type=None,
                source_ref_id=None,
                source_label=None,
                source_mode=None,
            )

    def _remaining_csv_sources(self, session_id: str) -> list[SessionSource]:
        manifest = self.manifest_store.load(session_id)
        return [source for source in manifest.sources if source.source_type == "csv"]

    def _sync_remaining_csv_sources(
        self,
        session_id: str,
        csv_sources: list[SessionSource],
        active_source_type: str | None,
    ) -> None:
        csv_tables = sorted(
            {
                table_name
                for source in csv_sources
                for table_name in source.csv_table_names
                if str(table_name).strip()
            }
        )
        info = self.csv_runtime.get_session_info(session_id)
        self.store.set_csv_runtime_state(
            session_id,
            csv_loaded=True,
            csv_session_id=session_id,
            csv_table_names=csv_tables,
            csv_expires_at=info.expires_at,
        )
        label = self._dataset_label(csv_sources)
        self.store.set_dataset_name(session_id, label)
        self._save_legacy_dataframe_from_first_source(session_id, csv_sources)
        if str(active_source_type or "").strip().lower() == "csv":
            self.store.bind_csv_source(session_id, filename=label)

    def _save_legacy_dataframe_from_first_source(
        self,
        session_id: str,
        csv_sources: list[SessionSource],
    ) -> None:
        for source in csv_sources:
            path = self._source_parquet_path(session_id, source)
            if path is None or not path.is_file():
                continue
            self.store.save_dataframe(
                session_id,
                pd.read_parquet(path, engine="pyarrow"),
            )
            return

    @staticmethod
    def _dataset_label(sources: list[SessionSource]) -> str:
        names = [
            str(source.display_name or source.file_name or source.alias).strip()
            for source in sources
            if str(source.display_name or source.file_name or source.alias).strip()
        ]
        if len(names) <= 3:
            return ", ".join(names)
        return f"{', '.join(names[:3])} +{len(names) - 3} more"

    def _refresh_catalog(self, session_id: str) -> None:
        refresh_session_catalog(
            self.store,
            session_id,
            csv_runtime=self.csv_runtime,
            db_runtime=self.db_runtime,
        )
