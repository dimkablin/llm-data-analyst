from __future__ import annotations

import hashlib
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.data_access.tabular_preprocessing import (
    PreprocessedTabularData,
    TabularPreprocessingOptions,
    TabularPreprocessingSummary,
    read_tabular_dataframe,
)
from backend.notebook.cell_builder import build_source_binding_cell
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.orchestrator import CellOp, NotebookEdit, NotebookOrchestrator
from backend.notebook.session_source import (
    SessionManifest,
    SessionSource,
    alias_to_variable_name,
    make_source_alias,
)
from backend.sessions.session_store import SessionStore

TabularFileFormat = Literal["csv", "xlsx"]

_SUPPORTED_EXTENSIONS: dict[str, TabularFileFormat] = {
    ".csv": "csv",
    ".xlsx": "xlsx",
}


class TabularUploadError(ValueError):
    """Raised when an uploaded tabular file cannot be ingested."""


class TabularUploadFile(BaseModel):
    file_name: str = Field(..., min_length=1, max_length=255)
    content: bytes = Field(..., min_length=1)
    content_type: str | None = Field(default=None, max_length=255)


class UploadedTabularFile(BaseModel):
    file_name: str
    file_format: TabularFileFormat
    table_name: str
    source_alias: str
    variable_name: str
    parquet_path: str
    rows: int
    columns: int
    preprocessing: TabularPreprocessingSummary


class TabularUploadResult(BaseModel):
    session_id: str
    csv_session_id: str
    table_names: list[str]
    files: list[UploadedTabularFile]
    expires_at: int
    total_rows: int
    total_columns: int
    dataset_name: str


class _ParsedTabularFile(BaseModel):
    file_name: str
    file_format: TabularFileFormat
    table_name: str
    df: object
    preprocessing: TabularPreprocessingSummary

    model_config = {"arbitrary_types_allowed": True}


class TabularUploadService:
    """Ingest uploaded CSV/XLSX files into the session DuckDB runtime."""

    def __init__(
        self,
        *,
        store: SessionStore,
        csv_runtime: CSVSessionRuntime,
        manifest_store: ManifestStore,
        notebook_orchestrator: NotebookOrchestrator,
        storage_dir: str | Path,
    ) -> None:
        self._store = store
        self._csv_runtime = csv_runtime
        self._manifest_store = manifest_store
        self._notebook_orchestrator = notebook_orchestrator
        self._storage_dir = Path(storage_dir)

    def ingest_files(
        self,
        *,
        session_id: str,
        files: Iterable[TabularUploadFile],
        ttl_seconds: int | None = None,
        max_bytes_per_file: int | None = None,
        preprocessing_options: TabularPreprocessingOptions | None = None,
    ) -> TabularUploadResult:
        state = self._store.load_session(session_id)
        if state is None:
            raise TabularUploadError("Session not found")

        upload_files = list(files)
        if not upload_files:
            raise TabularUploadError("At least one file is required")
        source_ref_id = self._dataset_fingerprint(upload_files)

        parsed = self._parse_files(
            upload_files,
            session_id=session_id,
            max_bytes_per_file=max_bytes_per_file,
            preprocessing_options=preprocessing_options,
        )
        tables = {item.table_name: item.df for item in parsed}
        runtime_registered = False
        try:
            csv_info = self._csv_runtime.register_dataframes(
                session_id=session_id,
                tables=tables,  # type: ignore[arg-type]
                ttl_seconds=ttl_seconds,
            )
            runtime_registered = True

            uploaded_files = self._persist_sources(
                session_id=session_id,
                parsed=parsed,
            )
        except Exception:
            if runtime_registered:
                with suppress(Exception):
                    self._csv_runtime.unregister_tables(session_id, tables.keys())
            raise

        dataset_name = self._dataset_label([item.file_name for item in uploaded_files])
        self._save_legacy_primary_dataframe(session_id, parsed)
        self._store.set_dataset_name(session_id, dataset_name)
        self._store.bind_csv_source(
            session_id,
            filename=dataset_name,
            source_ref_id=source_ref_id,
        )
        self._store.set_csv_runtime_state(
            session_id,
            csv_loaded=True,
            csv_session_id=csv_info.session_id,
            csv_table_names=list(csv_info.table_names),
            csv_expires_at=csv_info.expires_at,
        )

        return TabularUploadResult(
            session_id=session_id,
            csv_session_id=csv_info.session_id,
            table_names=list(csv_info.table_names),
            files=uploaded_files,
            expires_at=csv_info.expires_at,
            total_rows=sum(item.rows for item in uploaded_files),
            total_columns=sum(item.columns for item in uploaded_files),
            dataset_name=dataset_name,
        )

    def _save_legacy_primary_dataframe(
        self,
        session_id: str,
        parsed: list[_ParsedTabularFile],
    ) -> None:
        """Keep single-dataframe consumers working while DuckDB is the multi-table source."""
        first_df = parsed[0].df
        if isinstance(first_df, pd.DataFrame):
            self._store.save_dataframe(session_id, first_df)

    def _parse_files(
        self,
        files: list[TabularUploadFile],
        *,
        session_id: str,
        max_bytes_per_file: int | None,
        preprocessing_options: TabularPreprocessingOptions | None,
    ) -> list[_ParsedTabularFile]:
        existing_names = self._existing_table_names(session_id)
        parsed: list[_ParsedTabularFile] = []
        for upload in files:
            clean_name = self._safe_file_name(upload.file_name)
            if max_bytes_per_file is not None and len(upload.content) > max_bytes_per_file:
                raise TabularUploadError(f"File '{clean_name}' exceeds size limit")
            file_format = self._format_for_file(clean_name)
            preprocessed = self._read_dataframe(
                upload.content,
                file_format=file_format,
                preprocessing_options=preprocessing_options,
            )
            df = preprocessed.dataframe
            if df.empty and len(df.columns) == 0:
                raise TabularUploadError(f"File '{clean_name}' has no readable rows or columns")

            table_name = CSVSessionRuntime.unique_table_name(
                clean_name,
                [*existing_names, *(item.table_name for item in parsed)],
            )
            parsed.append(
                _ParsedTabularFile(
                    file_name=clean_name,
                    file_format=file_format,
                    table_name=table_name,
                    df=df,
                    preprocessing=preprocessed.summary,
                )
            )
        return parsed

    def _persist_sources(
        self,
        *,
        session_id: str,
        parsed: list[_ParsedTabularFile],
    ) -> list[UploadedTabularFile]:
        manifest = self._manifest_store.load(session_id)
        original_manifest = SessionManifest.from_dict(manifest.to_dict())
        existing_aliases = [source.alias for source in manifest.sources]
        source_dir = self._source_dir(session_id)
        source_dir.mkdir(parents=True, exist_ok=True)

        uploaded: list[UploadedTabularFile] = []
        notebook_edits: list[NotebookEdit] = []
        written_paths: list[Path] = []
        for item in parsed:
            if not isinstance(item.df, pd.DataFrame):
                raise TabularUploadError(f"File '{item.file_name}' did not parse as a DataFrame")

            parquet_rel = f"sources/{item.table_name}.parquet"
            parquet_abs = self._session_dir(session_id) / parquet_rel
            item.df.to_parquet(parquet_abs, engine="pyarrow")
            written_paths.append(parquet_abs)

            alias = make_source_alias(item.file_name, "csv", existing_aliases)
            existing_aliases.append(alias)
            variable_name = alias_to_variable_name(alias)
            preprocessing_summary = (
                item.preprocessing.model_dump()
                if hasattr(item.preprocessing, "model_dump")
                else item.preprocessing.dict()
            )
            source = SessionSource(
                alias=alias,
                source_type="csv",
                display_name=item.file_name,
                variable_name=variable_name,
                file_name=item.file_name,
                parquet_path=parquet_rel,
                csv_session_id=session_id,
                csv_table_names=[item.table_name],
                schema_hint={str(col): str(item.df[col].dtype) for col in list(item.df.columns)[:30]},
                preprocessing_summary=preprocessing_summary,
                row_count=len(item.df),
                column_count=len(item.df.columns),
            )
            manifest.add_source(source)
            load_code = f'{variable_name} = pd.read_parquet("{parquet_rel}")'
            notebook_edits.append(
                NotebookEdit(
                    op=CellOp.INSERT,
                    cell=build_source_binding_cell(
                        alias=alias,
                        variable_name=variable_name,
                        source_type="csv",
                        display_name=item.file_name,
                        load_code=load_code,
                    ),
                )
            )
            uploaded.append(
                UploadedTabularFile(
                    file_name=item.file_name,
                    file_format=item.file_format,
                    table_name=item.table_name,
                    source_alias=alias,
                    variable_name=variable_name,
                    parquet_path=parquet_rel,
                    rows=len(item.df),
                    columns=len(item.df.columns),
                    preprocessing=item.preprocessing,
                )
            )

        try:
            self._manifest_store.save(session_id, manifest)
            results = self._notebook_orchestrator.apply_batch(session_id, notebook_edits)
            failed = next((result for result in results if not result.ok), None)
            if failed is not None:
                raise TabularUploadError(f"Failed to update notebook source bindings: {failed.error}")
            return uploaded
        except Exception:
            with suppress(Exception):
                self._manifest_store.save(session_id, original_manifest)
            for path in written_paths:
                with suppress(FileNotFoundError):
                    path.unlink()
            raise

    def _existing_table_names(self, session_id: str) -> list[str]:
        try:
            info = self._csv_runtime.get_session_info(session_id)
        except Exception:
            return []
        return list(info.table_names)

    @staticmethod
    def _safe_file_name(file_name: str) -> str:
        clean = Path(str(file_name or "")).name.strip()
        if not clean:
            raise TabularUploadError("File name must not be empty")
        return clean

    @staticmethod
    def _format_for_file(file_name: str) -> TabularFileFormat:
        suffix = Path(file_name).suffix.lower()
        file_format = _SUPPORTED_EXTENSIONS.get(suffix)
        if file_format is None:
            supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
            raise TabularUploadError(f"Unsupported file type '{suffix}'. Supported: {supported}")
        return file_format

    @staticmethod
    def _read_dataframe(
        content: bytes,
        *,
        file_format: TabularFileFormat,
        preprocessing_options: TabularPreprocessingOptions | None,
    ) -> PreprocessedTabularData:
        try:
            return read_tabular_dataframe(
                content,
                file_format=file_format,
                options=preprocessing_options,
            )
        except ValueError as exc:
            raise TabularUploadError(str(exc)) from exc

    def _session_dir(self, session_id: str) -> Path:
        return self._storage_dir / "sessions" / session_id

    def _source_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "sources"

    @staticmethod
    def _dataset_label(file_names: list[str]) -> str:
        if not file_names:
            return ""
        if len(file_names) <= 3:
            return ", ".join(file_names)
        return f"{', '.join(file_names[:3])} +{len(file_names) - 3} more"

    @staticmethod
    def _dataset_fingerprint(files: list[TabularUploadFile]) -> str:
        digests = [hashlib.sha256(upload.content).hexdigest() for upload in files]
        if len(digests) == 1:
            return f"sha256:{digests[0]}"
        combined = hashlib.sha256()
        for digest in sorted(digests):
            combined.update(digest.encode("ascii"))
            combined.update(b"\n")
        return f"sha256:{combined.hexdigest()}"
