"""Build and persist session data catalogs after upload / DB bind."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any

import pandas as pd

from backend.data_access.data_catalog import (
    build_snapshot_from_csv_runtime,
    build_snapshot_from_dataframe,
    build_snapshot_from_db_helper,
    CatalogProfileOptions,
    catalog_cache_key,
    format_catalog_prompt_block,
    merge_snapshots,
    DataCatalogSnapshot,
)
from backend.data_access.session_catalog_cache import invalidate_candidates_cache
from backend.sessions.session_store import SessionStore
from backend.tools.impl.db_helpers import DBAnalyticsHelper

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _profile_options() -> tuple[CatalogProfileOptions, float]:
    return (
        CatalogProfileOptions(
            max_tables=_env_int("SEMANTIC_PROFILE_MAX_TABLES", 100, minimum=1),
            max_columns_per_table=_env_int("SEMANTIC_PROFILE_MAX_COLUMNS_PER_TABLE", 200, minimum=1),
            sample_rows=_env_int("SEMANTIC_PROFILE_SAMPLE_ROWS", 1000),
            top_values=_env_int("SEMANTIC_PROFILE_TOP_VALUES", 20),
        ),
        float(_env_int("SEMANTIC_PROFILE_TIMEOUT_SEC", 60, minimum=1)),
    )


def _fingerprint_df(df: pd.DataFrame) -> str:
    cols = ",".join(str(c) for c in df.columns[:64])
    payload = f"df|{df.shape}|{cols}"
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def build_catalog_snapshot(
    *,
    df: pd.DataFrame | None,
    csv_runtime: Any | None,
    csv_session_id: str | None,
    csv_source_ref_id: str | None = None,
    csv_loaded: bool,
    db_runtime: Any | None,
) -> DataCatalogSnapshot:
    profile, timeout_sec = _profile_options()
    deadline = time.monotonic() + timeout_sec
    parts: list[DataCatalogSnapshot] = []
    fp_parts: list[str] = []
    errors: list[str] = []
    csv_snapshot_added = False

    if csv_loaded and csv_runtime is not None and str(csv_session_id or "").strip():
        sid = str(csv_session_id).strip()
        csv_fp = _csv_fingerprint(csv_source_ref_id, sid)
        fp_parts.append(csv_fp)
        try:
            csv_snapshot = build_snapshot_from_csv_runtime(
                csv_runtime,
                sid,
                fingerprint=csv_fp,
                options=profile,
                deadline=deadline,
            )
            if csv_snapshot.tables:
                parts.append(csv_snapshot)
                csv_snapshot_added = True
        except Exception as exc:
            logger.warning("CSV catalog build failed: %s", exc)
            errors.append(f"CSV catalog build failed: {exc}")

    if db_runtime is not None:
        conn_id = str(getattr(db_runtime, "connection_id", "") or "")
        fp_parts.append(f"db:{conn_id}")
        try:
            helper = DBAnalyticsHelper(runtime=db_runtime, timeout_sec=15.0)
            parts.append(
                build_snapshot_from_db_helper(
                    helper,
                    fingerprint=f"db:{conn_id}",
                    options=profile,
                    deadline=deadline,
                )
            )
        except Exception as exc:
            logger.warning("DB catalog build failed: %s", exc)
            errors.append(f"DB catalog build failed: {exc}")

    if df is not None and not df.empty and not csv_snapshot_added:
        fp = _fingerprint_df(df)
        fp_parts.append(fp)
        parts.append(
            build_snapshot_from_dataframe(
                df,
                qualified_name="df",
                fingerprint=fp,
                options=profile,
            )
        )

    if not parts:
        return DataCatalogSnapshot(source_fingerprint="|".join(fp_parts), errors=errors)

    merged = merge_snapshots(*parts)
    merged.source_fingerprint = "|".join(fp_parts) or merged.source_fingerprint
    merged.errors = list(dict.fromkeys([*merged.errors, *errors]))
    return merged


def refresh_session_catalog(
    store: SessionStore,
    session_id: str,
    *,
    df: pd.DataFrame | None = None,
    csv_runtime: Any | None = None,
    db_runtime: Any | None = None,
) -> DataCatalogSnapshot:
    """Rebuild catalog from current session sources and persist to disk."""
    state = store.load_session(session_id)
    if state is None:
        return DataCatalogSnapshot()

    if df is None and state.df_path:
        df = store.get_dataframe(session_id)

    snapshot = build_catalog_snapshot(
        df=df,
        csv_runtime=csv_runtime,
        csv_session_id=state.csv_session_id,
        csv_source_ref_id=state.source_ref_id if state.source_type == "csv" else None,
        csv_loaded=bool(state.csv_loaded),
        db_runtime=db_runtime,
    )
    store.save_data_catalog(session_id, snapshot)
    invalidate_candidates_cache(
        catalog_cache_key(
            session_id=session_id,
            source_type=state.source_type,
            source_ref_id=state.source_ref_id,
            csv_session_id=state.csv_session_id,
        )
    )
    return snapshot


def load_catalog_prompt_for_session(
    store: SessionStore,
    session_id: str,
    *,
    session_source: dict[str, Any] | None = None,
) -> str:
    """Load persisted catalog and format prompt block (empty if missing)."""
    snapshot = store.load_data_catalog(session_id)
    if snapshot is None:
        return ""
    block = format_catalog_prompt_block(snapshot)
    if block and session_source is not None:
        session_source["data_catalog_fingerprint"] = snapshot.source_fingerprint
    return block


def _csv_fingerprint(source_ref_id: str | None, csv_session_id: str) -> str:
    ref_id = str(source_ref_id or "").strip()
    if ref_id.startswith("sha256:"):
        return f"csv:{ref_id}"
    return f"csv-session:{csv_session_id}"


def attach_catalog_to_session_source(
    store: SessionStore,
    session_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Add data_catalog_prompt to runtime session_source dict."""
    block = load_catalog_prompt_for_session(store, session_id, session_source=payload)
    if block:
        payload["data_catalog_prompt"] = block
    return payload
