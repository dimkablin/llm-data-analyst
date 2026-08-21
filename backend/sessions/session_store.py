from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.sessions.session_memory import StructuredSessionMemory

import pandas as pd

from backend.core.json_utils import NumpyEncoder as _NumpyEncoder
from backend.data_access.data_catalog import DataCatalogSnapshot
from backend.tools.sandbox_manager import SandboxManager

_DF_CACHE_MAX_SIZE = 20
_ARTIFACT_INLINE_BYTES = 256_000
_ARTIFACT_PREVIEW_ROWS = 100
_EXECUTION_ARTIFACT_BLOB_KIND = "execution_artifact"

logger = logging.getLogger(__name__)


def _artifact_id(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    return str(value.get("execution_artifact_id") or value.get("id") or "").strip()


def _message_artifact_ids(messages: list[dict[str, Any]]) -> set[str]:
    return {
        artifact_id
        for message in messages
        for artifact_id in (_artifact_id(artifact) for artifact in message.get("artifacts") or [])
        if artifact_id
    }


def _compact_artifact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        encoded_size = len(json.dumps(payload, cls=_NumpyEncoder).encode("utf-8"))
    except (TypeError, ValueError):
        return payload
    if encoded_size <= _ARTIFACT_INLINE_BYTES:
        return payload
    compact = copy.deepcopy(payload)
    outer = compact.get("data")
    split = outer.get("data") if isinstance(outer, dict) else None
    rows = split.get("data") if isinstance(split, dict) else None
    if not isinstance(rows, list):
        return payload
    split["data"] = rows[:_ARTIFACT_PREVIEW_ROWS]
    index = split.get("index")
    if isinstance(index, list):
        split["index"] = index[:_ARTIFACT_PREVIEW_ROWS]
    compact.setdefault("meta", {})["display_preview"] = {
        "shown_rows": min(len(rows), _ARTIFACT_PREVIEW_ROWS),
        "total_rows": len(rows),
    }
    return compact


def _artifact_blob_id(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    execution = payload.get("execution")
    storage = execution.get("storage") if isinstance(execution, dict) else None
    if not isinstance(storage, dict) or storage.get("kind") != "blob":
        return ""
    return str(storage.get("blob_id") or "").strip()


class SessionBusyError(RuntimeError):
    """Raised when another analytical request owns the session runtime."""


class SessionQueryLease:
    """Idempotent lease for one active query in a session."""

    def __init__(self, release: Callable[[], None]) -> None:
        self._release = release
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._release()

    def __enter__(self) -> SessionQueryLease:
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


@dataclass
class SessionState:
    session_id: str
    created_at: str
    last_access: str
    chat_history: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    df_path: str | None = None
    dataset_name: str | None = None
    source_type: str | None = None
    source_ref_id: str | None = None
    source_label: str | None = None
    source_mode: str | None = None
    selected_skill_ids: list[str] | None = None
    csv_loaded: bool = False
    csv_session_id: str | None = None
    csv_table_names: list[str] | None = None
    csv_expires_at: int | None = None
    session_memory: str = ""
    artifact_index_json: str = ""
    key_findings: list[str] = field(default_factory=list)
    session_turn_count: int = 0
    context_summary: str = ""
    compacted_message_count: int = 0
    context_usage: dict[str, Any] | None = None


def _state_from_payload(raw: dict[str, Any]) -> SessionState:
    return SessionState(
        session_id=raw["session_id"],
        created_at=raw["created_at"],
        last_access=raw.get("last_access", raw["created_at"]),
        chat_history=raw.get("chat_history", []),
        artifacts=raw.get("artifacts", []),
        df_path=raw.get("df_path"),
        dataset_name=raw.get("dataset_name"),
        source_type=raw.get("source_type"),
        source_ref_id=raw.get("source_ref_id"),
        source_label=raw.get("source_label"),
        source_mode=raw.get("source_mode"),
        selected_skill_ids=list(raw.get("selected_skill_ids", []) or []),
        csv_loaded=bool(raw.get("csv_loaded", False)),
        csv_session_id=raw.get("csv_session_id"),
        csv_table_names=list(raw.get("csv_table_names") or []),
        csv_expires_at=raw.get("csv_expires_at"),
        session_memory=str(raw.get("session_memory", "")),
        artifact_index_json=str(raw.get("artifact_index_json", "")),
        key_findings=list(raw.get("key_findings", []) or []),
        session_turn_count=int(raw.get("session_turn_count", 0) or 0),
        context_summary=str(raw.get("context_summary", "")),
        compacted_message_count=max(0, int(raw.get("compacted_message_count", 0) or 0)),
        context_usage=raw.get("context_usage") if isinstance(raw.get("context_usage"), dict) else None,
    )


def _state_payload(state: SessionState) -> dict[str, Any]:
    return {
        "session_id": state.session_id,
        "created_at": state.created_at,
        "last_access": state.last_access,
        "chat_history": state.chat_history,
        "artifacts": state.artifacts,
        "df_path": state.df_path,
        "dataset_name": state.dataset_name,
        "source_type": state.source_type,
        "source_ref_id": state.source_ref_id,
        "source_label": state.source_label,
        "source_mode": state.source_mode,
        "selected_skill_ids": list(state.selected_skill_ids or []),
        "csv_loaded": bool(state.csv_loaded),
        "csv_session_id": state.csv_session_id,
        "csv_table_names": list(state.csv_table_names or []),
        "csv_expires_at": state.csv_expires_at,
        "session_memory": state.session_memory or "",
        "artifact_index_json": state.artifact_index_json or "",
        "key_findings": list(state.key_findings or []),
        "session_turn_count": int(state.session_turn_count or 0),
        "context_summary": state.context_summary or "",
        "compacted_message_count": max(0, int(state.compacted_message_count or 0)),
        "context_usage": state.context_usage,
    }


class SessionStore:
    def __init__(
        self,
        root_dir: str,
        ttl_days: int,
        *,
        data_catalog_store: Any | None = None,
        artifact_blob_store: Any | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(days=ttl_days)
        self._data_catalog_store = data_catalog_store
        self._artifact_blob_store = artifact_blob_store
        self._df_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._locks_guard = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}
        self._query_locks: dict[str, threading.Lock] = {}
        self._cleanup_expired()

    @property
    def metadata_store(self) -> Any | None:
        return self._data_catalog_store

    def _get_session_lock(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = threading.Lock()
            return self._session_locks[session_id]

    def acquire_query_lease(self, session_id: str) -> SessionQueryLease:
        """Acquire a non-blocking single-writer lease for one session."""
        clean_id = str(session_id or "").strip()
        if not clean_id:
            raise ValueError("session_id is required")
        with self._locks_guard:
            lock = self._query_locks.setdefault(clean_id, threading.Lock())
        if not lock.acquire(blocking=False):
            raise SessionBusyError("A request is already running in this session.")
        return SessionQueryLease(lock.release)

    def _session_dir(self, session_id: str) -> Path:
        return self.root_dir / session_id

    def _state_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "state.json"

    def _data_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "data.parquet"

    def _now_iso(self) -> str:
        return datetime.now(UTC).isoformat()

    def _cleanup_expired(self) -> None:
        now = datetime.now(UTC)
        for session_dir in self.root_dir.iterdir():
            if not session_dir.is_dir():
                continue
            state_path = session_dir / "state.json"
            if not state_path.exists():
                continue
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                last_access = datetime.fromisoformat(state.get("last_access"))
                if last_access.tzinfo is None:
                    last_access = last_access.replace(tzinfo=UTC)
            except Exception:
                continue
            if now - last_access > self.ttl:
                if self._data_catalog_store is not None:
                    self._data_catalog_store.delete_data_profile(session_dir.name)
                shutil.rmtree(session_dir, ignore_errors=True)
                self._df_cache.pop(session_dir.name, None)

    def create_session(self, session_id: str | None = None) -> SessionState:
        session_id = str(session_id or os.urandom(16).hex())
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        state = SessionState(
            session_id=session_id,
            created_at=self._now_iso(),
            last_access=self._now_iso(),
            chat_history=[],
            artifacts=[],
            df_path=None,
            dataset_name=None,
            source_type=None,
            source_ref_id=None,
            source_label=None,
            source_mode=None,
            selected_skill_ids=[],
            csv_loaded=False,
            csv_session_id=None,
            csv_table_names=[],
            csv_expires_at=None,
            session_memory="",
            context_usage=None,
        )
        self._save_state(state)
        return state

    def _load_state(self, session_id: str) -> SessionState | None:
        """Read session state from disk without updating last_access."""
        state_path = self._state_path(session_id)
        if not state_path.exists():
            return None
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        return _state_from_payload(raw)

    def load_session(self, session_id: str) -> SessionState | None:
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return None
            state.last_access = self._now_iso()
            self._save_state(state)
            return state

    def _save_state(self, state: SessionState) -> None:
        payload = _state_payload(state)
        # Atomic write: flush to a temp file in the same directory, then rename.
        # This prevents a truncated/empty state file if the process is interrupted
        # mid-write (crash or SIGKILL).
        target = self._state_path(state.session_id)
        content = json.dumps(payload, ensure_ascii=False, cls=_NumpyEncoder)
        fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            os.replace(tmp_path, target)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def save_data_catalog(self, session_id: str, snapshot: DataCatalogSnapshot) -> None:
        if self._data_catalog_store is None:
            raise RuntimeError("PostgreSQL metadata store is required for data profiling")
        self._data_catalog_store.save_data_profile(session_id, snapshot)

    def load_data_catalog(self, session_id: str) -> DataCatalogSnapshot | None:
        if self._data_catalog_store is None:
            return None
        return self._data_catalog_store.load_data_profile(session_id)

    def save_dataframe(self, session_id: str, df: pd.DataFrame) -> None:
        data_path = self._data_path(session_id)
        df.to_parquet(data_path, engine="pyarrow")
        self._put_df_cache(session_id, df)
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return
            state.df_path = str(data_path)
            self._save_state(state)

    def clear_dataframe(self, session_id: str) -> None:
        data_path = self._data_path(session_id)
        try:
            data_path.unlink()
        except FileNotFoundError:
            pass
        self._df_cache.pop(session_id, None)
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return
            state.df_path = None
            state.dataset_name = None
            state.last_access = self._now_iso()
            self._save_state(state)

    def set_dataset_name(self, session_id: str, dataset_name: str | None) -> None:
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return
            clean = str(dataset_name or "").strip()
            state.dataset_name = clean or None
            state.last_access = self._now_iso()
            self._save_state(state)

    def set_source(
        self,
        session_id: str,
        *,
        source_type: str | None,
        source_ref_id: str | None,
        source_label: str | None,
        source_mode: str | None = None,
    ) -> None:
        source_changed = False
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return
            previous = (
                state.source_type,
                state.source_ref_id,
                state.source_mode,
            )
            state.source_type = str(source_type or "").strip() or None
            state.source_ref_id = str(source_ref_id or "").strip() or None
            state.source_label = str(source_label or "").strip() or None
            state.source_mode = str(source_mode or "").strip() or None
            source_changed = previous != (
                state.source_type,
                state.source_ref_id,
                state.source_mode,
            )
            state.last_access = self._now_iso()
            self._save_state(state)
        if source_changed:
            SandboxManager.get_instance().remove(session_id)

    def bind_db_connection_source(
        self,
        session_id: str,
        *,
        connection_id: str,
        label: str,
        source_mode: str | None = None,
    ) -> None:
        self.set_source(
            session_id,
            source_type="db_connection",
            source_ref_id=connection_id,
            source_label=label,
            source_mode=source_mode,
        )

    def bind_csv_source(
        self,
        session_id: str,
        *,
        filename: str | None,
        source_ref_id: str | None = None,
        source_mode: str | None = None,
    ) -> None:
        clean_name = str(filename or "").strip() or None
        clean_ref_id = str(source_ref_id or "").strip() or clean_name
        self.set_source(
            session_id,
            source_type="csv",
            source_ref_id=clean_ref_id,
            source_label=clean_name,
            source_mode=source_mode,
        )

    def set_selected_skill_ids(
        self,
        session_id: str,
        selected_skill_ids: list[str] | None,
    ) -> None:
        normalized = [
            str(skill_id).strip() for skill_id in (selected_skill_ids or []) if str(skill_id).strip()
        ]
        deduped = list(dict.fromkeys(normalized))
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return
            state.selected_skill_ids = deduped
            state.last_access = self._now_iso()
            self._save_state(state)

    def set_context_usage(self, session_id: str, snapshot: dict[str, Any] | None) -> None:
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return
            state.context_usage = dict(snapshot) if isinstance(snapshot, dict) else None
            state.last_access = self._now_iso()
            self._save_state(state)

    def get_dataframe(self, session_id: str) -> pd.DataFrame | None:
        if session_id in self._df_cache:
            self._df_cache.move_to_end(session_id)
            return self._df_cache[session_id]
        state = self.load_session(session_id)
        if state is None or not state.df_path:
            return None
        path = Path(state.df_path)
        if path.suffix == ".pkl":
            df = pd.read_pickle(path)
        else:
            df = pd.read_parquet(path, engine="pyarrow")
        self._put_df_cache(session_id, df)
        return df

    def _put_df_cache(self, session_id: str, df: pd.DataFrame) -> None:
        self._df_cache[session_id] = df
        self._df_cache.move_to_end(session_id)
        while len(self._df_cache) > _DF_CACHE_MAX_SIZE:
            self._df_cache.popitem(last=False)

    def add_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        artifacts: list[dict[str, Any]] | None = None,
        reasoning: str | None = None,
        reasoning_steps: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        anomaly_check: dict[str, Any] | None = None,
    ) -> None:
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return
            payload: dict[str, Any] = {
                "id": str(uuid.uuid4()),
                "role": role,
                "content": content,
                "timestamp": self._now_iso(),
            }
            if artifacts:
                payload["artifacts"] = [_compact_artifact_payload(item) for item in artifacts]
            if reasoning:
                payload["reasoning"] = reasoning
            if reasoning_steps:
                payload["reasoning_steps"] = reasoning_steps
            if tools:
                payload["tools"] = tools
            if anomaly_check:
                payload["anomaly_check"] = anomaly_check
            state.chat_history.append(payload)
            state.last_access = self._now_iso()
            self._save_state(state)

    def add_artifacts(
        self,
        session_id: str,
        artifacts: list,
        *,
        user_id: int | None = None,
    ) -> None:
        from backend.artifacts.bridge import execution_data_is_complete, execution_to_api_payload

        serialized = [execution_to_api_payload(artifact) for artifact in artifacts]
        blob_ids: list[str] = []
        if self._artifact_blob_store is not None and user_id is not None:
            from backend.auth.blob_store import BlobWrite

            writes: list[BlobWrite] = []
            indexes: list[int] = []
            for index, (artifact, payload) in enumerate(zip(artifacts, serialized, strict=True)):
                data = getattr(artifact, "data", None)
                if not isinstance(data, pd.DataFrame) or not execution_data_is_complete(artifact):
                    continue
                if len(json.dumps(payload, cls=_NumpyEncoder).encode("utf-8")) <= _ARTIFACT_INLINE_BYTES:
                    continue
                try:
                    buffer = BytesIO()
                    data.to_parquet(buffer, engine="pyarrow", index=True)
                except Exception:
                    logger.warning(
                        "Large artifact could not be serialized to parquet: %s",
                        getattr(artifact, "name", ""),
                        exc_info=True,
                    )
                    continue
                writes.append(
                    BlobWrite(
                        logical_name=f"{artifact.id}.parquet",
                        media_type="application/vnd.apache.parquet",
                        content=buffer.getvalue(),
                        metadata={"artifact_id": str(artifact.id)},
                    )
                )
                indexes.append(index)
            blob_ids = (
                self._artifact_blob_store.put_many(
                    user_id=user_id,
                    session_id=session_id,
                    kind=_EXECUTION_ARTIFACT_BLOB_KIND,
                    items=writes,
                )
                if writes
                else []
            )
            for index, blob_id in zip(indexes, blob_ids, strict=True):
                serialized[index] = _compact_artifact_payload(serialized[index])
                serialized[index]["execution"]["storage"] = {
                    "kind": "blob",
                    "blob_id": blob_id,
                    "media_type": "application/vnd.apache.parquet",
                }
        try:
            self.add_serialized_artifacts(session_id, serialized)
        except Exception:
            self._delete_artifact_blobs(session_id, blob_ids)
            raise

    def get_serialized_artifact(
        self,
        session_id: str,
        artifact_id: str,
    ) -> dict[str, Any] | None:
        clean_id = str(artifact_id or "").strip()
        if not clean_id:
            return None
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return None
            payload = next(
                (
                    dict(item)
                    for item in reversed(state.artifacts)
                    if isinstance(item, dict)
                    and str(item.get("execution_artifact_id") or item.get("id") or "") == clean_id
                ),
                None,
            )
        if payload is None:
            return None
        execution = payload.get("execution")
        storage = execution.get("storage") if isinstance(execution, dict) else None
        if not isinstance(storage, dict) or storage.get("kind") != "blob":
            return payload
        if self._artifact_blob_store is None:
            return payload
        blob = self._artifact_blob_store.get_for_session(
            session_id=session_id,
            blob_id=str(storage.get("blob_id") or ""),
            kind=_EXECUTION_ARTIFACT_BLOB_KIND,
        )
        if blob is None:
            return payload
        dataframe = pd.read_parquet(BytesIO(blob.content), engine="pyarrow")
        hydrated = copy.deepcopy(payload)
        hydrated["data"] = {
            "format": "split",
            "data": json.loads(dataframe.to_json(orient="split", date_format="iso")),
        }
        hydrated["execution"]["storage"] = {"kind": "inline"}
        return hydrated

    def add_serialized_artifacts(
        self,
        session_id: str,
        artifacts: list[dict[str, Any]],
        *,
        replace_producer_tool: str | None = None,
    ) -> None:
        obsolete_blob_ids: set[str] = set()
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return
            if replace_producer_tool:
                obsolete_blob_ids.update(
                    blob_id
                    for item in state.artifacts
                    if isinstance(item, dict)
                    and item.get("meta", {}).get("producer_tool") == replace_producer_tool
                    and (blob_id := _artifact_blob_id(item))
                )
                state.artifacts = [
                    item
                    for item in state.artifacts
                    if not (
                        isinstance(item, dict)
                        and item.get("meta", {}).get("producer_tool") == replace_producer_tool
                    )
                ]
            existing_ids = {
                str(item.get("id")) for item in state.artifacts if isinstance(item, dict) and item.get("id")
            }
            for artifact in artifacts:
                artifact_id = str(artifact.get("id") or "").strip()
                if artifact_id and artifact_id in existing_ids:
                    replacement_blob_id = _artifact_blob_id(artifact)
                    obsolete_blob_ids.update(
                        blob_id
                        for item in state.artifacts
                        if str(item.get("id") or "") == artifact_id
                        and (blob_id := _artifact_blob_id(item))
                        and blob_id != replacement_blob_id
                    )
                    state.artifacts = [
                        artifact if str(item.get("id") or "") == artifact_id else item
                        for item in state.artifacts
                    ]
                else:
                    state.artifacts.append(artifact)
                    if artifact_id:
                        existing_ids.add(artifact_id)
            state.last_access = self._now_iso()
            self._save_state(state)
        self._delete_artifact_blobs(session_id, list(obsolete_blob_ids))

    def _delete_artifact_blobs(self, session_id: str, blob_ids: list[str]) -> None:
        delete = getattr(self._artifact_blob_store, "delete_ids_for_session", None)
        if callable(delete) and blob_ids:
            delete(
                session_id=session_id,
                blob_ids=blob_ids,
                kind=_EXECUTION_ARTIFACT_BLOB_KIND,
            )

    def set_csv_runtime_state(
        self,
        session_id: str,
        *,
        csv_loaded: bool,
        csv_session_id: str | None,
        csv_table_names: list[str] | None = None,
        csv_expires_at: int | None = None,
    ) -> None:
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return
            state.csv_loaded = bool(csv_loaded)
            state.csv_session_id = str(csv_session_id or "").strip() or None
            state.csv_table_names = list(csv_table_names or [])
            state.csv_expires_at = csv_expires_at
            state.last_access = self._now_iso()
            self._save_state(state)

    def set_session_memory(self, session_id: str, content: str) -> None:
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return
            state.session_memory = content.strip()
            state.last_access = self._now_iso()
            self._save_state(state)

    def append_session_memory(self, session_id: str, note: str) -> None:
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return
            existing = state.session_memory or ""
            state.session_memory = (existing + "\n- " + note.strip()).lstrip()
            state.last_access = self._now_iso()
            self._save_state(state)

    def get_structured_memory(self, session_id: str) -> StructuredSessionMemory:
        """Load StructuredSessionMemory from session state. Returns empty if session not found."""
        import json as _json

        from backend.sessions.session_memory import SessionArtifactRef, StructuredSessionMemory

        state = self._load_state(session_id)
        if state is None:
            return StructuredSessionMemory()

        artifact_index: list[SessionArtifactRef] = []
        if state.artifact_index_json:
            try:
                raw_list = _json.loads(state.artifact_index_json)
                for item in raw_list:
                    if isinstance(item, dict):
                        try:
                            artifact_index.append(SessionArtifactRef(**item))
                        except Exception:
                            pass  # skip malformed entries
            except Exception:
                pass  # malformed JSON — start fresh

        return StructuredSessionMemory(
            notes=state.session_memory or "",
            artifact_index=artifact_index,
            key_findings=list(state.key_findings or []),
            turn_count=int(state.session_turn_count or 0),
            context_summary=state.context_summary or "",
            compacted_message_count=max(0, int(state.compacted_message_count or 0)),
        )

    def set_structured_memory(self, session_id: str, memory: StructuredSessionMemory) -> None:
        """Persist StructuredSessionMemory to session state."""
        import json as _json

        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return
            state.session_memory = memory.notes.strip()
            # Serialize artifact_index to JSON
            refs_as_dicts = [
                {
                    "id": ref.id,
                    "name": ref.name,
                    "type": ref.type,
                    "turn_index": ref.turn_index,
                    "schema": ref.schema,
                    "row_count": ref.row_count,
                    "summary": ref.summary,
                    "producer_tool": ref.producer_tool,
                    "parent_ids": list(ref.parent_ids),
                }
                for ref in memory.artifact_index
            ]
            state.artifact_index_json = _json.dumps(refs_as_dicts, ensure_ascii=False)
            state.key_findings = list(memory.key_findings)
            state.session_turn_count = int(memory.turn_count)
            state.context_summary = memory.context_summary.strip()
            state.compacted_message_count = max(0, int(memory.compacted_message_count or 0))
            state.last_access = self._now_iso()
            self._save_state(state)

    def clear_csv_runtime_state(self, session_id: str) -> None:
        self.set_csv_runtime_state(
            session_id,
            csv_loaded=False,
            csv_session_id=None,
            csv_table_names=[],
            csv_expires_at=None,
        )

    def delete_messages_from_id(self, session_id: str, message_id: str) -> int:
        """Delete the message with *message_id* and all messages after it.

        Returns the number of messages removed. If the ID is not found returns 0.
        """
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return 0
            history = state.chat_history
            cut_index: int | None = None
            for i, msg in enumerate(history):
                if msg.get("id") == message_id:
                    cut_index = i
                    break
            if cut_index is None:
                return 0
            removed = len(history) - cut_index
            removed_ids = _message_artifact_ids(history[cut_index:])
            retained_ids = _message_artifact_ids(history[:cut_index])
            discarded_ids = removed_ids - retained_ids
            compacted_count = max(0, int(state.compacted_message_count or 0))
            state.chat_history = history[:cut_index]
            if discarded_ids:
                discarded_blob_ids = [
                    blob_id
                    for artifact in state.artifacts
                    if _artifact_id(artifact) in discarded_ids and (blob_id := _artifact_blob_id(artifact))
                ]
                state.artifacts = [
                    artifact for artifact in state.artifacts if _artifact_id(artifact) not in discarded_ids
                ]
                try:
                    artifact_index = json.loads(state.artifact_index_json or "[]")
                except (TypeError, ValueError):
                    artifact_index = []
                state.artifact_index_json = json.dumps(
                    [artifact for artifact in artifact_index if _artifact_id(artifact) not in discarded_ids],
                    ensure_ascii=False,
                )
            if cut_index < compacted_count:
                state.context_summary = ""
                state.compacted_message_count = 0
            state.last_access = self._now_iso()
            self._save_state(state)
        if discarded_ids:
            self._delete_artifact_blobs(session_id, discarded_blob_ids)
        SandboxManager.get_instance().remove(session_id)
        return removed

    def delete_session(self, session_id: str) -> None:
        if self._data_catalog_store is not None:
            self._data_catalog_store.delete_data_profile(session_id)
        session_dir = self._session_dir(session_id)
        shutil.rmtree(session_dir, ignore_errors=True)
        self._df_cache.pop(session_id, None)
        SandboxManager.get_instance().remove(session_id)
