from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from backend.core.json_utils import NumpyEncoder as _NumpyEncoder

_DF_CACHE_MAX_SIZE = 20


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


class SessionStore:
    def __init__(self, root_dir: str, ttl_days: int) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(days=ttl_days)
        self._df_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._locks_guard = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}
        self._cleanup_expired()

    def _get_session_lock(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = threading.Lock()
            return self._session_locks[session_id]

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
                shutil.rmtree(session_dir, ignore_errors=True)
                self._df_cache.pop(session_dir.name, None)

    def create_session(self) -> SessionState:
        session_id = os.urandom(16).hex()
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
        )
        self._save_state(state)
        return state

    def _load_state(self, session_id: str) -> SessionState | None:
        """Read session state from disk without updating last_access."""
        state_path = self._state_path(session_id)
        if not state_path.exists():
            return None
        raw = json.loads(state_path.read_text(encoding="utf-8"))
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
        )

    def load_session(self, session_id: str) -> SessionState | None:
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return None
            state.last_access = self._now_iso()
            self._save_state(state)
            return state

    def _save_state(self, state: SessionState) -> None:
        payload = {
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
        }
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
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return
            state.source_type = str(source_type or "").strip() or None
            state.source_ref_id = str(source_ref_id or "").strip() or None
            state.source_label = str(source_label or "").strip() or None
            state.source_mode = str(source_mode or "").strip() or None
            state.last_access = self._now_iso()
            self._save_state(state)

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
        source_mode: str | None = None,
    ) -> None:
        clean_name = str(filename or "").strip() or None
        self.set_source(
            session_id,
            source_type="csv",
            source_ref_id=clean_name,
            source_label=clean_name,
            source_mode=source_mode,
        )

    def set_selected_skill_ids(
        self,
        session_id: str,
        selected_skill_ids: list[str] | None,
    ) -> None:
        normalized = [
            str(skill_id).strip()
            for skill_id in (selected_skill_ids or [])
            if str(skill_id).strip()
        ]
        deduped = list(dict.fromkeys(normalized))
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return
            state.selected_skill_ids = deduped
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
                payload["artifacts"] = artifacts
            if reasoning:
                payload["reasoning"] = reasoning
            if reasoning_steps:
                payload["reasoning_steps"] = reasoning_steps
            if tools:
                payload["tools"] = tools
            state.chat_history.append(payload)
            state.last_access = self._now_iso()
            self._save_state(state)

    def add_artifacts(self, session_id: str, artifacts: list) -> None:
        from backend.artifacts.bridge import execution_to_api_payload
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return
            serialized = [execution_to_api_payload(a) for a in artifacts]
            state.artifacts.extend(serialized)
            state.last_access = self._now_iso()
            self._save_state(state)


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

    def get_structured_memory(self, session_id: str) -> "StructuredSessionMemory":
        """Load StructuredSessionMemory from session state. Returns empty if session not found."""
        from backend.sessions.session_memory import StructuredSessionMemory, SessionArtifactRef
        import json as _json
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
        )

    def set_structured_memory(self, session_id: str, memory: "StructuredSessionMemory") -> None:
        """Persist StructuredSessionMemory to session state."""
        import json as _json
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return
            state.session_memory = memory.notes.strip()
            # Serialize artifact_index to JSON
            refs_as_dicts = []
            for ref in memory.artifact_index:
                refs_as_dicts.append({
                    "id": ref.id,
                    "name": ref.name,
                    "type": ref.type,
                    "turn_index": ref.turn_index,
                    "schema": ref.schema,
                    "row_count": ref.row_count,
                    "summary": ref.summary,
                })
            state.artifact_index_json = _json.dumps(refs_as_dicts, ensure_ascii=False)
            state.key_findings = list(memory.key_findings)
            state.session_turn_count = int(memory.turn_count)
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
            state.chat_history = history[:cut_index]
            state.last_access = self._now_iso()
            self._save_state(state)
            return removed

    def delete_session(self, session_id: str) -> None:
        session_dir = self._session_dir(session_id)
        shutil.rmtree(session_dir, ignore_errors=True)
        self._df_cache.pop(session_id, None)


