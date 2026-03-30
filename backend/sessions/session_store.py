from __future__ import annotations

import json
import os
import shutil
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


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
        return datetime.now(timezone.utc).isoformat()

    def _cleanup_expired(self) -> None:
        now = datetime.now(timezone.utc)
        for session_dir in self.root_dir.iterdir():
            if not session_dir.is_dir():
                continue
            state_path = session_dir / "state.json"
            if not state_path.exists():
                continue
            try:
                state = json.loads(state_path.read_text())
                last_access = datetime.fromisoformat(state.get("last_access"))
                if last_access.tzinfo is None:
                    last_access = last_access.replace(tzinfo=timezone.utc)
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
        )
        self._save_state(state)
        return state

    def _load_state(self, session_id: str) -> SessionState | None:
        """Read session state from disk without updating last_access."""
        state_path = self._state_path(session_id)
        if not state_path.exists():
            return None
        raw = json.loads(state_path.read_text())
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
        }
        self._state_path(state.session_id).write_text(
            json.dumps(payload, ensure_ascii=False)
        )

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
        normalized = [str(skill_id).strip() for skill_id in (selected_skill_ids or []) if str(skill_id).strip()]
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
    ) -> None:
        with self._get_session_lock(session_id):
            state = self._load_state(session_id)
            if state is None:
                return
            payload: dict[str, Any] = {
                "role": role,
                "content": content,
                "timestamp": self._now_iso(),
            }
            if artifacts:
                payload["artifacts"] = artifacts
            if reasoning:
                payload["reasoning"] = reasoning
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


    def delete_session(self, session_id: str) -> None:
        session_dir = self._session_dir(session_id)
        shutil.rmtree(session_dir, ignore_errors=True)
        self._df_cache.pop(session_id, None)


