from __future__ import annotations

import io
import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


_TABLE_RE = re.compile(r"[^A-Za-z0-9_]+")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DEFAULT_TTL_SEC = int(os.getenv("CSV_SESSION_TTL_SEC", "7200"))
_BASE_DIR = Path(os.getenv("CSV_SESSION_BASE_DIR", ".runtime/csv_sessions")).resolve()


@dataclass
class CSVSessionInfo:
    session_id: str
    db_path: str
    table_names: list[str]
    expires_at: int


class CSVSessionRuntime:
    _lock = threading.Lock()

    def __init__(
        self,
        *,
        base_dir: str | Path | None = None,
        default_ttl_sec: int = _DEFAULT_TTL_SEC,
    ) -> None:
        self.base_dir = Path(base_dir).resolve() if base_dir else _BASE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl_sec = max(60, int(default_ttl_sec))

    @staticmethod
    def sanitize_table_name(file_name: str) -> str:
        raw = Path(str(file_name or "uploaded.csv")).stem
        cleaned = _TABLE_RE.sub("_", raw).strip("_").lower()
        if not cleaned:
            cleaned = "uploaded_csv"
        if cleaned[0].isdigit():
            cleaned = f"t_{cleaned}"
        return cleaned[:120]

    @staticmethod
    def _quote_ident(value: str) -> str:
        if not _IDENTIFIER_RE.match(value):
            raise ValueError(f"Unsafe identifier: {value}")
        return f'"{value}"'

    def _session_dir(self, session_id: str) -> Path:
        clean = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(session_id or "").strip())
        if not clean:
            raise ValueError("session_id is empty")
        return self.base_dir / clean

    def _db_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "session.duckdb"

    def _meta_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "meta.json"

    def _read_meta(self, session_id: str) -> dict[str, Any]:
        path = self._meta_path(session_id)
        if not path.exists():
            return {
                "session_id": session_id,
                "table_names": [],
                "expires_at": 0,
            }
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {
                "session_id": session_id,
                "table_names": [],
                "expires_at": 0,
            }

    def _write_meta(self, session_id: str, meta: dict[str, Any]) -> None:
        path = self._meta_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _touch_session(self, session_id: str, *, ttl_seconds: int | None = None) -> dict[str, Any]:
        meta = self._read_meta(session_id)
        ttl = max(60, int(ttl_seconds or self.default_ttl_sec))
        meta["session_id"] = session_id
        meta["expires_at"] = int(time.time()) + ttl
        self._write_meta(session_id, meta)
        return meta

    def cleanup_expired_sessions(self) -> None:
        now = int(time.time())
        with self._lock:
            if not self.base_dir.exists():
                return
            for child in self.base_dir.iterdir():
                if not child.is_dir():
                    continue
                meta_path = child / "meta.json"
                if not meta_path.exists():
                    continue
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
                expires_at = int(meta.get("expires_at") or 0)
                if expires_at and expires_at < now:
                    shutil.rmtree(child, ignore_errors=True)

    def register_dataframe(
        self,
        *,
        session_id: str,
        table_name: str,
        df: pd.DataFrame,
        ttl_seconds: int | None = None,
    ) -> CSVSessionInfo:
        self.cleanup_expired_sessions()

        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        db_path = self._db_path(session_id)

        safe_table = self.sanitize_table_name(table_name)

        with self._lock:
            con = duckdb.connect(str(db_path))
            try:
                con.register("_uploaded_df", df)
                con.execute(
                    f"CREATE OR REPLACE TABLE {self._quote_ident(safe_table)} AS "
                    "SELECT * FROM _uploaded_df"
                )
            finally:
                try:
                    con.unregister("_uploaded_df")
                except Exception:
                    pass
                con.close()

            meta = self._touch_session(session_id, ttl_seconds=ttl_seconds)
            tables = list(meta.get("table_names") or [])
            if safe_table not in tables:
                tables.append(safe_table)
            meta["table_names"] = sorted(set(tables))
            self._write_meta(session_id, meta)

        return CSVSessionInfo(
            session_id=session_id,
            db_path=str(db_path),
            table_names=list(meta["table_names"]),
            expires_at=int(meta["expires_at"]),
        )

    def register_csv_bytes(
        self,
        *,
        session_id: str,
        file_name: str,
        csv_bytes: bytes,
        ttl_seconds: int | None = None,
        pandas_read_csv_kwargs: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, CSVSessionInfo]:
        df = pd.read_csv(io.BytesIO(csv_bytes), **dict(pandas_read_csv_kwargs or {}))
        info = self.register_dataframe(
            session_id=session_id,
            table_name=file_name,
            df=df,
            ttl_seconds=ttl_seconds,
        )
        return df, info

    def get_session_info(self, session_id: str) -> CSVSessionInfo:
        self.cleanup_expired_sessions()
        meta = self._touch_session(session_id)
        return CSVSessionInfo(
            session_id=session_id,
            db_path=str(self._db_path(session_id)),
            table_names=list(meta.get("table_names") or []),
            expires_at=int(meta.get("expires_at") or 0),
        )

    def list_tables(self, session_id: str) -> list[dict[str, Any]]:
        self.cleanup_expired_sessions()
        self._touch_session(session_id)
        con = duckdb.connect(str(self._db_path(session_id)), read_only=True)
        try:
            rows = con.execute(
                """
                select
                    table_schema,
                    table_name,
                    table_type
                from information_schema.tables
                where table_schema = 'main'
                order by table_name
                """
            ).fetchdf()
        finally:
            con.close()

        result: list[dict[str, Any]] = []
        for row in rows.to_dict(orient="records"):
            table_name = str(row["table_name"])
            result.append(
                {
                    "schema": "main",
                    "table_name": table_name,
                    "table_type": str(row["table_type"]).lower(),
                    "qualified_name": table_name,
                }
            )
        return result

    def describe_table(self, session_id: str, table_name: str) -> list[dict[str, Any]]:
        self.cleanup_expired_sessions()
        self._touch_session(session_id)
        con = duckdb.connect(str(self._db_path(session_id)), read_only=True)
        try:
            rows = con.execute(
                f"PRAGMA table_info({self._quote_ident(table_name)})"
            ).fetchdf()
        finally:
            con.close()

        result: list[dict[str, Any]] = []
        for row in rows.to_dict(orient="records"):
            result.append(
                {
                    "schema": "main",
                    "table_name": table_name,
                    "column_name": row["name"],
                    "data_type": row["type"],
                    "is_nullable": not bool(row["notnull"]),
                    "ordinal_position": int(row["cid"]) + 1,
                    "default_expression": row.get("dflt_value"),
                }
            )
        return result

    def query_dataframe(self, session_id: str, sql: str) -> pd.DataFrame:
        self.cleanup_expired_sessions()
        self._touch_session(session_id)
        con = duckdb.connect(str(self._db_path(session_id)), read_only=True)
        try:
            return con.execute(sql).fetchdf()
        finally:
            con.close()

    def delete_session(self, session_id: str) -> None:
        shutil.rmtree(self._session_dir(session_id), ignore_errors=True)


