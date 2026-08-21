from __future__ import annotations

import io
import json
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd

_TABLE_RE = re.compile(r"[^A-Za-z0-9_]+")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DEFAULT_TTL_SEC = int(os.getenv("CSV_SESSION_TTL_SEC", "7200"))
_BASE_DIR = Path(
    os.getenv(
        "CSV_SESSION_BASE_DIR",
        str(Path(tempfile.gettempdir()) / "llm-data-analyst" / "csv_sessions"),
    )
).resolve()


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
    def unique_table_name(file_name: str, existing: Iterable[str]) -> str:
        base = CSVSessionRuntime.sanitize_table_name(file_name)
        taken = {str(item).strip().lower() for item in existing if str(item).strip()}
        if base not in taken:
            return base
        counter = 2
        while True:
            suffix = f"_{counter}"
            candidate = f"{base[: 120 - len(suffix)]}{suffix}"
            if candidate not in taken:
                return candidate
            counter += 1

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

    def db_exists(self, session_id: str) -> bool:
        return self._db_path(session_id).exists()

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

    def register_dataframes(
        self,
        *,
        session_id: str,
        tables: dict[str, pd.DataFrame],
        ttl_seconds: int | None = None,
    ) -> CSVSessionInfo:
        self.cleanup_expired_sessions()

        if not tables:
            raise ValueError("No tables to register")

        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        db_path = self._db_path(session_id)

        safe_tables = {
            self.sanitize_table_name(table_name): df
            for table_name, df in tables.items()
        }

        with self._lock:
            con = duckdb.connect(str(db_path))
            try:
                con.execute("BEGIN TRANSACTION")
                for safe_table, df in safe_tables.items():
                    con.register("_uploaded_df", df)
                    try:
                        con.execute(
                            f"CREATE OR REPLACE TABLE {self._quote_ident(safe_table)} AS "
                            "SELECT * FROM _uploaded_df"
                        )
                    finally:
                        try:
                            con.unregister("_uploaded_df")
                        except Exception:
                            pass
                con.execute("COMMIT")
            except Exception:
                try:
                    con.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                con.close()

            meta = self._touch_session(session_id, ttl_seconds=ttl_seconds)
            names = list(meta.get("table_names") or [])
            names.extend(safe_tables.keys())
            meta["table_names"] = sorted(set(names))
            self._write_meta(session_id, meta)

        return CSVSessionInfo(
            session_id=session_id,
            db_path=str(db_path),
            table_names=list(meta["table_names"]),
            expires_at=int(meta["expires_at"]),
        )

    def unregister_tables(self, session_id: str, table_names: Iterable[str]) -> None:
        """Drop session tables and remove them from runtime metadata."""
        safe_tables = [
            self.sanitize_table_name(table_name)
            for table_name in table_names
            if str(table_name or "").strip()
        ]
        if not safe_tables:
            return
        safe_table_set = set(safe_tables)

        with self._lock:
            db_path = self._db_path(session_id)
            if db_path.exists():
                con = duckdb.connect(str(db_path))
                try:
                    con.execute("BEGIN TRANSACTION")
                    for table_name in safe_tables:
                        con.execute(f"DROP TABLE IF EXISTS {self._quote_ident(table_name)}")
                    con.execute("COMMIT")
                except Exception:
                    try:
                        con.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise
                finally:
                    con.close()

            meta = self._read_meta(session_id)
            remaining = [
                name
                for name in list(meta.get("table_names") or [])
                if str(name) not in safe_table_set
            ]
            meta["table_names"] = sorted(set(remaining))
            self._write_meta(session_id, meta)

    def register_csv_bytes(
        self,
        *,
        session_id: str,
        file_name: str,
        csv_bytes: bytes,
        ttl_seconds: int | None = None,
        pandas_read_csv_kwargs: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, CSVSessionInfo]:
        df = self._read_csv_resilient(
            csv_bytes,
            pandas_read_csv_kwargs=dict(pandas_read_csv_kwargs or {}),
        )
        info = self.register_dataframe(
            session_id=session_id,
            table_name=file_name,
            df=df,
            ttl_seconds=ttl_seconds,
        )
        return df, info

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize empty/duplicate columns for stable downstream prompts/tools."""
        raw_columns = [str(col).replace("\ufeff", "").strip() for col in list(df.columns)]
        normalized: list[str] = []
        seen: dict[str, int] = {}
        for idx, col in enumerate(raw_columns):
            base = col or f"column_{idx + 1}"
            count = seen.get(base, 0) + 1
            seen[base] = count
            normalized.append(base if count == 1 else f"{base}_{count}")
        df = df.copy()
        df.columns = normalized
        return df

    @staticmethod
    def _read_csv_resilient(
        csv_bytes: bytes,
        *,
        pandas_read_csv_kwargs: dict[str, Any],
    ) -> pd.DataFrame:
        """Best-effort CSV parsing for demo stability across common encodings/separators."""
        if not csv_bytes:
            raise ValueError("CSV file is empty")

        base_kwargs = dict(pandas_read_csv_kwargs)
        base_kwargs.setdefault("skip_blank_lines", True)
        base_kwargs.setdefault("low_memory", False)
        # Pragmatic fallback for noisy real-world CSVs; caller can override.
        base_kwargs.setdefault("on_bad_lines", "skip")

        encodings = (
            [str(base_kwargs["encoding"])]
            if "encoding" in base_kwargs
            else ["utf-8-sig", "utf-8", "cp1251", "latin-1"]
        )

        separators: list[Any]
        if "sep" in base_kwargs:
            separators = [base_kwargs["sep"]]
        else:
            # Try auto-detect first, then common explicit delimiters.
            separators = [None, ",", ";", "\t", "|"]

        last_error: Exception | None = None
        for encoding in encodings:
            for sep in separators:
                kwargs = dict(base_kwargs)
                kwargs["encoding"] = encoding
                kwargs["sep"] = sep
                if sep is None and "engine" not in kwargs:
                    kwargs["engine"] = "python"
                try:
                    df = pd.read_csv(io.BytesIO(csv_bytes), **kwargs)
                    return CSVSessionRuntime._normalize_columns(df)
                except Exception as exc:  # noqa: BLE001 - keep trying fallbacks
                    last_error = exc
                    continue

        raise ValueError(f"Unable to parse CSV with common encodings/separators: {last_error}")

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

        result: list[dict[str, Any]] = [
            {
                "schema": "main",
                "table_name": table_name,
                "column_name": row["name"],
                "data_type": row["type"],
                "is_nullable": not bool(row["notnull"]),
                "ordinal_position": int(row["cid"]) + 1,
                "default_expression": row.get("dflt_value"),
            }
            for row in rows.to_dict(orient="records")
        ]
        return result

    def query_dataframe(self, session_id: str, sql: str) -> pd.DataFrame:
        self.cleanup_expired_sessions()
        self._touch_session(session_id)
        con = duckdb.connect(str(self._db_path(session_id)), read_only=True)
        try:
            df = con.execute(sql).fetchdf()
            for column in df.select_dtypes(include=["object", "string"]).columns:
                df[column] = df[column].astype(object).where(pd.notna(df[column]), None)
            return df
        finally:
            con.close()

    def delete_session(self, session_id: str) -> None:
        shutil.rmtree(self._session_dir(session_id), ignore_errors=True)
