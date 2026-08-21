from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class SkillOverride:
    skill_id: str
    name: str | None
    description: str | None
    triggers: tuple[str, ...] | None
    core_markdown: str | None
    details_markdown: str | None
    updated_by: int | None
    updated_at: str


class SkillOverrideStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(UTC).isoformat()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS skill_overrides (
                    skill_id        TEXT PRIMARY KEY,
                    name            TEXT,
                    description     TEXT,
                    triggers_json   TEXT,
                    core_markdown   TEXT,
                    details_markdown TEXT,
                    updated_by      INTEGER NOT NULL,
                    updated_at      TEXT NOT NULL,
                    FOREIGN KEY(updated_by) REFERENCES users(id) ON DELETE SET NULL
                );
                """
            )

    def get_override(self, skill_id: str) -> SkillOverride | None:
        clean_id = str(skill_id or "").strip()
        if not clean_id:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT skill_id, name, description, triggers_json,
                       core_markdown, details_markdown,
                       updated_by, updated_at
                FROM skill_overrides
                WHERE skill_id = ?
                """,
                (clean_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_override(row)

    def get_all(self) -> dict[str, SkillOverride]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT skill_id, name, description, triggers_json,
                       core_markdown, details_markdown,
                       updated_by, updated_at
                FROM skill_overrides
                ORDER BY skill_id ASC
                """,
            ).fetchall()
        return {str(r["skill_id"]): self._row_to_override(r) for r in rows}

    def save_override(
        self,
        skill_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        triggers: list[str] | tuple[str, ...] | None = None,
        core_markdown: str | None = None,
        details_markdown: str | None = None,
        user_id: int,
    ) -> SkillOverride:
        clean_id = str(skill_id or "").strip()
        if not clean_id:
            raise ValueError("skill_id is required")

        triggers_json: str | None = None
        if triggers is not None:
            triggers_json = json.dumps(list(triggers), ensure_ascii=False)

        now = self._now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO skill_overrides
                    (skill_id, name, description, triggers_json,
                     core_markdown, details_markdown, updated_by, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(skill_id) DO UPDATE SET
                    name            = coalesce(excluded.name, skill_overrides.name),
                    description     = coalesce(excluded.description, skill_overrides.description),
                    triggers_json   = CASE
                        WHEN excluded.triggers_json IS NOT NULL THEN excluded.triggers_json
                        ELSE skill_overrides.triggers_json
                    END,
                    core_markdown   = CASE
                        WHEN excluded.core_markdown IS NOT NULL THEN excluded.core_markdown
                        ELSE skill_overrides.core_markdown
                    END,
                    details_markdown = CASE
                        WHEN excluded.details_markdown IS NOT NULL THEN excluded.details_markdown
                        ELSE skill_overrides.details_markdown
                    END,
                    updated_by      = excluded.updated_by,
                    updated_at      = excluded.updated_at
                """,
                (
                    clean_id,
                    name,
                    description,
                    triggers_json,
                    core_markdown,
                    details_markdown,
                    user_id,
                    now,
                ),
            )

        return SkillOverride(
            skill_id=clean_id,
            name=name,
            description=description,
            triggers=tuple(triggers) if triggers else None,
            core_markdown=core_markdown,
            details_markdown=details_markdown,
            updated_by=user_id,
            updated_at=now,
        )

    def delete_override(self, skill_id: str) -> bool:
        clean_id = str(skill_id or "").strip()
        if not clean_id:
            return False
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM skill_overrides WHERE skill_id = ?",
                (clean_id,),
            )
            return cursor.rowcount > 0

    def has_override(self, skill_id: str) -> bool:
        return self.get_override(skill_id) is not None

    def override_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM skill_overrides").fetchone()
            return int(row["cnt"]) if row else 0

    @staticmethod
    def _row_to_override(row: sqlite3.Row) -> SkillOverride:
        triggers: tuple[str, ...] | None = None
        tj = row["triggers_json"]
        if tj is not None:
            try:
                parsed = tj if isinstance(tj, list) else json.loads(str(tj))
                if isinstance(parsed, list):
                    triggers = tuple(str(t).strip() for t in parsed if str(t).strip())
            except (json.JSONDecodeError, TypeError):
                pass
        return SkillOverride(
            skill_id=str(row["skill_id"]),
            name=str(row["name"]) if row["name"] is not None else None,
            description=str(row["description"]) if row["description"] is not None else None,
            triggers=triggers,
            core_markdown=str(row["core_markdown"]) if row["core_markdown"] is not None else None,
            details_markdown=str(row["details_markdown"]) if row["details_markdown"] is not None else None,
            updated_by=int(row["updated_by"]) if row["updated_by"] is not None else None,
            updated_at=str(row["updated_at"]),
        )


class PostgresSkillOverrideStore(SkillOverrideStore):
    def __init__(self, dsn: str, *, schema: str = "public") -> None:
        from backend.auth.app_data_postgres import AppDataPostgresStore

        self._store = AppDataPostgresStore(dsn, schema=schema)

    def initialize(self) -> None:
        self._store.ensure_schema()

    def _connect(self):
        return self._store.connect()
