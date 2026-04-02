from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid


USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,64}$")
THEME_VALUES = {"light", "dark"}
ANSWER_STYLE_VALUES = {"concise", "detailed"}
ANALYSIS_DEPTH_VALUES = {"light", "medium", "deep"}
# Must match agent.runner.DEPTH_PROFILES max_steps_cap.
ANALYSIS_DEPTH_MAX_OUTER_STEPS = {"light": 20, "medium": 30, "deep": 50}


@dataclass(frozen=True)
class AuthUser:
    id: int
    username: str
    is_admin: bool
    created_at: str


@dataclass(frozen=True)
class UserSettings:
    theme: str
    default_include_reasoning: bool
    default_answer_style: str
    analysis_depth: str
    llm_temperature_chat: float
    llm_temperature_tool: float
    llm_max_tokens_default: int
    llm_max_tokens_reasoning: int
    backend_query_timeout_sec: int
    agent_max_steps: int
    agent_step_timeout_sec: int
    agent_inner_recursion_limit: int
    ui_scale: int = 100


@dataclass(frozen=True)
class DBConnectionRecord:
    id: str
    user_id: int
    name: str
    db_type: str
    host: str
    port: int | None
    database: str | None
    username: str | None
    options_json: dict[str, object] | None
    last_test_at: str | None
    last_test_ok: bool | None
    last_error: str | None
    created_at: str
    updated_at: str
    password_present: bool


class AuthDB:
    def __init__(self, db_path: str, token_ttl_days: int = 30) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_ttl_days = token_ttl_days
        self._init_schema()
        self._cleanup_expired_tokens()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_password(password: str) -> str:
        salt = secrets.token_bytes(16)
        n = 2**14
        r = 8
        p = 1
        key = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=32,
        )
        salt_b64 = base64.b64encode(salt).decode("ascii")
        key_b64 = base64.b64encode(key).decode("ascii")
        return f"scrypt${n}${r}${p}${salt_b64}${key_b64}"

    @staticmethod
    def _verify_password(password: str, encoded: str) -> bool:
        try:
            algorithm, n_raw, r_raw, p_raw, salt_b64, key_b64 = encoded.split("$", 5)
        except ValueError:
            return False
        if algorithm != "scrypt":
            return False
        try:
            n = int(n_raw)
            r = int(r_raw)
            p = int(p_raw)
            salt = base64.b64decode(salt_b64.encode("ascii"))
            expected = base64.b64decode(key_b64.encode("ascii"))
        except Exception:
            return False
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(expected),
        )
        return hmac.compare_digest(derived, expected)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auth_tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL DEFAULT 'Новый чат',
                    created_at TEXT NOT NULL,
                    last_access TEXT NOT NULL,
                    has_dataset INTEGER NOT NULL DEFAULT 0,
                    last_message_preview TEXT,
                    allow_auto_title INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user
                    ON chat_sessions(user_id, last_access DESC);

                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id INTEGER PRIMARY KEY,
                    theme TEXT NOT NULL DEFAULT 'dark',
                    default_include_reasoning INTEGER NOT NULL DEFAULT 1,
                    default_answer_style TEXT NOT NULL DEFAULT 'detailed',
                    llm_temperature_chat REAL NOT NULL DEFAULT 0.7,
                    llm_temperature_tool REAL NOT NULL DEFAULT 0.5,
                    llm_max_tokens_default INTEGER NOT NULL DEFAULT 2048,
                    llm_max_tokens_reasoning INTEGER NOT NULL DEFAULT 4096,
                    backend_query_timeout_sec INTEGER NOT NULL DEFAULT 180,
                    agent_max_steps INTEGER NOT NULL DEFAULT 20,
                    agent_step_timeout_sec INTEGER NOT NULL DEFAULT 45,
                    agent_inner_recursion_limit INTEGER NOT NULL DEFAULT 14,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS user_tool_settings (
                    user_id INTEGER NOT NULL,
                    tool_key TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, tool_key),
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_user_tool_settings_user
                    ON user_tool_settings(user_id, tool_key);

                CREATE TABLE IF NOT EXISTS user_db_connections (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    db_type TEXT NOT NULL,
                    host TEXT NOT NULL,
                    port INTEGER,
                    database_name TEXT,
                    username TEXT,
                    options_json TEXT,
                    last_test_at TEXT,
                    last_test_ok INTEGER,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                    UNIQUE(user_id, name)
                );

                CREATE INDEX IF NOT EXISTS idx_user_db_connections_user
                    ON user_db_connections(user_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS user_db_connection_secrets (
                    connection_id TEXT PRIMARY KEY,
                    secret_blob_encrypted TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(connection_id) REFERENCES user_db_connections(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS user_memories (
                    user_id    INTEGER NOT NULL,
                    mem_type   TEXT    NOT NULL,
                    content    TEXT    NOT NULL DEFAULT '',
                    updated_at REAL    NOT NULL DEFAULT 0,
                    PRIMARY KEY (user_id, mem_type),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                """
            )
            self._ensure_chat_sessions_columns(conn)
            self._ensure_user_settings_columns(conn)
            self._ensure_user_db_connections_columns(conn)

    @staticmethod
    def _ensure_chat_sessions_columns(conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(chat_sessions)").fetchall()
        existing_columns = {str(row["name"]) for row in rows}
        if "title_is_custom" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE chat_sessions
                ADD COLUMN title_is_custom INTEGER NOT NULL DEFAULT 0
                """
            )
        if "allow_auto_title" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE chat_sessions
                ADD COLUMN allow_auto_title INTEGER NOT NULL DEFAULT 0
                """
            )

    @staticmethod
    def _ensure_user_settings_columns(conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(user_settings)").fetchall()
        existing_columns = {str(row["name"]) for row in rows}
        if "theme" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_settings
                ADD COLUMN theme TEXT NOT NULL DEFAULT 'dark'
                """
            )
        if "default_include_reasoning" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_settings
                ADD COLUMN default_include_reasoning INTEGER NOT NULL DEFAULT 1
                """
            )
        if "default_answer_style" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_settings
                ADD COLUMN default_answer_style TEXT NOT NULL DEFAULT 'detailed'
                """
            )
        if "updated_at" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_settings
                ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''
                """
            )
            conn.execute(
                "UPDATE user_settings SET updated_at = ? WHERE updated_at = ''",
                (datetime.now(timezone.utc).isoformat(),),
            )
        if "llm_temperature_chat" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_settings
                ADD COLUMN llm_temperature_chat REAL NOT NULL DEFAULT 0.7
                """
            )
        if "llm_temperature_tool" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_settings
                ADD COLUMN llm_temperature_tool REAL NOT NULL DEFAULT 0.5
                """
            )
        if "llm_max_tokens_default" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_settings
                ADD COLUMN llm_max_tokens_default INTEGER NOT NULL DEFAULT 2048
                """
            )
        if "llm_max_tokens_reasoning" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_settings
                ADD COLUMN llm_max_tokens_reasoning INTEGER NOT NULL DEFAULT 4096
                """
            )
        if "backend_query_timeout_sec" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_settings
                ADD COLUMN backend_query_timeout_sec INTEGER NOT NULL DEFAULT 180
                """
            )
        if "agent_max_steps" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_settings
                ADD COLUMN agent_max_steps INTEGER NOT NULL DEFAULT 20
                """
            )
        if "agent_step_timeout_sec" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_settings
                ADD COLUMN agent_step_timeout_sec INTEGER NOT NULL DEFAULT 45
                """
            )
        if "agent_inner_recursion_limit" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_settings
                ADD COLUMN agent_inner_recursion_limit INTEGER NOT NULL DEFAULT 14
                """
            )
        else:
            # Migrate existing rows that still have the old default of 6
            conn.execute(
                """
                UPDATE user_settings
                SET agent_inner_recursion_limit = 14
                WHERE agent_inner_recursion_limit <= 6
                """
            )
        if "analysis_depth" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_settings
                ADD COLUMN analysis_depth TEXT NOT NULL DEFAULT 'light'
                """
            )
        if "ui_scale" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_settings
                ADD COLUMN ui_scale INTEGER NOT NULL DEFAULT 100
                """
            )

    @staticmethod
    def _ensure_user_db_connections_columns(conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(user_db_connections)").fetchall()
        existing_columns = {str(row["name"]) for row in rows}
        if "database_name" not in existing_columns and "database" in existing_columns:
            conn.execute(
                """
                ALTER TABLE user_db_connections
                RENAME COLUMN database TO database_name
                """
            )

    def _cleanup_expired_tokens(self) -> None:
        now_iso = self._now_iso()
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM auth_tokens WHERE expires_at <= ?",
                (now_iso,),
            )

    @staticmethod
    def _normalize_theme(theme: str | None) -> str:
        value = str(theme or "").strip().lower()
        if value not in THEME_VALUES:
            raise ValueError("Допустимые темы: light или dark")
        return value

    @staticmethod
    def _normalize_answer_style(answer_style: str | None) -> str:
        value = str(answer_style or "").strip().lower()
        if value not in ANSWER_STYLE_VALUES:
            raise ValueError("Допустимые стили ответа: concise или detailed")
        return value

    @staticmethod
    def _normalize_analysis_depth(depth: str | None) -> str:
        value = str(depth or "").strip().lower()
        if value not in ANALYSIS_DEPTH_VALUES:
            raise ValueError("Допустимые уровни: light, medium или deep")
        return value

    def _ensure_user_settings_row(self, conn: sqlite3.Connection, user_id: int) -> None:
        conn.execute(
            """
            INSERT INTO user_settings(
                user_id,
                theme,
                default_include_reasoning,
                default_answer_style,
                analysis_depth,
                llm_temperature_chat,
                llm_temperature_tool,
                llm_max_tokens_default,
                llm_max_tokens_reasoning,
                backend_query_timeout_sec,
                agent_max_steps,
                agent_step_timeout_sec,
                agent_inner_recursion_limit,
                updated_at
            )
            VALUES (?, 'dark', 1, 'detailed', 'light', 0.7, 0.5, 2048, 4096, 180, 20, 45, 14, ?)
            ON CONFLICT(user_id) DO NOTHING
            """,
            (user_id, self._now_iso()),
        )

    @staticmethod
    def _to_auth_user(row: sqlite3.Row) -> AuthUser:
        return AuthUser(
            id=int(row["id"]),
            username=str(row["username"]),
            is_admin=bool(row["is_admin"]),
            created_at=str(row["created_at"]),
        )

    def ensure_default_admin(self, username: str, password: str) -> None:
        existing = self.get_user_by_username(username)
        if existing is not None:
            return
        self.create_user(username=username, password=password, is_admin=True)

    def get_user_by_username(self, username: str) -> AuthUser | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, username, is_admin, created_at FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if row is None:
            return None
        return self._to_auth_user(row)

    def create_user(self, username: str, password: str, is_admin: bool = False) -> AuthUser:
        normalized = username.strip()
        if not USERNAME_RE.match(normalized):
            raise ValueError(
                "Username должен содержать 3-64 символа: буквы, цифры, ., _, -"
            )
        if len(password) < 4:
            raise ValueError("Пароль должен быть не короче 4 символов")

        password_hash = self._hash_password(password)
        created_at = self._now_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users(username, password_hash, is_admin, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (normalized, password_hash, 1 if is_admin else 0, created_at),
            )
            user_id = int(cursor.lastrowid)
            self._ensure_user_settings_row(conn, user_id)
        return AuthUser(
            id=user_id,
            username=normalized,
            is_admin=is_admin,
            created_at=created_at,
        )

    def authenticate(self, username: str, password: str) -> AuthUser | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, username, password_hash, is_admin, created_at
                FROM users
                WHERE username = ?
                """,
                (username.strip(),),
            ).fetchone()
        if row is None:
            return None
        if not self._verify_password(password, str(row["password_hash"])):
            return None
        return self._to_auth_user(row)

    def create_access_token(self, user_id: int) -> str:
        token = secrets.token_urlsafe(40)
        token_hash = self._hash_token(token)
        created_at = self._now_iso()
        expires_at = (
            datetime.now(timezone.utc) + timedelta(days=self.token_ttl_days)
        ).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO auth_tokens(token_hash, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (token_hash, user_id, created_at, expires_at),
            )
        return token

    def get_user_by_token(self, token: str) -> AuthUser | None:
        token_hash = self._hash_token(token)
        now_iso = self._now_iso()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.username, u.is_admin, u.created_at
                FROM auth_tokens t
                JOIN users u ON u.id = t.user_id
                WHERE t.token_hash = ? AND t.expires_at > ?
                """,
                (token_hash, now_iso),
            ).fetchone()
        if row is None:
            return None
        return self._to_auth_user(row)

    def revoke_token(self, token: str) -> None:
        token_hash = self._hash_token(token)
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM auth_tokens WHERE token_hash = ?",
                (token_hash,),
            )

    @staticmethod
    def _count_admins(conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT COUNT(*) AS count FROM users WHERE is_admin = 1").fetchone()
        if row is None:
            return 0
        return int(row["count"])

    @staticmethod
    def _validate_new_password(password: str) -> None:
        if len(password) < 4:
            raise ValueError("Пароль должен быть не короче 4 символов")

    def get_user_by_id(self, user_id: int) -> AuthUser | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, username, is_admin, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return self._to_auth_user(row)

    def list_users(self) -> list[AuthUser]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, username, is_admin, created_at
                FROM users
                ORDER BY is_admin DESC, username ASC
                """
            ).fetchall()
        return [self._to_auth_user(row) for row in rows]

    def update_password_with_current(
        self, user_id: int, current_password: str, new_password: str
    ) -> None:
        self._validate_new_password(new_password)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Пользователь не найден")
            if not self._verify_password(current_password, str(row["password_hash"])):
                raise ValueError("Текущий пароль указан неверно")
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (self._hash_password(new_password), user_id),
            )

    def set_user_password(self, user_id: int, new_password: str) -> bool:
        self._validate_new_password(new_password)
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (self._hash_password(new_password), user_id),
            )
            return cursor.rowcount > 0

    def set_user_admin(self, user_id: int, is_admin: bool) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT is_admin FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            if row is None:
                return False
            current_is_admin = bool(row["is_admin"])
            if current_is_admin and not is_admin and self._count_admins(conn) <= 1:
                raise ValueError("Нельзя снять роль у последнего администратора")
            cursor = conn.execute(
                "UPDATE users SET is_admin = ? WHERE id = ?",
                (1 if is_admin else 0, user_id),
            )
            return cursor.rowcount > 0

    def delete_user(self, user_id: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT is_admin FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            if row is None:
                return False
            if bool(row["is_admin"]) and self._count_admins(conn) <= 1:
                raise ValueError("Нельзя удалить последнего администратора")
            cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            return cursor.rowcount > 0

    @staticmethod
    def _parse_user_settings(row: sqlite3.Row) -> UserSettings:
        raw_theme = str(row["theme"] or "").strip().lower()
        theme = raw_theme if raw_theme in THEME_VALUES else "dark"
        raw_style = str(row["default_answer_style"] or "").strip().lower()
        answer_style = raw_style if raw_style in ANSWER_STYLE_VALUES else "detailed"
        raw_depth = str(row["analysis_depth"] if "analysis_depth" in row.keys() else "light").strip().lower()
        depth = raw_depth if raw_depth in ANALYSIS_DEPTH_VALUES else "light"
        return UserSettings(
            theme=theme,
            default_include_reasoning=bool(row["default_include_reasoning"]),
            default_answer_style=answer_style,
            analysis_depth=depth,
            llm_temperature_chat=float(row["llm_temperature_chat"] or 0.7),
            llm_temperature_tool=float(row["llm_temperature_tool"] or 0.5),
            llm_max_tokens_default=int(row["llm_max_tokens_default"] or 2048),
            llm_max_tokens_reasoning=int(row["llm_max_tokens_reasoning"] or 4096),
            backend_query_timeout_sec=int(row["backend_query_timeout_sec"] or 180),
            agent_max_steps=min(
                max(2, int(row["agent_max_steps"] or 20)),
                ANALYSIS_DEPTH_MAX_OUTER_STEPS.get(depth, 20),
            ),
            agent_step_timeout_sec=int(row["agent_step_timeout_sec"] or 45),
            agent_inner_recursion_limit=int(row["agent_inner_recursion_limit"] or 14),
            ui_scale=int(row["ui_scale"] if "ui_scale" in row.keys() else 100),
        )

    def get_user_settings(self, user_id: int) -> UserSettings:
        with self._connect() as conn:
            self._ensure_user_settings_row(conn, user_id)
            row = conn.execute(
                """
                SELECT theme, default_include_reasoning, default_answer_style
                    , analysis_depth
                    , llm_temperature_chat, llm_temperature_tool
                    , llm_max_tokens_default, llm_max_tokens_reasoning
                    , backend_query_timeout_sec, agent_max_steps
                    , agent_step_timeout_sec, agent_inner_recursion_limit
                    , ui_scale
                FROM user_settings
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        if row is None:
            return UserSettings(
                theme="dark",
                default_include_reasoning=True,
                default_answer_style="detailed",
                analysis_depth="light",
                llm_temperature_chat=0.7,
                llm_temperature_tool=0.5,
                llm_max_tokens_default=2048,
                llm_max_tokens_reasoning=4096,
                backend_query_timeout_sec=180,
                agent_max_steps=20,
                agent_step_timeout_sec=45,
                agent_inner_recursion_limit=6,
                ui_scale=100,
            )
        return self._parse_user_settings(row)

    def update_user_settings(
        self,
        user_id: int,
        *,
        theme: str | None = None,
        default_include_reasoning: bool | None = None,
        default_answer_style: str | None = None,
        analysis_depth: str | None = None,
        llm_temperature_chat: float | None = None,
        llm_temperature_tool: float | None = None,
        llm_max_tokens_default: int | None = None,
        llm_max_tokens_reasoning: int | None = None,
        backend_query_timeout_sec: int | None = None,
        agent_max_steps: int | None = None,
        agent_step_timeout_sec: int | None = None,
        agent_inner_recursion_limit: int | None = None,
        ui_scale: int | None = None,
    ) -> UserSettings:
        with self._connect() as conn:
            self._ensure_user_settings_row(conn, user_id)
            current_row = conn.execute(
                """
                SELECT theme, default_include_reasoning, default_answer_style
                    , analysis_depth
                    , llm_temperature_chat, llm_temperature_tool
                    , llm_max_tokens_default, llm_max_tokens_reasoning
                    , backend_query_timeout_sec, agent_max_steps
                    , agent_step_timeout_sec, agent_inner_recursion_limit
                    , ui_scale
                FROM user_settings
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
            if current_row is None:
                raise ValueError("Настройки пользователя не найдены")
            current = self._parse_user_settings(current_row)
            next_theme = (
                self._normalize_theme(theme) if theme is not None else current.theme
            )
            next_reasoning = (
                bool(default_include_reasoning)
                if default_include_reasoning is not None
                else current.default_include_reasoning
            )
            next_answer_style = (
                self._normalize_answer_style(default_answer_style)
                if default_answer_style is not None
                else current.default_answer_style
            )
            next_analysis_depth = (
                self._normalize_analysis_depth(analysis_depth)
                if analysis_depth is not None
                else current.analysis_depth
            )
            next_llm_temperature_chat = (
                float(llm_temperature_chat)
                if llm_temperature_chat is not None
                else current.llm_temperature_chat
            )
            next_llm_temperature_tool = (
                float(llm_temperature_tool)
                if llm_temperature_tool is not None
                else current.llm_temperature_tool
            )
            next_llm_max_tokens_default = (
                int(llm_max_tokens_default)
                if llm_max_tokens_default is not None
                else current.llm_max_tokens_default
            )
            next_llm_max_tokens_reasoning = (
                int(llm_max_tokens_reasoning)
                if llm_max_tokens_reasoning is not None
                else current.llm_max_tokens_reasoning
            )
            next_backend_query_timeout_sec = (
                int(backend_query_timeout_sec)
                if backend_query_timeout_sec is not None
                else current.backend_query_timeout_sec
            )
            next_agent_max_steps = (
                int(agent_max_steps)
                if agent_max_steps is not None
                else current.agent_max_steps
            )
            _depth_cap = ANALYSIS_DEPTH_MAX_OUTER_STEPS.get(next_analysis_depth, 20)
            next_agent_max_steps = min(max(2, next_agent_max_steps), _depth_cap)
            next_agent_step_timeout_sec = (
                int(agent_step_timeout_sec)
                if agent_step_timeout_sec is not None
                else current.agent_step_timeout_sec
            )
            next_agent_inner_recursion_limit = (
                int(agent_inner_recursion_limit)
                if agent_inner_recursion_limit is not None
                else current.agent_inner_recursion_limit
            )
            next_ui_scale = (
                min(max(70, int(ui_scale)), 150)
                if ui_scale is not None
                else current.ui_scale
            )
            conn.execute(
                """
                UPDATE user_settings
                SET theme = ?, default_include_reasoning = ?, default_answer_style = ?,
                    analysis_depth = ?,
                    llm_temperature_chat = ?, llm_temperature_tool = ?,
                    llm_max_tokens_default = ?, llm_max_tokens_reasoning = ?,
                    backend_query_timeout_sec = ?, agent_max_steps = ?,
                    agent_step_timeout_sec = ?, agent_inner_recursion_limit = ?,
                    ui_scale = ?,
                    updated_at = ?
                WHERE user_id = ?
                """,
                (
                    next_theme,
                    1 if next_reasoning else 0,
                    next_answer_style,
                    next_analysis_depth,
                    next_llm_temperature_chat,
                    next_llm_temperature_tool,
                    next_llm_max_tokens_default,
                    next_llm_max_tokens_reasoning,
                    next_backend_query_timeout_sec,
                    next_agent_max_steps,
                    next_agent_step_timeout_sec,
                    next_agent_inner_recursion_limit,
                    next_ui_scale,
                    self._now_iso(),
                    user_id,
                ),
            )
        return UserSettings(
            theme=next_theme,
            default_include_reasoning=next_reasoning,
            default_answer_style=next_answer_style,
            analysis_depth=next_analysis_depth,
            llm_temperature_chat=next_llm_temperature_chat,
            llm_temperature_tool=next_llm_temperature_tool,
            llm_max_tokens_default=next_llm_max_tokens_default,
            llm_max_tokens_reasoning=next_llm_max_tokens_reasoning,
            backend_query_timeout_sec=next_backend_query_timeout_sec,
            agent_max_steps=next_agent_max_steps,
            agent_step_timeout_sec=next_agent_step_timeout_sec,
            agent_inner_recursion_limit=next_agent_inner_recursion_limit,
            ui_scale=next_ui_scale,
        )

    def list_user_tool_settings(self, user_id: int) -> dict[str, bool]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT tool_key, enabled
                FROM user_tool_settings
                WHERE user_id = ?
                ORDER BY tool_key ASC
                """,
                (user_id,),
            ).fetchall()
        return {
            str(row["tool_key"]): bool(row["enabled"])
            for row in rows
            if str(row["tool_key"]).strip()
        }

    def set_user_tool_enabled(self, user_id: int, tool_key: str, enabled: bool) -> None:
        clean_tool_key = str(tool_key or "").strip()
        if not clean_tool_key:
            raise ValueError("tool_key is required")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_tool_settings(user_id, tool_key, enabled, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, tool_key) DO UPDATE SET
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    clean_tool_key,
                    1 if enabled else 0,
                    self._now_iso(),
                ),
            )

    def register_session(
        self, session_id: str, user_id: int, allow_auto_title: bool = False
    ) -> None:
        now_iso = self._now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO chat_sessions(
                    session_id, user_id, title, created_at, last_access, has_dataset, last_message_preview, title_is_custom, allow_auto_title
                )
                VALUES (
                    ?,
                    ?,
                    COALESCE((SELECT title FROM chat_sessions WHERE session_id = ?), 'Новый чат'),
                    COALESCE((SELECT created_at FROM chat_sessions WHERE session_id = ?), ?),
                    ?,
                    COALESCE((SELECT has_dataset FROM chat_sessions WHERE session_id = ?), 0),
                    COALESCE((SELECT last_message_preview FROM chat_sessions WHERE session_id = ?), NULL),
                    COALESCE((SELECT title_is_custom FROM chat_sessions WHERE session_id = ?), 0),
                    COALESCE((SELECT allow_auto_title FROM chat_sessions WHERE session_id = ?), ?)
                )
                """,
                (
                    session_id,
                    user_id,
                    session_id,
                    session_id,
                    now_iso,
                    now_iso,
                    session_id,
                    session_id,
                    session_id,
                    session_id,
                    1 if allow_auto_title else 0,
                ),
            )

    def is_session_owner(self, session_id: str, user_id: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM chat_sessions WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
        return row is not None

    def touch_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chat_sessions SET last_access = ? WHERE session_id = ?",
                (self._now_iso(), session_id),
            )

    def set_session_title(self, session_id: str, user_id: int, title: str) -> bool:
        cleaned = title.strip()
        if len(cleaned) > 120:
            cleaned = cleaned[:120].rstrip()
        if not cleaned:
            return False
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE chat_sessions
                SET title = ?, title_is_custom = 1, last_access = ?
                WHERE session_id = ? AND user_id = ?
                """,
                (cleaned, self._now_iso(), session_id, user_id),
            )
            return cursor.rowcount > 0

    def is_session_title_custom(self, session_id: str, user_id: int) -> bool | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT title_is_custom
                FROM chat_sessions
                WHERE session_id = ? AND user_id = ?
                """,
                (session_id, user_id),
            ).fetchone()
        if row is None:
            return None
        return bool(row["title_is_custom"])

    def mark_session_has_dataset(self, session_id: str, has_dataset: bool) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chat_sessions SET has_dataset = ?, last_access = ? WHERE session_id = ?",
                (1 if has_dataset else 0, self._now_iso(), session_id),
            )

    def update_session_after_reply(
        self,
        session_id: str,
        assistant_text: str,
        auto_title: str | None = None,
    ) -> None:
        title_candidate = str(auto_title or "").strip() or None
        if title_candidate and len(title_candidate) > 120:
            title_candidate = title_candidate[:120].rstrip() or None

        preview = assistant_text.strip()
        if len(preview) > 200:
            preview = preview[:200].rstrip() + "..."

        now_iso = self._now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE chat_sessions
                SET
                    title = CASE
                        WHEN allow_auto_title = 1
                            AND title_is_custom = 0
                            AND ? IS NOT NULL
                            AND trim(?) <> ''
                        THEN ?
                        ELSE title
                    END,
                    last_message_preview = ?,
                    last_access = ?
                WHERE session_id = ?
                """,
                (
                    title_candidate,
                    title_candidate,
                    title_candidate,
                    preview or None,
                    now_iso,
                    session_id,
                ),
            )

    def delete_session(self, session_id: str, user_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM chat_sessions WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            )
            return cursor.rowcount > 0

    def delete_all_sessions(self, user_id: int) -> list[str]:
        """Delete all sessions owned by *user_id*.

        Returns a list of deleted session_ids so the caller can also clean up
        any associated file-system state.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id FROM chat_sessions WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            session_ids = [str(row["session_id"]) for row in rows]
            if session_ids:
                conn.execute(
                    "DELETE FROM chat_sessions WHERE user_id = ?",
                    (user_id,),
                )
        return session_ids

    def list_sessions(self, user_id: int) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    session_id,
                    title,
                    created_at,
                    last_access,
                    has_dataset,
                    last_message_preview
                FROM chat_sessions
                WHERE user_id = ?
                ORDER BY datetime(last_access) DESC
                """,
                (user_id,),
            ).fetchall()

        result: list[dict[str, object]] = []
        for row in rows:
            result.append(
                {
                    "session_id": str(row["session_id"]),
                    "title": str(row["title"] or "Новый чат"),
                    "created_at": str(row["created_at"]),
                    "last_access": str(row["last_access"]),
                    "has_dataset": bool(row["has_dataset"]),
                    "last_message_preview": (
                        str(row["last_message_preview"])
                        if row["last_message_preview"] is not None
                        else None
                    ),
                }
            )
        return result

    def get_session_metadata(
        self, session_id: str, user_id: int
    ) -> dict[str, object] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    session_id,
                    title,
                    created_at,
                    last_access,
                    has_dataset,
                    last_message_preview
                FROM chat_sessions
                WHERE session_id = ? AND user_id = ?
                """,
                (session_id, user_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "session_id": str(row["session_id"]),
            "title": str(row["title"] or "Новый чат"),
            "created_at": str(row["created_at"]),
            "last_access": str(row["last_access"]),
            "has_dataset": bool(row["has_dataset"]),
            "last_message_preview": (
                str(row["last_message_preview"])
                if row["last_message_preview"] is not None
                else None
            ),
        }

    @staticmethod
    def _parse_options_json(raw: str | None) -> dict[str, object] | None:
        if raw is None or str(raw).strip() == "":
            return None
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _to_db_connection_record(row: sqlite3.Row) -> DBConnectionRecord:
        return DBConnectionRecord(
            id=str(row["id"]),
            user_id=int(row["user_id"]),
            name=str(row["name"]),
            db_type=str(row["db_type"]),
            host=str(row["host"]),
            port=int(row["port"]) if row["port"] is not None else None,
            database=str(row["database_name"]) if row["database_name"] is not None else None,
            username=str(row["username"]) if row["username"] is not None else None,
            options_json=AuthDB._parse_options_json(
                str(row["options_json"]) if row["options_json"] is not None else None
            ),
            last_test_at=str(row["last_test_at"]) if row["last_test_at"] is not None else None,
            last_test_ok=bool(row["last_test_ok"]) if row["last_test_ok"] is not None else None,
            last_error=str(row["last_error"]) if row["last_error"] is not None else None,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            password_present=bool(row["password_present"]),
        )

    def list_db_connections(self, user_id: int) -> list[DBConnectionRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.id, c.user_id, c.name, c.db_type, c.host, c.port,
                    c.database_name, c.username, c.options_json,
                    c.last_test_at, c.last_test_ok, c.last_error,
                    c.created_at, c.updated_at,
                    CASE WHEN s.connection_id IS NOT NULL THEN 1 ELSE 0 END AS password_present
                FROM user_db_connections c
                LEFT JOIN user_db_connection_secrets s ON s.connection_id = c.id
                WHERE c.user_id = ?
                ORDER BY datetime(c.updated_at) DESC, datetime(c.created_at) DESC
                """,
                (user_id,),
            ).fetchall()
        return [self._to_db_connection_record(row) for row in rows]

    def get_db_connection(self, user_id: int, connection_id: str) -> DBConnectionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    c.id, c.user_id, c.name, c.db_type, c.host, c.port,
                    c.database_name, c.username, c.options_json,
                    c.last_test_at, c.last_test_ok, c.last_error,
                    c.created_at, c.updated_at,
                    CASE WHEN s.connection_id IS NOT NULL THEN 1 ELSE 0 END AS password_present
                FROM user_db_connections c
                LEFT JOIN user_db_connection_secrets s ON s.connection_id = c.id
                WHERE c.user_id = ? AND c.id = ?
                """,
                (user_id, connection_id),
            ).fetchone()
        if row is None:
            return None
        return self._to_db_connection_record(row)

    def create_db_connection(
        self,
        user_id: int,
        *,
        name: str,
        db_type: str,
        host: str,
        port: int | None,
        database: str | None,
        username: str | None,
        options_json: dict[str, object] | None,
    ) -> DBConnectionRecord:
        connection_id = uuid.uuid4().hex
        now_iso = self._now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_db_connections(
                    id, user_id, name, db_type, host, port, database_name, username,
                    options_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    connection_id,
                    user_id,
                    name,
                    db_type,
                    host,
                    port,
                    database,
                    username,
                    json.dumps(options_json, ensure_ascii=False) if options_json is not None else None,
                    now_iso,
                    now_iso,
                ),
            )
        created = self.get_db_connection(user_id, connection_id)
        if created is None:
            raise RuntimeError("Failed to read created DB connection.")
        return created

    def update_db_connection(
        self,
        user_id: int,
        connection_id: str,
        *,
        name: str | None,
        db_type: str | None,
        host: str | None,
        port: int | None,
        database: str | None,
        username: str | None,
        options_json: dict[str, object] | None,
        options_json_set: bool,
    ) -> DBConnectionRecord | None:
        current = self.get_db_connection(user_id, connection_id)
        if current is None:
            return None

        next_name = name if name is not None else current.name
        next_type = db_type if db_type is not None else current.db_type
        next_host = host if host is not None else current.host
        next_port = port if port is not None else current.port
        next_database = database if database is not None else current.database
        next_username = username if username is not None else current.username
        next_options = options_json if options_json_set else current.options_json

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE user_db_connections
                SET name = ?, db_type = ?, host = ?, port = ?, database_name = ?,
                    username = ?, options_json = ?, updated_at = ?
                WHERE user_id = ? AND id = ?
                """,
                (
                    next_name,
                    next_type,
                    next_host,
                    next_port,
                    next_database,
                    next_username,
                    json.dumps(next_options, ensure_ascii=False) if next_options is not None else None,
                    self._now_iso(),
                    user_id,
                    connection_id,
                ),
            )
        return self.get_db_connection(user_id, connection_id)

    def delete_db_connection(self, user_id: int, connection_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM user_db_connections WHERE user_id = ? AND id = ?",
                (user_id, connection_id),
            )
            return cursor.rowcount > 0

    def set_db_connection_secret(
        self, connection_id: str, secret_blob_encrypted: str
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_db_connection_secrets(connection_id, secret_blob_encrypted, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(connection_id) DO UPDATE SET
                    secret_blob_encrypted = excluded.secret_blob_encrypted,
                    updated_at = excluded.updated_at
                """,
                (connection_id, secret_blob_encrypted, self._now_iso()),
            )

    def clear_db_connection_secret(self, connection_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM user_db_connection_secrets WHERE connection_id = ?",
                (connection_id,),
            )

    def get_db_connection_secret_blob(
        self, user_id: int, connection_id: str
    ) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT s.secret_blob_encrypted
                FROM user_db_connection_secrets s
                JOIN user_db_connections c ON c.id = s.connection_id
                WHERE c.user_id = ? AND c.id = ?
                """,
                (user_id, connection_id),
            ).fetchone()
        if row is None:
            return None
        return str(row["secret_blob_encrypted"])

    def update_db_connection_test_status(
        self,
        user_id: int,
        connection_id: str,
        *,
        ok: bool,
        error: str | None,
        tested_at: str | None = None,
    ) -> DBConnectionRecord | None:
        checked_at = tested_at or self._now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE user_db_connections
                SET last_test_at = ?, last_test_ok = ?, last_error = ?, updated_at = ?
                WHERE user_id = ? AND id = ?
                """,
                (
                    checked_at,
                    1 if ok else 0,
                    error,
                    checked_at,
                    user_id,
                    connection_id,
                ),
            )
        return self.get_db_connection(user_id, connection_id)

    # ── User memory (profile / notes) ────────────────────────────────────────

    def get_user_memory(self, user_id: int, mem_type: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content FROM user_memories WHERE user_id = ? AND mem_type = ?",
                (user_id, mem_type),
            ).fetchone()
        return str(row["content"]) if row else ""

    def set_user_memory(self, user_id: int, mem_type: str, content: str) -> None:
        import time as _time
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_memories (user_id, mem_type, content, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, mem_type) DO UPDATE
                    SET content = excluded.content,
                        updated_at = excluded.updated_at
                """,
                (user_id, mem_type, content, _time.time()),
            )


