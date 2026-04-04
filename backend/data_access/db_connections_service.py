from __future__ import annotations

import ipaddress
import socket
import sqlite3
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, status

from backend.auth.auth_db import AuthDB, DBConnectionRecord
from backend.core.config import Settings
from backend.core.redaction import sanitize_error_text
from backend.data_access.crypto_service import SecretCryptoError, SecretCryptoService
from backend.data_access.db_connectors import ResolvedDBConnection, build_connection_adapter

SUPPORTED_DB_TYPES = {"postgres", "postgresql", "clickhouse"}


@dataclass(frozen=True)
class DBConnectionSecretState:
    password: str | None = None


class DBConnectionsService:
    def __init__(self, auth_db: AuthDB, settings: Settings) -> None:
        self.auth_db = auth_db
        self.settings = settings
        self.crypto = SecretCryptoService(settings)

    @staticmethod
    def normalize_db_type(db_type: str) -> str:
        normalized = str(db_type or "").strip().lower()
        if normalized == "postgres":
            normalized = "postgresql"
        if normalized not in SUPPORTED_DB_TYPES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Unsupported db_type.",
            )
        return normalized

    @staticmethod
    def _default_port(db_type: str) -> int:
        if db_type == "postgresql":
            return 5432
        if db_type == "clickhouse":
            return 8123
        return 0

    @staticmethod
    def _is_private_or_local_ip(value: str) -> bool:
        ip = ipaddress.ip_address(value)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        )

    def validate_host(self, host: str) -> str:
        clean = str(host or "").strip()
        if not clean:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Host must not be blank.",
            )
        if self.settings.db_connections_allow_private_hosts:
            return clean

        lowered = clean.lower()
        if lowered in {"localhost", "localhost.localdomain"}:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Private/loopback DB hosts are not allowed.",
            )

        try:
            if self._is_private_or_local_ip(clean):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Private/loopback DB hosts are not allowed.",
                )
            return clean
        except ValueError:
            pass

        try:
            infos = socket.getaddrinfo(clean, None)
            for info in infos:
                resolved_ip = info[4][0]
                if self._is_private_or_local_ip(resolved_ip):
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail="Private/loopback DB hosts are not allowed.",
                    )
        except socket.gaierror:
            return clean

        return clean

    @staticmethod
    def _normalize_name(name: str) -> str:
        clean = str(name or "").strip()
        if not clean:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Connection name must not be blank.",
            )
        return clean

    @staticmethod
    def _normalize_optional_text(value: str | None) -> str | None:
        if value is None:
            return None
        clean = str(value).strip()
        return clean or None

    @staticmethod
    def _normalize_options_json(options_json: dict[str, Any] | None) -> dict[str, Any] | None:
        if options_json is None:
            return None
        if not isinstance(options_json, dict):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="options_json must be an object.",
            )
        normalized = dict(options_json)
        schema = normalized.get("schema")
        if schema is None:
            normalized.pop("schema", None)
        else:
            clean_schema = str(schema).strip()
            if clean_schema:
                normalized["schema"] = clean_schema
            else:
                normalized.pop("schema", None)
        return normalized

    def _load_secret_state(self, user_id: int, connection_id: str) -> DBConnectionSecretState:
        encrypted_blob = self.auth_db.get_db_connection_secret_blob(user_id, connection_id)
        if not encrypted_blob:
            return DBConnectionSecretState()
        try:
            payload = self.crypto.decrypt_payload(encrypted_blob)
        except SecretCryptoError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        password = payload.get("password")
        return DBConnectionSecretState(
            password=str(password) if password is not None else None,
        )

    def _store_secret_state(
        self,
        connection_id: str,
        *,
        password: str | None,
    ) -> None:
        payload = {"password": password}
        try:
            encrypted = self.crypto.encrypt_payload(payload)
        except SecretCryptoError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        self.auth_db.set_db_connection_secret(connection_id, encrypted)

    def _encrypt_secret_payload(
        self,
        *,
        password: str | None,
    ) -> str:
        payload = {"password": password}
        try:
            return self.crypto.encrypt_payload(payload)
        except SecretCryptoError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    def list_connections(self, user_id: int) -> list[DBConnectionRecord]:
        return self.auth_db.list_db_connections(user_id)

    def get_connection(self, user_id: int, connection_id: str) -> DBConnectionRecord:
        row = self.auth_db.get_db_connection(user_id, connection_id)
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="DB connection not found.")
        return row

    def create_connection(
        self,
        user_id: int,
        *,
        name: str,
        db_type: str,
        host: str,
        port: int | None,
        database: str | None,
        username: str | None,
        password: str | None,
        options_json: dict[str, Any] | None,
    ) -> DBConnectionRecord:
        normalized_type = self.normalize_db_type(db_type)
        normalized_host = self.validate_host(host)
        normalized_options = self._normalize_options_json(options_json)
        normalized_password = self._normalize_optional_text(password)
        encrypted_secret_blob: str | None = None
        if normalized_password is not None:
            encrypted_secret_blob = self._encrypt_secret_payload(
                password=normalized_password
            )
        try:
            created = self.auth_db.create_db_connection(
                user_id,
                name=self._normalize_name(name),
                db_type=normalized_type,
                host=normalized_host,
                port=port or self._default_port(normalized_type),
                database=self._normalize_optional_text(database),
                username=self._normalize_optional_text(username),
                options_json=normalized_options,
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="Connection name already exists."
            ) from exc

        if encrypted_secret_blob is not None:
            self.auth_db.set_db_connection_secret(created.id, encrypted_secret_blob)
        return self.get_connection(user_id, created.id)

    def update_connection(
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
        password: str | None,
        clear_password: bool,
        options_json: dict[str, Any] | None,
        options_json_set: bool,
    ) -> DBConnectionRecord:
        current = self.get_connection(user_id, connection_id)
        next_type = self.normalize_db_type(db_type) if db_type is not None else current.db_type
        next_host = self.validate_host(host) if host is not None else current.host
        encrypted_secret_blob: str | None = None
        if password is not None and not clear_password:
            encrypted_secret_blob = self._encrypt_secret_payload(
                password=self._normalize_optional_text(password),
            )

        try:
            updated = self.auth_db.update_db_connection(
                user_id,
                connection_id,
                name=self._normalize_name(name) if name is not None else None,
                db_type=next_type if db_type is not None else None,
                host=next_host if host is not None else None,
                port=port,
                database=self._normalize_optional_text(database) if database is not None else None,
                username=self._normalize_optional_text(username) if username is not None else None,
                options_json=self._normalize_options_json(options_json),
                options_json_set=options_json_set,
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="Connection name already exists."
            ) from exc
        if updated is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="DB connection not found.")

        if clear_password:
            self.auth_db.clear_db_connection_secret(connection_id)
        elif encrypted_secret_blob is not None:
            self.auth_db.set_db_connection_secret(connection_id, encrypted_secret_blob)
        return self.get_connection(user_id, connection_id)

    def delete_connection(self, user_id: int, connection_id: str) -> None:
        deleted = self.auth_db.delete_db_connection(user_id, connection_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="DB connection not found.")

    def resolve_connection_for_runtime(
        self, user_id: int, connection_id: str
    ) -> ResolvedDBConnection:
        row = self.get_connection(user_id, connection_id)
        secret_state = self._load_secret_state(user_id, connection_id)
        return ResolvedDBConnection(
            connection_id=row.id,
            user_id=row.user_id,
            name=row.name,
            db_type=row.db_type,
            host=row.host,
            port=row.port,
            database=row.database,
            username=row.username,
            password=secret_state.password,
            options=row.options_json or {},
        )

    def test_connection(self, user_id: int, connection_id: str) -> DBConnectionRecord:
        resolved = self.resolve_connection_for_runtime(user_id, connection_id)
        adapter = build_connection_adapter(
            resolved,
            timeout_sec=self.settings.db_connections_test_timeout_sec,
        )
        try:
            adapter.test_connection()
        except HTTPException:
            raise
        except Exception as exc:
            sanitized = sanitize_error_text(str(exc))
            updated = self.auth_db.update_db_connection_test_status(
                user_id,
                connection_id,
                ok=False,
                error=sanitized,
            )
            if updated is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="DB connection not found."
                ) from None
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=sanitized) from exc

        updated = self.auth_db.update_db_connection_test_status(
            user_id,
            connection_id,
            ok=True,
            error=None,
        )
        if updated is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="DB connection not found.")
        return updated

