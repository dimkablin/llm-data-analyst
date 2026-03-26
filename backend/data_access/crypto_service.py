from __future__ import annotations

import json
from base64 import urlsafe_b64decode
from typing import Any

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from backend.core.config import Settings


class SecretCryptoError(RuntimeError):
    pass


def _validate_key(raw_key: str) -> bytes:
    encoded = raw_key.encode("utf-8")
    try:
        urlsafe_b64decode(encoded)
    except Exception as exc:  # noqa: BLE001
        raise SecretCryptoError("Invalid DB connections encryption key format.") from exc
    return encoded


class SecretCryptoService:
    def __init__(self, settings: Settings) -> None:
        current = (
            settings.db_connections_encryption_key_current
            or settings.db_connections_encryption_key
        ).strip()
        old_keys = [
            item.strip()
            for item in settings.db_connections_encryption_keys_old.split(",")
            if item.strip()
        ]
        self._current_key = current
        if not current:
            self._current = None
            self._multi = None
            return

        fernet_keys = [_validate_key(current), *(_validate_key(key) for key in old_keys)]
        fernets = [Fernet(key) for key in fernet_keys]
        self._current = fernets[0]
        self._multi = MultiFernet(fernets)

    def encrypt_payload(self, payload: dict[str, Any]) -> str:
        if self._current is None:
            raise SecretCryptoError(
                "DB_CONNECTIONS_ENCRYPTION_KEY_CURRENT (or DB_CONNECTIONS_ENCRYPTION_KEY) is required."
            )
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return self._current.encrypt(raw).decode("utf-8")

    def decrypt_payload(self, ciphertext: str) -> dict[str, Any]:
        if not ciphertext:
            return {}
        if self._multi is None:
            raise SecretCryptoError(
                "DB_CONNECTIONS_ENCRYPTION_KEY_CURRENT (or DB_CONNECTIONS_ENCRYPTION_KEY) is required."
            )
        try:
            raw = self._multi.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
        except InvalidToken as exc:
            raise SecretCryptoError("Unable to decrypt DB connection secret with configured keys.") from exc

        try:
            payload = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            raise SecretCryptoError("Stored DB connection secret payload is invalid.") from exc
        if not isinstance(payload, dict):
            raise SecretCryptoError("Stored DB connection secret payload must be an object.")
        return payload


