from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig


class PredictIntegrationError(RuntimeError):
    pass


def clean_str(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def ensure_scheme(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return value
    if "://" not in value:
        value = "http://" + value
    return value


def strip_path_keep_origin(url: str) -> str:
    parsed = urlparse(ensure_scheme(url).rstrip("/"))
    if not parsed.hostname:
        raise PredictIntegrationError(f"Bad URL: {url}")
    scheme = parsed.scheme or "http"
    port = parsed.port or (443 if scheme == "https" else 80)
    return f"{scheme}://{parsed.hostname}:{int(port)}"


def build_llm_payload(
    *,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
) -> dict[str, str]:
    base_url = clean_str(llm_base_url)
    api_key = clean_str(llm_api_key)
    model = clean_str(llm_model)
    if not base_url or not api_key or not model:
        raise PredictIntegrationError(
            "Predict integration requires llm_base_url, llm_api_key and llm_model."
        )
    return {
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "model": model,
    }


def build_db_payload(
    *,
    db_runtime_config: RuntimeDBConnectionConfig | None,
    csv_session_id: str | None,
    backend_api_url: str,
) -> dict[str, Any]:
    if csv_session_id:
        origin = strip_path_keep_origin(backend_api_url)
        parsed = urlparse(origin)
        return {
            "host": origin,
            "port": int(parsed.port or (443 if parsed.scheme == "https" else 80)),
            "database": "duckdb",
            "user": "",
            "password": "",
            "schema": str(csv_session_id).strip(),
            "table_prefix": "",
        }

    if db_runtime_config is None:
        raise PredictIntegrationError("No DB source or CSV session is attached.")

    schema = str(db_runtime_config.options.get("schema") or "public").strip()

    return {
        "host": str(db_runtime_config.host or ""),
        "port": int(db_runtime_config.port or 5432),
        "database": str(db_runtime_config.database or ""),
        "user": str(db_runtime_config.username or ""),
        "password": str(db_runtime_config.password or ""),
        "schema": schema,
        "table_prefix": "",
    }


def post_json(url: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            raw_body = response.read()
    except HTTPError as exc:
        body_preview = exc.read().decode("utf-8", errors="replace")[:1000]
        raise PredictIntegrationError(
            f"Predict backend returned HTTP {exc.code}: {body_preview}"
        ) from exc
    except URLError as exc:
        raise PredictIntegrationError(f"Predict backend is unavailable: {exc.reason}") from exc
    except TimeoutError as exc:
        raise PredictIntegrationError("Predict backend request timed out.") from exc

    if not raw_body:
        return {}

    decoded = json.loads(raw_body.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise PredictIntegrationError("Predict backend returned non-object JSON payload.")
    return decoded
