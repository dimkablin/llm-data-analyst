from __future__ import annotations

import copy
from typing import Any, Mapping


def _clean_str(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def build_source_status(*, enabled: bool, available: bool) -> str:
    if available:
        return "available"
    if enabled:
        return "misconfigured"
    return "disabled"


def build_source_descriptor(
    *,
    source_type: str,
    source_ref_id: str | None,
    source_label: str,
    display_name_ru: str | None = None,
    source_mode: str | None,
    enabled: bool,
    available: bool,
    description: str | None,
    description_ru: str | None = None,
    capabilities: list[str] | tuple[str, ...],
    requires_session_data: bool,
    timeout_hint_sec: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_type": source_type,
        "source_ref_id": source_ref_id,
        "source_label": source_label,
        "display_name_ru": _clean_str(display_name_ru),
        "source_mode": source_mode,
        "enabled": enabled,
        "available": available,
        "status": build_source_status(enabled=enabled, available=available),
        "description": _clean_str(description),
        "description_ru": _clean_str(description_ru),
        "capabilities": list(capabilities),
        "requires_session_data": requires_session_data,
    }
    if timeout_hint_sec is not None:
        payload["timeout_hint_sec"] = float(timeout_hint_sec)
    return payload


def build_operation_meta(
    *,
    status: str | None,
    warnings: list[str] | tuple[str, ...] | None,
    request_params: Mapping[str, Any] | None,
    timeout_sec: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": _clean_str(status) or "completed",
        "warnings": [str(item) for item in (warnings or []) if str(item).strip()],
        "request_params": copy.deepcopy(dict(request_params or {})),
    }
    if timeout_sec is not None:
        payload["timeout_sec"] = float(timeout_sec)
    if extra:
        payload.update(copy.deepcopy(dict(extra)))
    return payload
