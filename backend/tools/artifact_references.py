from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from backend.tools.sandbox import is_user_variable_name

ARTIFACT_REFERENCE_KEY = "$artifact"
QUERY_META_ATTR = "llm_data_analyst.query"
EXECUTION_ARTIFACT_ATTR = "llm_data_analyst.execution_artifact"

_MAX_ARTIFACT_ROWS = 1_000
_MAX_ARTIFACT_CELLS = 12_000
_MAX_ARGUMENT_BYTES = 1_000_000


def attach_query_metadata(
    dataframe: pd.DataFrame,
    query_metadata: Mapping[str, Any],
) -> None:
    dataframe.attrs[QUERY_META_ATTR] = dict(query_metadata)


def load_artifact_dataframe(
    artifact_id: str,
    *,
    session_id: str,
    session_store: Any,
    execution_store: Any | None = None,
) -> tuple[pd.DataFrame, Any]:
    """Load one session-owned execution artifact by stable ID."""
    clean_id = str(artifact_id or "").strip()
    if not clean_id:
        raise ValueError("artifact_id is required.")

    artifact, _missing = _restore_execution_artifact(
        clean_id,
        session_id=session_id,
        session_store=session_store,
        execution_store=execution_store,
        visiting=set(),
    )
    if artifact is None:
        location = "this runtime" if session_store is None else "the current session"
        raise ValueError(f"Artifact '{clean_id}' was not found in {location}.")

    dataframe = getattr(artifact, "data", None)
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError(f"Artifact '{clean_id}' is not a DataFrame.")
    restored = dataframe.copy()
    restored.attrs[EXECUTION_ARTIFACT_ATTR] = {
        "artifact_id": str(artifact.id),
        "content_hash": str(artifact.content_hash or ""),
        "name": str(artifact.name or ""),
    }
    return restored, artifact


def _restore_execution_artifact(
    artifact_id: str,
    *,
    session_id: str,
    session_store: Any,
    execution_store: Any | None,
    visiting: set[str],
) -> tuple[Any | None, set[str]]:
    if artifact_id in visiting:
        return None, {artifact_id}

    artifact = execution_store.get(artifact_id) if execution_store is not None else None
    if artifact is None:
        if session_store is None:
            return None, {artifact_id}
        payload = session_store.get_serialized_artifact(session_id, artifact_id)
        if payload is None:
            return None, {artifact_id}
        from backend.artifacts.bridge import execution_from_api_payload

        artifact = execution_from_api_payload(payload, session_id=session_id)

    visiting.add(artifact_id)
    missing: set[str] = set()
    for parent_id in artifact.parent_ids:
        _parent, unresolved = _restore_execution_artifact(
            parent_id,
            session_id=session_id,
            session_store=session_store,
            execution_store=execution_store,
            visiting=visiting,
        )
        missing.update(unresolved)
    visiting.remove(artifact_id)

    if missing:
        artifact.meta = dict(artifact.meta or {})
        artifact.meta["lineage_incomplete"] = {"missing_parent_ids": sorted(missing)}
    if execution_store is not None and execution_store.get(artifact.id) is None:
        artifact = execution_store.put(artifact)
    return artifact, missing


def materialize_artifact_inputs(
    inputs: Mapping[str, str] | None,
    *,
    session_id: str,
    session_store: Any,
    execution_store: Any | None,
    sandbox: Any,
) -> list[str]:
    """Load explicitly referenced artifacts into the current worker sandbox."""
    requested = dict(inputs or {})
    for raw_alias in requested:
        alias = str(raw_alias or "").strip()
        if not is_user_variable_name(alias):
            raise ValueError(f"Artifact alias '{alias}' is not a valid sandbox variable.")

    resolved: list[tuple[str, pd.DataFrame, Any]] = []
    current_scope = sandbox.get_user_scope()
    for raw_alias, raw_artifact_id in requested.items():
        alias = str(raw_alias or "").strip()
        artifact_id = str(raw_artifact_id or "").strip()
        if artifact_id == alias and isinstance(current_scope.get(alias), pd.DataFrame):
            continue
        dataframe, artifact = load_artifact_dataframe(
            artifact_id,
            session_id=session_id,
            session_store=session_store,
            execution_store=execution_store,
        )
        resolved.append((alias, dataframe, artifact))

    materialized: list[str] = []
    for alias, dataframe, artifact in resolved:
        sandbox.put(alias, dataframe)
        materialized.append(str(artifact.id))
    return materialized


def resolve_artifact_references(
    value: Any,
    *,
    artifacts: Mapping[str, Any],
) -> Any:
    if not _contains_artifact_reference(value):
        return value

    resolved = _resolve(value, artifacts=artifacts)
    encoded = json.dumps(
        resolved,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_ARGUMENT_BYTES:
        raise ValueError(f"Expanded MCP arguments exceed {_MAX_ARGUMENT_BYTES} bytes.")
    return resolved


def artifact_reference_names(value: Any) -> list[str]:
    names: list[str] = []

    def collect(item: Any) -> None:
        if isinstance(item, dict):
            if ARTIFACT_REFERENCE_KEY in item:
                name = str(item.get(ARTIFACT_REFERENCE_KEY) or "").strip()
                if name and name not in names:
                    names.append(name)
                return
            for nested in item.values():
                collect(nested)
        elif isinstance(item, list):
            for nested in item:
                collect(nested)

    collect(value)
    return names


def _contains_artifact_reference(value: Any) -> bool:
    if isinstance(value, dict):
        return ARTIFACT_REFERENCE_KEY in value or any(
            _contains_artifact_reference(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_artifact_reference(item) for item in value)
    return False


def _resolve(value: Any, *, artifacts: Mapping[str, Any]) -> Any:
    if isinstance(value, dict):
        if ARTIFACT_REFERENCE_KEY in value:
            if set(value) != {ARTIFACT_REFERENCE_KEY}:
                raise ValueError(f"{ARTIFACT_REFERENCE_KEY} must be the only artifact reference field.")
            return _dataframe_records(_artifact(value[ARTIFACT_REFERENCE_KEY], artifacts=artifacts))
        return {str(key): _resolve(item, artifacts=artifacts) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, artifacts=artifacts) for item in value]
    return value


def _artifact(name: Any, *, artifacts: Mapping[str, Any]) -> pd.DataFrame:
    clean_name = str(name or "").strip()
    if not clean_name or clean_name not in artifacts:
        raise ValueError(f"Sandbox artifact '{clean_name}' was not found.")

    artifact = artifacts[clean_name]
    if not isinstance(artifact, pd.DataFrame):
        raise ValueError(f"Sandbox artifact '{clean_name}' is not a DataFrame.")

    query_metadata = artifact.attrs.get(QUERY_META_ATTR)
    if isinstance(query_metadata, Mapping) and query_metadata.get("truncated"):
        raise ValueError(
            f"Sandbox artifact '{clean_name}' was truncated; aggregate the query before sending it to MCP."
        )
    if len(artifact) > _MAX_ARTIFACT_ROWS:
        raise ValueError(f"Sandbox artifact '{clean_name}' exceeds {_MAX_ARTIFACT_ROWS} rows.")
    if artifact.size > _MAX_ARTIFACT_CELLS:
        raise ValueError(f"Sandbox artifact '{clean_name}' exceeds {_MAX_ARTIFACT_CELLS} cells.")
    return artifact


def _dataframe_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    encoded = dataframe.to_json(
        orient="records",
        date_format="iso",
        date_unit="ms",
        force_ascii=False,
    )
    records = json.loads(encoded)
    if not isinstance(records, list):
        raise ValueError("DataFrame artifact could not be serialized as records.")
    return records
