from __future__ import annotations

import copy
import json
from typing import Any


PROVENANCE_SCHEMA_VERSION = "1.0"
RECIPE_SCHEMA_VERSION = "1.0"
RESERVED_ARTIFACT_HINT_KEYS = ("meta", "recipe", "source", "provenance")
RECIPE_KIND_ALIASES = {
    "step": "python",
    "py": "python",
    "code": "python",
    "metadata": "db_metadata",
    "db": "db_metadata",
    "search": "source_query",
    "external_query": "source_query",
    "forecast": "model_inference",
    "inference": "model_inference",
    "model": "model_inference",
    "plot": "chart",
    "plotly": "chart",
    "figure": "chart",
}
RECIPE_KINDS = {"sql", "python", "db_metadata", "source_query", "model_inference", "chart"}


def _clean_str(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _normalize_recipe_kind(value: object) -> str:
    clean = _clean_str(value) or "python"
    normalized = RECIPE_KIND_ALIASES.get(clean.lower(), clean.lower())
    if normalized not in RECIPE_KINDS:
        return "python"
    return normalized


def _default_recipe_title(kind: str) -> str:
    if kind == "sql":
        return "Executed SQL"
    if kind == "db_metadata":
        return "DB metadata"
    if kind == "source_query":
        return "Source query"
    if kind == "model_inference":
        return "Model inference"
    if kind == "chart":
        return "Chart step"
    return "Tool code"


def _default_recipe_language(kind: str) -> str | None:
    if kind == "sql":
        return "sql"
    if kind == "python":
        return "python"
    return None


def _normalize_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _normalize_list_str(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        clean = _clean_str(raw)
        return [clean] if clean else []
    if not isinstance(raw, (list, tuple, set)):
        return []
    items: list[str] = []
    for item in raw:
        clean = _clean_str(item)
        if clean:
            items.append(clean)
    return items


def normalize_source_ref(raw: object) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None

    source_type = _clean_str(
        raw.get("source_type") or raw.get("type") or raw.get("kind")
    )
    source_ref_id = _clean_str(
        raw.get("source_ref_id") or raw.get("ref_id") or raw.get("id")
    )
    source_label = _clean_str(
        raw.get("source_label") or raw.get("label") or raw.get("name")
    )
    source_mode = _clean_str(raw.get("source_mode") or raw.get("mode"))

    if not any((source_type, source_ref_id, source_label, source_mode)):
        return None

    payload: dict[str, str] = {}
    if source_type:
        payload["source_type"] = source_type
    if source_ref_id:
        payload["source_ref_id"] = source_ref_id
    if source_label:
        payload["source_label"] = source_label
    if source_mode:
        payload["source_mode"] = source_mode
    return payload


def merge_source_refs(*candidates: object) -> dict[str, str] | None:
    merged: dict[str, str] = {}
    for candidate in candidates:
        normalized = normalize_source_ref(candidate)
        if not normalized:
            continue
        merged.update(normalized)
    return merged or None


def build_recipe_step(
    *,
    kind: str,
    title: str | None = None,
    tool_name: str | None = None,
    code: str | None = None,
    summary: str | None = None,
    language: str | None = None,
    depends_on: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    normalized_kind = _normalize_recipe_kind(kind)
    step: dict[str, Any] = {
        "kind": normalized_kind,
        "title": _clean_str(title) or _default_recipe_title(normalized_kind),
    }
    clean_tool_name = _clean_str(tool_name)
    if clean_tool_name:
        step["tool_name"] = clean_tool_name

    clean_code = _clean_str(code)
    if clean_code:
        step["code"] = clean_code

    clean_summary = _clean_str(summary)
    if clean_summary:
        step["summary"] = clean_summary

    clean_language = _clean_str(language) or _default_recipe_language(normalized_kind)
    if clean_language:
        step["language"] = clean_language

    normalized_depends_on = _normalize_list_str(depends_on)
    if normalized_depends_on:
        step["depends_on"] = normalized_depends_on
    return step


def build_sql_recipe_step(
    *,
    sql: str,
    tool_name: str | None = None,
    summary: str | None = None,
    title: str | None = None,
    depends_on: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return build_recipe_step(
        kind="sql",
        title=title or "Executed SQL",
        tool_name=tool_name,
        code=sql,
        summary=summary,
        language="sql",
        depends_on=depends_on,
    )


def build_db_metadata_recipe_step(
    *,
    action: str | None = None,
    tool_name: str | None = None,
    summary: str | None = None,
    title: str | None = None,
    depends_on: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    clean_action = _clean_str(action)
    resolved_summary = _clean_str(summary)
    if clean_action and not resolved_summary:
        resolved_summary = clean_action
    return build_recipe_step(
        kind="db_metadata",
        title=title or "DB metadata",
        tool_name=tool_name,
        summary=resolved_summary,
        depends_on=depends_on,
    )


def build_chart_recipe_step(
    *,
    tool_name: str | None = None,
    summary: str | None = None,
    title: str | None = None,
    depends_on: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return build_recipe_step(
        kind="chart",
        title=title or "Chart step",
        tool_name=tool_name,
        summary=summary,
        depends_on=depends_on,
    )


def build_source_query_recipe_step(
    *,
    query: str,
    source_type: str,
    tool_name: str | None = None,
    summary: str | None = None,
    title: str | None = None,
    params: dict[str, Any] | None = None,
    result_count: int | None = None,
    depends_on: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    step = build_recipe_step(
        kind="source_query",
        title=title or "Source query",
        tool_name=tool_name,
        summary=summary,
        depends_on=depends_on,
    )
    clean_query = _clean_str(query)
    if clean_query:
        step["query_text"] = clean_query
    clean_source_type = _clean_str(source_type)
    if clean_source_type:
        step["source_type"] = clean_source_type
    if isinstance(params, dict) and params:
        step["params"] = copy.deepcopy(params)
    normalized_result_count = _normalize_int(result_count)
    if normalized_result_count is not None:
        step["result_count"] = normalized_result_count
    return step


def build_model_inference_recipe_step(
    *,
    source_type: str,
    tool_name: str | None = None,
    title: str | None = None,
    summary: str | None = None,
    model_name: str | None = None,
    params: dict[str, Any] | None = None,
    result_count: int | None = None,
    depends_on: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    step = build_recipe_step(
        kind="model_inference",
        title=title or "Model inference",
        tool_name=tool_name,
        summary=summary,
        depends_on=depends_on,
    )
    clean_source_type = _clean_str(source_type)
    if clean_source_type:
        step["source_type"] = clean_source_type
    clean_model_name = _clean_str(model_name)
    if clean_model_name:
        step["model_name"] = clean_model_name
    if isinstance(params, dict) and params:
        step["params"] = copy.deepcopy(params)
    normalized_result_count = _normalize_int(result_count)
    if normalized_result_count is not None:
        step["result_count"] = normalized_result_count
    return step


def normalize_recipe_steps(raw: object) -> list[dict[str, Any]]:
    if raw is None:
        return []

    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, (list, tuple)):
        candidates = list(raw)
    else:
        return []

    steps: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue

        kind = _normalize_recipe_kind(
            candidate.get("kind")
            or candidate.get("type")
            or candidate.get("step_type")
        )
        title = _clean_str(candidate.get("title") or candidate.get("name"))
        tool_name = _clean_str(candidate.get("tool_name") or candidate.get("tool"))
        language = _clean_str(candidate.get("language") or candidate.get("lang"))
        summary = _clean_str(
            candidate.get("summary") or candidate.get("description")
        )
        depends_on = _normalize_list_str(candidate.get("depends_on"))
        query_text = _clean_str(
            candidate.get("query_text") or candidate.get("query")
        )
        source_type = _clean_str(candidate.get("source_type"))
        model_name = _clean_str(candidate.get("model_name"))
        params = candidate.get("params") if isinstance(candidate.get("params"), dict) else None
        result_count = _normalize_int(candidate.get("result_count"))

        code = _clean_str(candidate.get("code"))
        if code is None:
            sql = _clean_str(candidate.get("sql"))
            if sql:
                code = sql
                language = language or "sql"
                kind = "sql"

        step = build_recipe_step(
            kind=kind,
            title=title,
            tool_name=tool_name,
            code=code,
            summary=summary,
            language=language,
            depends_on=depends_on,
        )
        step["index"] = _normalize_int(candidate.get("index")) or index
        if kind == "source_query":
            if query_text:
                step["query_text"] = query_text
            if source_type:
                step["source_type"] = source_type
            if params:
                step["params"] = copy.deepcopy(params)
            if result_count is not None:
                step["result_count"] = result_count
        if kind == "model_inference":
            if source_type:
                step["source_type"] = source_type
            if model_name:
                step["model_name"] = model_name
            if params:
                step["params"] = copy.deepcopy(params)
            if result_count is not None:
                step["result_count"] = result_count

        if kind in {"sql", "python"} and "code" not in step:
            continue
        if kind == "db_metadata" and "summary" not in step and "title" not in step:
            continue
        steps.append(step)
    return steps


def build_python_recipe_step(
    *,
    tool_name: str | None = None,
    code: str | None = None,
) -> dict[str, Any] | None:
    clean_code = _clean_str(code)
    if not clean_code:
        return None

    return build_recipe_step(
        kind="python",
        title="Tool code",
        tool_name=tool_name,
        code=clean_code,
        language="python",
    )


def extract_artifact_hints(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    hints: dict[str, Any] = {}
    for key in RESERVED_ARTIFACT_HINT_KEYS:
        value = payload.get(key)
        if value is None:
            continue
        if key == "meta" and not isinstance(value, dict):
            continue
        if key in {"source", "provenance"} and not isinstance(value, dict):
            continue
        if key == "recipe" and not isinstance(value, (dict, list, tuple)):
            continue
        hints[key] = copy.deepcopy(value)
    return hints


def _dedupe_recipe_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, step in enumerate(steps, start=1):
        candidate = dict(step)
        candidate["index"] = index
        signature = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        normalized.append(candidate)
    return normalized


def _compat_code_from_recipe(steps: list[dict[str, Any]]) -> str | None:
    for step in steps:
        if step.get("kind") == "python":
            return _clean_str(step.get("code"))
    return None


def _inject_python_step(
    steps: list[dict[str, Any]],
    python_step: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if python_step is None:
        return list(steps)
    if any(step.get("kind") == "python" for step in steps):
        return list(steps)

    normalized_steps = list(steps)
    for index, step in enumerate(normalized_steps):
        if step.get("kind") == "chart":
            normalized_steps.insert(index, python_step)
            return normalized_steps
    normalized_steps.append(python_step)
    return normalized_steps


def build_artifact_meta(
    *,
    base_meta: dict[str, Any] | None = None,
    tool_name: str | None = None,
    tool_code: str | None = None,
    source_context: dict[str, Any] | None = None,
    artifact_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = copy.deepcopy(base_meta or {})
    hints = artifact_hints or {}

    hinted_meta = hints.get("meta")
    if isinstance(hinted_meta, dict):
        meta.update(copy.deepcopy(hinted_meta))

    clean_tool_name = _clean_str(
        meta.get("tool_name")
        or (hints.get("provenance") or {}).get("tool_name")
        or tool_name
    )
    compat_code_seed = _clean_str(meta.get("code") or tool_code)
    if clean_tool_name:
        meta["tool_name"] = clean_tool_name

    hint_provenance = hints.get("provenance")
    provenance_source = None
    provenance_recipe = None
    if isinstance(hint_provenance, dict):
        provenance_source = hint_provenance.get("source")
        provenance_recipe = hint_provenance.get("recipe")

    source = merge_source_refs(
        source_context,
        meta.get("source"),
        provenance_source,
        hints.get("source"),
    )
    if source:
        meta["source"] = source

    recipe: list[dict[str, Any]] = []
    recipe.extend(normalize_recipe_steps(meta.get("recipe")))
    recipe.extend(normalize_recipe_steps(hints.get("recipe")))
    recipe.extend(normalize_recipe_steps(provenance_recipe))

    if _compat_code_from_recipe(recipe) is None:
        python_step = build_python_recipe_step(
            tool_name=clean_tool_name,
            code=compat_code_seed,
        )
    else:
        python_step = None
    recipe = _inject_python_step(recipe, python_step)

    recipe = _dedupe_recipe_steps(recipe)
    if recipe:
        meta["recipe"] = recipe

    compat_code = _compat_code_from_recipe(recipe) or compat_code_seed
    if compat_code:
        meta["code"] = compat_code
    else:
        meta.pop("code", None)

    provenance: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "recipe_schema_version": RECIPE_SCHEMA_VERSION,
    }
    if clean_tool_name:
        provenance["tool_name"] = clean_tool_name
    if source:
        provenance["source"] = source
    if recipe:
        provenance["recipe"] = recipe
        provenance["step_count"] = len(recipe)

    hinted_provenance_meta = {
        key: value
        for key, value in (hint_provenance or {}).items()
        if key
        not in {
            "schema_version",
            "recipe_schema_version",
            "tool_name",
            "source",
            "recipe",
            "step_count",
        }
    }
    if hinted_provenance_meta:
        provenance.update(copy.deepcopy(hinted_provenance_meta))

    meta["provenance"] = provenance
    return meta


