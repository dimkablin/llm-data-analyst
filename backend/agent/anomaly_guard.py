"""Heuristic numeric consistency check for final answers and their artifacts."""

from __future__ import annotations

import math
import re
from typing import Any

_NUMBER_RE = re.compile(
    r"(?<![\w])(?P<number>[+-]?(?:\d{1,3}(?:[ \u00a0]\d{3})+|\d+)(?:[.,]\d+)?)"
    r"\s*(?P<scale>тыс(?:\.|яч[аиу]?)?|млн|миллион(?:а|ов)?|млрд|миллиард(?:а|ов)?)?"
    r"\s*(?P<unit>%|₽|руб(?:\.|ля|лей)?|шт(?:\.|ук[аи]?)?)?",
    re.IGNORECASE,
)
_EMBEDDED_NUMBER_RE = re.compile(_NUMBER_RE.pattern.replace(r"(?<![\w])", ""), re.IGNORECASE)
_DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}(?:-\d{2})?|\d{1,2}[./]\d{1,2}[./]\d{2,4})\b")
_STRUCTURAL_PREFIX_RE = re.compile(r"(?:№|id|sku|артикул|код|шаг|пункт|топ)\s*[-:#№]?\s*$", re.IGNORECASE)
_SCALE = {
    "тыс": 1_000.0,
    "млн": 1_000_000.0,
    "миллион": 1_000_000.0,
    "млрд": 1_000_000_000.0,
    "миллиард": 1_000_000_000.0,
}


def _scale_value(label: str | None) -> float:
    normalized = str(label or "").lower().rstrip(".")
    for prefix, value in _SCALE.items():
        if normalized.startswith(prefix):
            return value
    return 1.0


def _is_structural_number(text: str, start: int, end: int, raw_number: str, unit: str) -> bool:
    for match in _DATE_RE.finditer(text):
        if start < match.end() and end > match.start():
            return True

    line_start = text.rfind("\n", 0, start) + 1
    before_on_line = text[line_start:start]
    after = text[end : end + 3]
    if re.fullmatch(r"\s*(?:[-*]>?\s*)?", before_on_line) and re.match(r"[.)]\s", after):
        return True
    prefix = re.sub(r"[*_`~]+", "", text[max(0, start - 24) : start])
    if _STRUCTURAL_PREFIX_RE.search(prefix):
        return True
    if re.match(r"-(?:е|я|й|ое)\b", after, re.IGNORECASE):
        return True

    digits = raw_number.replace(" ", "").replace("\u00a0", "")
    if not unit and digits.isdigit() and len(digits) == 4 and 1900 <= int(digits) <= 2100:
        return True
    return False


def _answer_claims(text: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for match in _NUMBER_RE.finditer(text):
        raw_number = match.group("number")
        unit = str(match.group("unit") or "").lower()
        if _is_structural_number(text, match.start(), match.end(), raw_number, unit):
            continue
        normalized_number = raw_number.replace(" ", "").replace("\u00a0", "").replace(",", ".")
        try:
            base_value = float(normalized_number)
        except ValueError:
            continue
        scale = _scale_value(match.group("scale"))
        value = base_value * scale
        decimals = len(normalized_number.partition(".")[2])
        claims.append(
            {
                "id": f"claim-{len(claims) + 1}",
                "text": match.group(0).strip(),
                "normalized_value": value,
                "unit": "percent" if unit == "%" else unit or None,
                "scale": scale,
                "decimals": decimals,
                "start": match.start(),
                "end": match.end(),
            }
        )
    return claims


def _embedded_numbers(text: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    dates = list(_DATE_RE.finditer(text))
    for match in _EMBEDDED_NUMBER_RE.finditer(text):
        if any(match.start() < date.end() and match.end() > date.start() for date in dates):
            continue
        raw_number = match.group("number")
        normalized = raw_number.replace(" ", "").replace("\u00a0", "").replace(",", ".")
        try:
            value = float(normalized) * _scale_value(match.group("scale"))
        except ValueError:
            continue
        values.append(
            {
                "value": value,
                "percent": match.group("unit") == "%",
                "raw_value": match.group(0).strip(),
            }
        )
    return values


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    numeric_text = r"\s*[+-]?(?:\d+(?:[.,]\d+)?|\d{1,3}(?:[ \u00a0]\d{3})+(?:[.,]\d+)?)\s*"
    if isinstance(value, str) and re.fullmatch(numeric_text, value):
        try:
            number = float(value.strip().replace(" ", "").replace("\u00a0", "").replace(",", "."))
            return number if math.isfinite(number) else None
        except ValueError:
            return None
    return None


def _artifact_scale(label: Any) -> tuple[float, bool]:
    text = str(label or "").lower()
    scale_match = re.search(r"\b(тыс\.?|млн|миллион\w*|млрд|миллиард\w*)\b", text)
    return _scale_value(scale_match.group(1) if scale_match else None), "%" in text


def _artifact_values(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for artifact in artifacts:
        artifact_id = str(artifact.get("id") or "")
        title = str(artifact.get("text") or artifact.get("type") or artifact_id)
        artifact_type = artifact.get("type")
        payload = artifact.get("data", {}).get("data")
        if artifact_type == "table" and isinstance(payload, dict):
            columns = payload.get("columns") if isinstance(payload.get("columns"), list) else []
            rows = payload.get("data") if isinstance(payload.get("data"), list) else []
            index = payload.get("index") if isinstance(payload.get("index"), list) else []
            for row_index, row in enumerate(rows):
                if not isinstance(row, list):
                    continue
                row_label = index[row_index] if row_index < len(index) else row_index + 1
                if isinstance(row_label, list):
                    row_label = " / ".join(map(str, row_label))
                descriptive_cell = next(
                    (
                        raw
                        for raw in row
                        if isinstance(raw, str) and raw.strip() and _numeric(raw) is None
                    ),
                    None,
                )
                if descriptive_cell is not None:
                    row_label = descriptive_cell
                for column_index, raw in enumerate(row):
                    number = _numeric(raw)
                    column = columns[column_index] if column_index < len(columns) else column_index + 1
                    scale, is_percent = _artifact_scale(column)
                    cell_values = (
                        [{"value": number * scale, "percent": is_percent, "raw_value": raw}]
                        if number is not None
                        else _embedded_numbers(raw) if isinstance(raw, str) else []
                    )
                    values.extend(
                        {
                            **cell_value,
                            "artifact_id": artifact_id,
                            "artifact_title": title,
                            "row": str(row_label),
                            "column": str(column),
                        }
                        for cell_value in cell_values
                    )
        elif artifact_type == "plot" and isinstance(payload, dict):
            traces = payload.get("data") if isinstance(payload.get("data"), list) else []
            for trace in traces:
                if not isinstance(trace, dict):
                    continue
                series = str(trace.get("name") or "график")
                labels = trace.get("x") if isinstance(trace.get("x"), list) else []
                points = trace.get("y") if isinstance(trace.get("y"), list) else trace.get("values")
                if not isinstance(points, list):
                    continue
                scale, is_percent = _artifact_scale(series)
                for index, raw in enumerate(points):
                    number = _numeric(raw)
                    if number is None:
                        continue
                    values.append(
                        {
                            "value": number * scale,
                            "percent": is_percent,
                            "artifact_id": artifact_id,
                            "artifact_title": title,
                            "row": str(labels[index]) if index < len(labels) else str(index + 1),
                            "column": series,
                            "raw_value": raw,
                        }
                    )
        elif artifact_type in {"value", "json"}:

            def walk(
                node: Any,
                path: list[str],
                artifact_id: str = artifact_id,
                title: str = title,
            ) -> None:
                number = _numeric(node)
                if number is not None:
                    label = " / ".join(path) or "value"
                    scale, is_percent = _artifact_scale(label)
                    values.append(
                        {
                            "value": number * scale,
                            "percent": is_percent,
                            "artifact_id": artifact_id,
                            "artifact_title": title,
                            "row": None,
                            "column": label,
                            "raw_value": node,
                        }
                    )
                    return
                if isinstance(node, dict):
                    for key, child in node.items():
                        walk(child, [*path, str(key)])
                elif isinstance(node, list):
                    for index, child in enumerate(node):
                        walk(child, [*path, str(index + 1)])

            walk(payload, [])
    return values


def _difference(claim: dict[str, Any], source: dict[str, Any]) -> tuple[float, float]:
    expected = float(claim["normalized_value"])
    candidates = [float(source["value"])]
    if claim["unit"] == "percent" and not source["percent"]:
        candidates.append(float(source["value"]) * 100.0)
    actual = min(candidates, key=lambda value: abs(value - expected))
    return abs(actual - expected), actual


def _matches(claim: dict[str, Any], source: dict[str, Any]) -> tuple[bool, float]:
    difference, _actual = _difference(claim, source)
    expected = float(claim["normalized_value"])
    rounding = 0.5 * float(claim["scale"]) * (10 ** -int(claim["decimals"]))
    tolerance = max(abs(expected) * 0.001, rounding if claim["scale"] > 1 else 0.0, 1e-9)
    percent = (
        0.0
        if expected == 0 and difference == 0
        else (math.inf if expected == 0 else difference / abs(expected) * 100.0)
    )
    return difference <= tolerance, percent


def check_numeric_consistency(
    text: str,
    artifacts: list[dict[str, Any]],
    source_text: str = "",
) -> dict[str, Any]:
    """Return an auditable, non-blocking consistency report for numeric claims."""
    claims = _answer_claims(text)
    sources = _artifact_values(artifacts)
    sources.extend(
        {
            **value,
            "artifact_id": "",
            "artifact_title": "Текст запроса",
            "row": None,
            "column": "текст запроса",
        }
        for value in _embedded_numbers(source_text)
    )
    items: list[dict[str, Any]] = []
    matched_count = 0
    for claim in claims:
        matches: list[dict[str, Any]] = []
        for source in sources:
            matched, difference_percent = _matches(claim, source)
            if matched:
                matches.append({**source, "difference_percent": round(difference_percent, 6)})
        matches.sort(key=lambda item: item["difference_percent"])
        if matches:
            matched_count += 1
        items.append(
            {
                "id": claim["id"],
                "text": claim["text"],
                "normalized_value": claim["normalized_value"],
                "unit": claim["unit"],
                "status": "matched" if matches else "unmatched",
                "sources": matches[:5],
            }
        )

    status = "no_values" if not claims else ("passed" if matched_count == len(claims) else "warning")
    return {
        "status": status,
        "checked": len(claims),
        "matched": matched_count,
        "warnings": len(claims) - matched_count,
        "items": items,
    }
