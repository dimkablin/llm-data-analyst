from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

_SUPPORTED_TABLES = {
    "planfact_plan_raw",
    "planfact_fact_raw",
    "planfact_by_cfo_period",
    "planfact_by_cfo_article_period",
    "planfact_focus_by_cfo_period",
    "planfact_focus_by_cfo_article_period",
    "planfact_fact_monthly",
    "planfact_plan_long",
}
_SQL_IDENTIFIER = r'(?:"[^"]+"|[A-Za-z_]\w*)(?:\.(?:"[^"]+"|[A-Za-z_]\w*))?'
_AGGREGATE_RE = re.compile(
    rf"""^\s*(?P<function>sum|avg|count|min|max)\s*
        \(\s*(?P<column>\*|{_SQL_IDENTIFIER})\s*\)
        (?:\s+(?:as\s+)?(?P<alias>"[^"]+"|[A-Za-z_]\w*))?\s*$""",
    re.IGNORECASE | re.VERBOSE,
)
_COLUMN_RE = re.compile(
    rf"""^\s*(?P<column>{_SQL_IDENTIFIER})
        (?:\s+(?:as\s+)?(?P<alias>"[^"]+"|[A-Za-z_]\w*))?\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class GroupColumn:
    source: str
    output: str


@dataclass(frozen=True)
class AggregateColumn:
    function: str
    source: str
    output: str


@dataclass(frozen=True)
class ValidationSpec:
    table: str
    from_sql: str
    where_sql: str
    groups: list[GroupColumn]
    aggregates: list[AggregateColumn]


def _split_sql_list(value: str) -> list[str]:
    items: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    for index, char in enumerate(value):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            items.append(value[start:index].strip())
            start = index + 1
    items.append(value[start:].strip())
    return [item for item in items if item]


def _clean_identifier(value: str) -> str:
    return str(value or "").strip().split(".")[-1].strip('"')


def parse_validation_sql(sql: str, output_columns: list[str]) -> ValidationSpec | None:
    clean = str(sql or "").strip().rstrip(";").strip()
    if not clean or re.match(r"^with\b", clean, re.IGNORECASE):
        return None
    if re.search(r"\b(join|union|having|qualify)\b", clean, re.IGNORECASE):
        return None

    from_match = re.search(r"\bfrom\b", clean, re.IGNORECASE)
    group_match = re.search(r"\bgroup\s+by\b", clean, re.IGNORECASE)
    if not from_match:
        return None

    tail_matches = [
        match
        for pattern in (r"\bgroup\s+by\b", r"\border\s+by\b", r"\blimit\b", r"\boffset\b")
        if (match := re.search(pattern, clean[from_match.end() :], re.IGNORECASE))
    ]
    from_end = from_match.end() + min((match.start() for match in tail_matches), default=len(clean))
    where_match = re.search(r"\bwhere\b", clean[from_match.end() : from_end], re.IGNORECASE)
    if where_match:
        where_start = from_match.end() + where_match.start()
        table_part = clean[from_match.end() : where_start].strip()
        where_sql = clean[where_start + len(where_match.group(0)) : from_end].strip()
    else:
        table_part = clean[from_match.end() : from_end].strip()
        where_sql = ""

    table_tokens = table_part.replace('"', "").split()
    if not table_tokens:
        return None
    table = _clean_identifier(table_tokens[0]).lower()
    if table not in _SUPPORTED_TABLES or len(table_tokens) > 2:
        return None

    select_match = re.match(r"^\s*select\b", clean, re.IGNORECASE)
    if not select_match:
        return None
    select_sql = clean[select_match.end() : from_match.start()].strip()
    select_sql = re.sub(r"^distinct\b", "", select_sql, count=1, flags=re.IGNORECASE).strip()
    select_items = _split_sql_list(select_sql)
    if select_items == ["*"]:
        select_items = [f'"{column}"' for column in output_columns]
    if len(select_items) != len(output_columns):
        return None

    parsed_columns: list[GroupColumn | AggregateColumn] = []
    for item, output in zip(select_items, output_columns, strict=True):
        aggregate_match = _AGGREGATE_RE.match(item)
        if aggregate_match:
            parsed_columns.append(
                AggregateColumn(
                    function=aggregate_match.group("function").lower(),
                    source=_clean_identifier(aggregate_match.group("column")),
                    output=str(output),
                )
            )
            continue
        column_match = _COLUMN_RE.match(item)
        if not column_match:
            return None
        parsed_columns.append(
            GroupColumn(
                source=_clean_identifier(column_match.group("column")),
                output=str(output),
            )
        )

    groups = [item for item in parsed_columns if isinstance(item, GroupColumn)]
    aggregates = [item for item in parsed_columns if isinstance(item, AggregateColumn)]
    if not groups and not aggregates:
        return None

    if not aggregates:
        return ValidationSpec(
            table=table,
            from_sql=table_part,
            where_sql=where_sql,
            groups=groups,
            aggregates=[],
        )
    if group_match is None:
        if groups:
            return None
        return ValidationSpec(
            table=table,
            from_sql=table_part,
            where_sql=where_sql,
            groups=[],
            aggregates=aggregates,
        )
    group_end_candidates = [
        match.start()
        for pattern in (r"\border\s+by\b", r"\blimit\b", r"\boffset\b")
        if (match := re.search(pattern, clean[group_match.end() :], re.IGNORECASE))
    ]
    group_end = group_match.end() + min(group_end_candidates) if group_end_candidates else len(clean)
    group_items = _split_sql_list(clean[group_match.end() : group_end])

    expected_groups: list[str] = []
    for item in group_items:
        if item.isdigit():
            position = int(item) - 1
            if position < 0 or position >= len(parsed_columns):
                return None
            selected = parsed_columns[position]
            if not isinstance(selected, GroupColumn):
                return None
            expected_groups.append(selected.source.lower())
        else:
            match = _COLUMN_RE.match(item)
            if not match:
                return None
            expected_groups.append(_clean_identifier(match.group("column")).lower())
    if expected_groups != [group.source.lower() for group in groups]:
        return None

    return ValidationSpec(
        table=table,
        from_sql=table_part,
        where_sql=where_sql,
        groups=groups,
        aggregates=aggregates,
    )


def _artifact_sql(artifact: dict[str, Any]) -> str:
    meta = artifact.get("meta") if isinstance(artifact.get("meta"), dict) else {}
    recipes = meta.get("recipe")
    if not isinstance(recipes, list):
        provenance = meta.get("provenance") if isinstance(meta.get("provenance"), dict) else {}
        recipes = provenance.get("recipe")
    for recipe in recipes if isinstance(recipes, list) else []:
        if isinstance(recipe, dict) and str(recipe.get("kind") or "").lower() == "sql":
            return str(recipe.get("code") or recipe.get("sql") or "").strip()
    query = meta.get("query") if isinstance(meta.get("query"), dict) else {}
    return str(query.get("requested_sql") or query.get("executed_sql") or "").strip()


def _artifact_table(artifact: dict[str, Any]) -> tuple[list[str], list[list[Any]]] | None:
    data = artifact.get("data") if isinstance(artifact.get("data"), dict) else {}
    if str(data.get("format") or "").lower() != "split":
        return None
    payload = data.get("export_data") or data.get("data")
    if not isinstance(payload, dict):
        return None
    columns = payload.get("columns")
    rows = payload.get("data")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return None
    return [str(column) for column in columns], [list(row) for row in rows if isinstance(row, list)]


def _match_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _excel_column(index: int) -> str:
    value = index
    result = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _validation_formula(
    *,
    function: str,
    control_row: int,
    detail_column: int | None,
    detail_end_row: int,
) -> str:
    ids = f"'Расчетная детализация'!$A$2:$A${detail_end_row}"
    criterion = f"$A{control_row}"
    if function == "count" and detail_column is None:
        return f"=COUNTIF({ids},{criterion})"
    if detail_column is None:
        raise ValueError("detail_column is required for this aggregation")
    values_letter = _excel_column(detail_column)
    values = f"'Расчетная детализация'!${values_letter}$2:${values_letter}${detail_end_row}"
    if function == "sum":
        return f"=SUMIF({ids},{criterion},{values})"
    if function == "avg":
        return f"=AVERAGEIF({ids},{criterion},{values})"
    if function == "count":
        return f'=COUNTIFS({ids},{criterion},{values},"<>")'
    if function == "min":
        return f"=MINIFS({values},{ids},{criterion})"
    return f"=MAXIFS({values},{ids},{criterion})"


def _direct_validation_formula(*, control_row: int, detail_column: int, detail_end_row: int) -> str:
    ids = f"'Расчетная детализация'!$A$2:$A${detail_end_row}"
    values_letter = _excel_column(detail_column)
    values = f"'Расчетная детализация'!${values_letter}$2:${values_letter}${detail_end_row}"
    return f'=IFERROR(AVERAGEIF({ids},$A{control_row},{values}),"")'


def build_artifact_validation(
    artifact: dict[str, Any],
    *,
    query_dataframe: Callable[[str], pd.DataFrame],
    max_detail_rows: int = 500_000,
) -> dict[str, Any] | None:
    table_payload = _artifact_table(artifact)
    if table_payload is None:
        return None
    result_columns, result_rows = table_payload
    spec = parse_validation_sql(_artifact_sql(artifact), result_columns)
    if spec is None:
        return None

    where = f" WHERE {spec.where_sql}" if spec.where_sql else ""
    detail = query_dataframe(f"SELECT * FROM {spec.from_sql}{where} LIMIT {max_detail_rows + 1}")
    if len(detail) > max_detail_rows:
        raise ValueError(f"Расшифровка содержит больше {max_detail_rows} строк; уточните фильтры запроса.")
    if any(group.source not in detail.columns for group in spec.groups):
        return None

    result_indices = {column: index for index, column in enumerate(result_columns)}
    result_keys: dict[tuple[Any, ...], list[int]] = {}
    for result_id, row in enumerate(result_rows, start=1):
        key = tuple(_match_value(row[result_indices[group.output]]) for group in spec.groups)
        result_keys.setdefault(key, []).append(result_id)

    matched_rows: list[list[Any]] = []
    plan_links: dict[int, set[int]] = {}
    fact_links: dict[int, set[int]] = {}
    display_columns = [
        column for column in detail.columns if column not in {"plan_source_row_ids", "fact_source_row_ids"}
    ]
    for _, row in detail.iterrows():
        key = tuple(_match_value(row[group.source]) for group in spec.groups)
        result_ids = result_keys.get(key)
        if not result_ids:
            continue
        for result_id in result_ids:
            matched_rows.append([result_id, *[row[column] for column in display_columns]])
            for column, links in (
                ("plan_source_row_ids", plan_links),
                ("fact_source_row_ids", fact_links),
                ("plan_source_row_id", plan_links),
                ("fact_source_row_id", fact_links),
            ):
                values = row.get(column)
                if hasattr(values, "tolist"):
                    values = values.tolist()
                if column.endswith("_row_ids") and isinstance(values, str):
                    values = values.split(",")
                if not isinstance(values, list | tuple | set):
                    values = [values]
                for value in values:
                    try:
                        links.setdefault(int(value), set()).add(result_id)
                    except (TypeError, ValueError):
                        continue

    detail_columns = ["Строка результата", *[str(column) for column in display_columns]]
    detail_index = {column: index + 1 for index, column in enumerate(detail_columns)}
    detail_end = max(2, len(matched_rows) + 1)
    control_groups = spec.groups if spec.aggregates else []
    control_columns = [
        "Строка результата",
        *[group.output for group in control_groups],
        "Показатель",
        "Значение результата",
        "Проверка Excel",
        "Разница",
        "Статус",
    ]
    control_rows: list[list[Any]] = []
    controls: list[AggregateColumn] = spec.aggregates
    if not controls:
        controls = [
            AggregateColumn(function="direct", source=group.source, output=group.output)
            for group in spec.groups
            if group.source in detail.columns
            and any(
                isinstance(row[result_indices[group.output]], int | float)
                and not isinstance(row[result_indices[group.output]], bool)
                for row in result_rows
            )
        ]
    for result_id, result_row in enumerate(result_rows, start=1):
        for aggregate in controls:
            if aggregate.source != "*" and aggregate.source not in detail_index:
                return None
            excel_row = len(control_rows) + 2
            result_value = result_row[result_indices[aggregate.output]]
            formula = (
                _direct_validation_formula(
                    control_row=excel_row,
                    detail_column=detail_index[aggregate.source],
                    detail_end_row=detail_end,
                )
                if aggregate.function == "direct"
                else _validation_formula(
                    function=aggregate.function,
                    control_row=excel_row,
                    detail_column=(None if aggregate.source == "*" else detail_index[aggregate.source]),
                    detail_end_row=detail_end,
                )
            )
            check_column = _excel_column(len(control_columns) - 2)
            result_column = _excel_column(len(control_columns) - 3)
            difference_column = _excel_column(len(control_columns) - 1)
            control_rows.append(
                [
                    result_id,
                    *[result_row[result_indices[group.output]] for group in control_groups],
                    aggregate.output,
                    result_value,
                    formula,
                    (
                        f'=IF(OR({check_column}{excel_row}="",'
                        f'{result_column}{excel_row}=""),"",'
                        f"{check_column}{excel_row}-{result_column}{excel_row})"
                    ),
                    (
                        f'=IF({difference_column}{excel_row}="","НЕ ПРОВЕРЕНО",'
                        f"IF(ABS({difference_column}{excel_row})<=0.01,"
                        '"OK","РАСХОЖДЕНИЕ"))'
                    ),
                ]
            )

    return {
        "tables": {
            "Контроль": {"columns": control_columns, "rows": control_rows},
            "Расчетная детализация": {
                "columns": detail_columns,
                "rows": matched_rows,
            },
        },
        "plan_links": plan_links,
        "fact_links": fact_links,
    }
