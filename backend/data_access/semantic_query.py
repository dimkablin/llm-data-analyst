from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticDimension,
    SemanticFact,
    SemanticMetric,
)
from backend.data_access.semantic_validator import relationship_safety_error


class SemanticQueryFilter(BaseModel):
    field: str
    op: Literal["=", "!=", ">", ">=", "<", "<=", "in"]
    value: str | int | float | list[str | int | float]


class SemanticQueryOrder(BaseModel):
    field: str
    direction: Literal["asc", "desc"] = "desc"


class SemanticQuery(BaseModel):
    metrics: list[str] = Field(default_factory=list, min_length=1)
    dimensions: list[str] = Field(default_factory=list)
    time_dimension: str | None = None
    time_grain: Literal["day", "week", "month", "quarter", "year"] | None = None
    filters: list[SemanticQueryFilter] = Field(default_factory=list)
    order_by: list[SemanticQueryOrder] = Field(default_factory=list)
    limit: int = Field(default=100, ge=1, le=1000)


class SemanticQueryCompiler:
    def __init__(self, catalog: SemanticCatalog, *, dialect: str = "duckdb") -> None:
        self.catalog = catalog
        self.dialect = dialect
        self.metrics = {metric.key: metric for metric in catalog.metrics if metric.is_active}
        self.dimensions = {dimension.name: dimension for dimension in catalog.dimensions if dimension.is_active}
        self.facts = {fact.name: fact for fact in catalog.facts}

    def compile(self, query: SemanticQuery) -> str:
        metrics = [self._metric(key) for key in query.metrics]
        metric_base_tables = {metric.base_table for metric in metrics}
        if len(metric_base_tables) != 1:
            raise ValueError("Only one metric base table is supported")
        base_table = next(iter(metric_base_tables))
        dimensions = [self._dimension(name) for name in query.dimensions]
        if query.time_dimension and query.time_dimension not in query.dimensions:
            dimensions.append(self._dimension(query.time_dimension))
        joins = self._joins_for_dimensions(base_table, dimensions)
        aliases = self._aliases(base_table, joins)
        qualified = bool(joins)

        select_parts: list[str] = []
        group_parts: list[str] = []
        for dimension in dimensions:
            expr = self._dimension_expr(
                dimension,
                query.time_grain if dimension.name == query.time_dimension else None,
                aliases if qualified else None,
            )
            select_parts.append(f'{expr} AS {self._quote(dimension.name)}')
            group_parts.append(expr)
        for metric in metrics:
            select_parts.append(f'{self._metric_expr(metric, aliases if qualified else None)} AS {self._quote(metric.key)}')

        sql = f"SELECT {', '.join(select_parts)} FROM {self._from_clause(base_table, joins, aliases)}"
        where = self._where(query.filters, aliases if qualified else None)
        if where:
            sql += f" WHERE {where}"
        if group_parts:
            sql += f" GROUP BY {', '.join(group_parts)}"
        order_by = self._order_by(query.order_by, metrics)
        if order_by:
            sql += f" ORDER BY {order_by}"
        sql += f" LIMIT {int(query.limit)}"
        return sql

    def _metric(self, key: str) -> SemanticMetric:
        metric = self.metrics.get(key)
        if metric is None:
            raise ValueError(f"Unknown semantic metric: {key}")
        return metric

    def _dimension(self, name: str) -> SemanticDimension:
        dimension = self.dimensions.get(name)
        if dimension is None:
            raise ValueError(f"Unknown semantic dimension: {name}")
        return dimension

    def _metric_expr(self, metric: SemanticMetric, aliases: dict[str, str] | None = None) -> str:
        if metric.type == "simple":
            return self._simple_metric_expr(metric, aliases)
        if metric.type == "ratio":
            numerator = self._metric_expr(self._metric(str(metric.numerator)), aliases)
            denominator = self._metric_expr(self._metric(str(metric.denominator)), aliases)
            return f"{numerator} / NULLIF({denominator}, 0)"
        if metric.type == "derived":
            return self._derived_metric_expr(metric, aliases)
        raise ValueError(f"Unsupported semantic metric type: {metric.type}")

    def _simple_metric_expr(self, metric: SemanticMetric, aliases: dict[str, str] | None) -> str:
        column = self._column_ref(metric.base_table, str(metric.expr), aliases)
        if metric.agg == "count_distinct":
            return f"COUNT(DISTINCT {column})"
        return f"{str(metric.agg).upper()}({column})"

    def _derived_metric_expr(self, metric: SemanticMetric, aliases: dict[str, str] | None) -> str:
        tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", metric.formula))
        expr = metric.formula
        for token in sorted(tokens, key=len, reverse=True):
            if token in self.metrics:
                expr = re.sub(rf"\b{re.escape(token)}\b", f"({self._metric_expr(self.metrics[token], aliases)})", expr)
        return expr

    def _dimension_expr(
        self,
        dimension: SemanticDimension,
        grain: str | None,
        aliases: dict[str, str] | None,
    ) -> str:
        column = self._column_ref(dimension.table, dimension.expr, aliases)
        if dimension.type != "time" or not grain:
            return column
        if self.dialect in {"duckdb", "postgres", "postgresql"}:
            return f"DATE_TRUNC('{grain}', {column})"
        return column

    def _joins_for_dimensions(self, base_table: str, dimensions: list[SemanticDimension]) -> list[tuple[str, str, str, str]]:
        joins: list[tuple[str, str, str, str]] = []
        for dimension in dimensions:
            if dimension.table == base_table:
                continue
            relationship = next(
                (
                    rel
                    for rel in self.catalog.relationships
                    if rel.is_active
                    and rel.cardinality in {"many_to_one", "one_to_one"}
                    and not relationship_safety_error(self.catalog, rel)
                    and rel.from_table == base_table
                    and rel.to_table == dimension.table
                ),
                None,
            )
            if relationship is None:
                raise ValueError(f"No safe many-to-one relationship from {base_table} to {dimension.table}")
            join = (
                relationship.from_table,
                relationship.from_column,
                relationship.to_table,
                relationship.to_column,
            )
            if join not in joins:
                joins.append(join)
        return joins

    @staticmethod
    def _aliases(base_table: str, joins: list[tuple[str, str, str, str]]) -> dict[str, str]:
        aliases = {base_table: "t0"}
        for index, (_from_table, _from_column, to_table, _to_column) in enumerate(joins, start=1):
            aliases.setdefault(to_table, f"t{index}")
        return aliases

    def _from_clause(
        self,
        base_table: str,
        joins: list[tuple[str, str, str, str]],
        aliases: dict[str, str],
    ) -> str:
        if not joins:
            return self._quote_table(base_table)
        sql = f"{self._quote_table(base_table)} AS {aliases[base_table]}"
        for from_table, from_column, to_table, to_column in joins:
            sql += (
                f" LEFT JOIN {self._quote_table(to_table)} AS {aliases[to_table]}"
                f" ON {self._column_ref(from_table, from_column, aliases)} = {self._column_ref(to_table, to_column, aliases)}"
            )
        return sql

    def _where(self, filters: list[SemanticQueryFilter], aliases: dict[str, str] | None) -> str:
        parts = []
        for item in filters:
            field = self._filter_field(item.field, aliases)
            if item.op == "in":
                values = item.value if isinstance(item.value, list) else [item.value]
                parts.append(f"{field} IN ({', '.join(self._literal(value) for value in values)})")
            else:
                parts.append(f"{field} {item.op} {self._literal(item.value)}")
        return " AND ".join(parts)

    def _filter_field(self, field: str, aliases: dict[str, str] | None) -> str:
        name = str(field or "").strip()
        dimension = self.dimensions.get(name)
        if dimension is not None:
            return self._dimension_expr(dimension, None, aliases)
        fact = self.facts.get(name)
        if fact is not None:
            return self._column_ref(fact.table, fact.expr, aliases)
        raise ValueError(f"Unknown semantic filter field: {field}")

    def _order_by(self, order_by: list[SemanticQueryOrder], metrics: list[SemanticMetric]) -> str:
        if order_by:
            return ", ".join(f"{self._quote(item.field)} {item.direction.upper()}" for item in order_by)
        return f"{self._quote(metrics[0].key)} DESC" if metrics else ""

    @staticmethod
    def _literal(value: Any) -> str:
        if isinstance(value, (int, float)):
            return str(value)
        return "'" + str(value).replace("'", "''") + "'"

    def _quote_table(self, value: str) -> str:
        return ".".join(self._quote(part) for part in str(value).split(".") if part)

    def _column_ref(self, table: str, column: str, aliases: dict[str, str] | None) -> str:
        if aliases:
            alias = aliases.get(table)
            if not alias:
                raise ValueError(f"Missing table alias for {table}")
            return f"{alias}.{self._quote(column)}"
        return self._quote(column)

    @staticmethod
    def _quote(value: str) -> str:
        return '"' + str(value).replace('"', '""') + '"'


def semantic_query_from_hints(
    hints: dict[str, object],
    *,
    question: str,
    catalog: SemanticCatalog,
) -> SemanticQuery | None:
    metrics = [item for item in hints.get("metrics", []) if isinstance(item, dict)]
    if not metrics:
        return None
    metric_keys = [
        str(item.get("key") or "").strip()
        for item in metrics
        if str(item.get("key") or "").strip()
    ]
    q = str(question or "").lower()
    for metric in catalog.metrics:
        haystack = " ".join([metric.key, metric.name, *metric.synonyms]).lower()
        if any(token and token in q for token in _tokens(haystack)):
            metric_keys.append(metric.key)
    metric_keys = list(dict.fromkeys(metric_keys))
    if not metric_keys:
        return None
    dimensions = [
        dim.name
        for dim in catalog.dimensions
        if dim.is_active and dim.name.lower() in q
    ]
    time_dimension = next((dim.name for dim in catalog.dimensions if dim.type == "time" and dim.name.lower() in q), None)
    return SemanticQuery(metrics=metric_keys, dimensions=dimensions[:3], time_dimension=time_dimension)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", str(text or "").lower())
        if len(token) >= 3
    }
