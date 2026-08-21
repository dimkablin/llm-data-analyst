from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from backend.data_access.semantic_models import (
    SemanticCatalog,
    SemanticDimension,
    SemanticFact,
    SemanticMetric,
)
from backend.data_access.semantic_validator import metric_references, relationship_safety_error


class SemanticQueryFilter(BaseModel):
    field: str
    op: Literal["=", "!=", ">", ">=", "<", "<=", "in", "not_in", "starts_with"]
    value: str | int | float | bool | list[str | int | float | bool]

    @model_validator(mode="after")
    def validate_value_shape(self) -> SemanticQueryFilter:
        if self.op in {"in", "not_in"}:
            if not isinstance(self.value, list) or not self.value:
                raise ValueError(f"{self.op} filter requires a non-empty list or tuple")
        elif isinstance(self.value, list):
            raise ValueError(f"{self.op} filter requires a scalar value")
        return self


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
        self.columns_by_table = {
            table: {column.name for column in catalog.columns if column.table == table}
            for table in {column.table for column in catalog.columns}
        }
        self.dimensions_by_name: dict[str, list[SemanticDimension]] = {}
        self.dimensions_by_ref: dict[str, SemanticDimension] = {}
        for dimension in catalog.dimensions:
            if not dimension.is_active:
                continue
            self.dimensions_by_name.setdefault(dimension.name, []).append(dimension)
            self.dimensions_by_ref[f"{dimension.table}.{dimension.name}"] = dimension
            self.dimensions_by_ref[dimension.dimension_id] = dimension
        self.facts_by_name: dict[str, list[SemanticFact]] = {}
        self.facts_by_ref: dict[str, SemanticFact] = {}
        for fact in catalog.facts:
            self.facts_by_name.setdefault(fact.name, []).append(fact)
            self.facts_by_ref[f"{fact.table}.{fact.name}"] = fact
            self.facts_by_ref[fact.fact_id] = fact

    def compile(self, query: SemanticQuery) -> str:
        metrics = [self._metric(key) for key in query.metrics]
        metric_base_tables = {metric.base_table for metric in metrics}
        if len(metric_base_tables) != 1:
            raise ValueError("Only one metric base table is supported")
        base_table = next(iter(metric_base_tables))
        time_dimension_name = query.time_dimension
        if query.time_grain and not time_dimension_name:
            time_dimension_name = self.shared_default_time_dimension(metrics)
            if time_dimension_name is None:
                raise ValueError(
                    "time_grain requires an explicit time_dimension or one shared metric default"
                )
        dimensions = [self._dimension(name, base_table=base_table) for name in query.dimensions]
        time_dimension = None
        if time_dimension_name:
            time_dimension = self._dimension(time_dimension_name, base_table=base_table)
            if query.time_grain and time_dimension.type != "time":
                raise ValueError("time_grain requires an active time dimension")
            if all(item.dimension_id != time_dimension.dimension_id for item in dimensions):
                dimensions.append(time_dimension)
        common_filters = self._fixed_filter_contract(metrics)
        dimension_filters = []
        for item in query.filters:
            try:
                signature = self._filter_signature(item, base_table=base_table)
            except ValueError:
                dimension_filters.append(item)
                continue
            if signature in common_filters:
                continue
            dimension_filters.append(item)
        resolved_filters = [
            (item, self._filter_dimension(item.field, base_table=base_table)) for item in dimension_filters
        ]
        unsupported_filters = [item.field for item, dimension in resolved_filters if dimension is None]
        if unsupported_filters:
            raise ValueError(
                "Query filters require active semantic dimensions: " + ", ".join(unsupported_filters)
            )
        filter_dimensions = [dimension for _item, dimension in resolved_filters if dimension is not None]
        required_dimensions = list(
            {dimension.dimension_id: dimension for dimension in [*dimensions, *filter_dimensions]}.values()
        )
        self._validate_allowed_dimensions(metrics, required_dimensions)
        joins = self._joins_for_dimensions(base_table, required_dimensions)
        aliases = self._aliases(base_table, joins)
        qualified = bool(joins)

        select_parts: list[str] = []
        group_parts: list[str] = []
        for dimension in dimensions:
            expr = self._dimension_expr(
                dimension,
                (
                    query.time_grain
                    if time_dimension is not None and dimension.dimension_id == time_dimension.dimension_id
                    else None
                ),
                aliases if qualified else None,
            )
            select_parts.append(f"{expr} AS {self._quote(dimension.name)}")
            group_parts.append(expr)
        select_parts.extend(
            (f"{self._metric_expr(metric, aliases if qualified else None)} AS {self._quote(metric.key)}")
            for metric in metrics
        )

        sql = f"SELECT {', '.join(select_parts)} FROM {self._from_clause(base_table, joins, aliases)}"
        where = self._where(dimension_filters, aliases if qualified else None, base_table=base_table)
        if where:
            sql += f" WHERE {where}"
        if group_parts:
            sql += f" GROUP BY {', '.join(group_parts)}"
        order_by = self._order_by(query.order_by, metrics, dimensions)
        if order_by:
            sql += f" ORDER BY {order_by}"
        sql += f" LIMIT {int(query.limit)}"
        return sql

    def shared_default_time_dimension(self, metrics: list[SemanticMetric]) -> str | None:
        base_tables = {metric.base_table for metric in metrics}
        defaults = [str(metric.default_time_dimension or "").strip() for metric in metrics]
        if len(base_tables) != 1 or not defaults or any(not item for item in defaults):
            return None
        base_table = next(iter(base_tables))
        resolved = {self._dimension(item, base_table=base_table).dimension_id for item in defaults}
        return defaults[0] if len(resolved) == 1 else None

    def _metric(self, key: str) -> SemanticMetric:
        metric = self.metrics.get(key)
        if metric is None:
            raise ValueError(f"Unknown semantic metric: {key}")
        return metric

    def _dimension(self, name: str, *, base_table: str) -> SemanticDimension:
        candidates = self._dimension_candidates(name)
        if not candidates:
            raise ValueError(f"Unknown semantic dimension: {name}")
        return self._prefer_base_table(candidates, name=name, base_table=base_table, kind="dimension")

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
        filtered_column = column
        if metric.filters:
            predicate = self._where(
                [SemanticQueryFilter.model_validate(item.model_dump()) for item in metric.filters],
                aliases,
                base_table=metric.base_table,
            )
            filtered_column = f"CASE WHEN {predicate} THEN {column} END"
        if metric.agg == "count_distinct":
            return f"COUNT(DISTINCT {filtered_column})"
        return f"{str(metric.agg).upper()}({filtered_column})"

    def _fixed_filter_contract(
        self,
        metrics: list[SemanticMetric],
    ) -> set[tuple[Any, ...]]:
        leaf_filters = [
            filters for metric in metrics for filters in self._metric_leaf_filters(metric, visiting=set())
        ]
        if not leaf_filters:
            return set()
        return set.intersection(*leaf_filters)

    def _metric_leaf_filters(
        self,
        metric: SemanticMetric,
        *,
        visiting: set[str],
    ) -> list[set[tuple[Any, ...]]]:
        if metric.key in visiting:
            raise ValueError(f"Metric dependency cycle detected: {metric.key}")
        if metric.type == "simple":
            return [{self._filter_signature(item, base_table=metric.base_table) for item in metric.filters}]

        next_visiting = {*visiting, metric.key}
        dependencies = [self.metrics[ref] for ref in metric_references(metric) if ref in self.metrics]
        leaves = [
            filters
            for dependency in dependencies
            for filters in self._metric_leaf_filters(dependency, visiting=next_visiting)
        ]
        if metric.type == "derived" and not self._is_metric_only_formula(metric):
            leaves.append(set())
        return leaves or [set()]

    def _is_metric_only_formula(self, metric: SemanticMetric) -> bool:
        tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", metric.formula or ""))
        # ponytail: fail closed; use a SQL AST if more scalar wrappers need filter inheritance.
        return all(token in self.metrics or token.upper() == "NULLIF" for token in tokens)

    def _derived_metric_expr(self, metric: SemanticMetric, aliases: dict[str, str] | None) -> str:
        expr = metric.formula
        for column in sorted(
            self.columns_by_table.get(metric.base_table, set()),
            key=len,
            reverse=True,
        ):
            qualified = f"{metric.base_table}.{column}"
            column_ref = self._column_ref(metric.base_table, column, aliases)
            expr = re.sub(
                rf"(?<![\w.\"']){re.escape(qualified)}(?![\w.\"'])",
                lambda _match, replacement=column_ref: replacement,
                expr,
            )
        tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr))
        for token in sorted(tokens, key=len, reverse=True):
            if token in self.metrics:
                replacement = f"({self._metric_expr(self.metrics[token], aliases)})"
                expr = re.sub(
                    rf"(?<![\w.\"']){re.escape(token)}(?![\w.\"'])",
                    lambda _match, value=replacement: value,
                    expr,
                )
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
        if dimension.grains and grain not in dimension.grains:
            raise ValueError(f"Unsupported time grain {grain} for semantic dimension {dimension.name}")
        if self.dialect in {"duckdb", "postgres", "postgresql"}:
            return f"DATE_TRUNC('{grain}', {column})"
        if self.dialect == "clickhouse":
            function = {
                "day": "toStartOfDay",
                "week": "toStartOfWeek",
                "month": "toStartOfMonth",
                "quarter": "toStartOfQuarter",
                "year": "toStartOfYear",
            }[grain]
            return f"{function}({column})"
        raise ValueError(f"Semantic time grain is not supported for dialect: {self.dialect}")

    @staticmethod
    def _validate_allowed_dimensions(
        metrics: list[SemanticMetric],
        dimensions: list[SemanticDimension],
    ) -> None:
        for metric in metrics:
            if not metric.allowed_dimensions:
                continue
            allowed = set(metric.allowed_dimensions)
            unsupported = [
                dimension.name
                for dimension in dimensions
                if not allowed
                & {
                    dimension.name,
                    f"{dimension.table}.{dimension.name}",
                    dimension.dimension_id,
                }
            ]
            if unsupported:
                raise ValueError(f"Metric {metric.key} does not allow dimensions: {', '.join(unsupported)}")

    def _joins_for_dimensions(
        self, base_table: str, dimensions: list[SemanticDimension]
    ) -> list[tuple[str, str, str, str]]:
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
            left = self._column_ref(from_table, from_column, aliases)
            right = self._column_ref(to_table, to_column, aliases)
            sql += f" LEFT JOIN {self._quote_table(to_table)} AS {aliases[to_table]} ON {left} = {right}"
        return sql

    def _where(
        self,
        filters: list[SemanticQueryFilter],
        aliases: dict[str, str] | None,
        *,
        base_table: str,
    ) -> str:
        parts = []
        for item in filters:
            field = self._filter_field(item.field, aliases, base_table=base_table)
            if item.op in {"in", "not_in"}:
                values = item.value if isinstance(item.value, list) else [item.value]
                operator = "IN" if item.op == "in" else "NOT IN"
                parts.append(f"{field} {operator} ({', '.join(self._literal(value) for value in values)})")
            elif item.op == "starts_with":
                parts.append(f"{field} LIKE {self._literal(str(item.value) + '%')}")
            else:
                parts.append(f"{field} {item.op} {self._literal(item.value)}")
        return " AND ".join(parts)

    def _filter_field(
        self,
        field: str,
        aliases: dict[str, str] | None,
        *,
        base_table: str,
    ) -> str:
        table, column = self._filter_binding(field, base_table=base_table)
        return self._column_ref(table, column, aliases)

    def _filter_binding(self, field: str, *, base_table: str) -> tuple[str, str]:
        name = str(field or "").strip()
        dimension_candidates = self._dimension_candidates(name)
        if dimension_candidates:
            dimension = self._prefer_base_table(
                dimension_candidates,
                name=name,
                base_table=base_table,
                kind="dimension",
            )
            return dimension.table, dimension.expr
        fact_candidates = self._fact_candidates(name)
        if fact_candidates:
            fact = self._prefer_base_table(
                fact_candidates,
                name=name,
                base_table=base_table,
                kind="fact",
            )
            if fact.table != base_table:
                raise ValueError("Semantic fact filters must use the metric base table")
            return fact.table, fact.expr
        raw_name = name.rsplit(".", 1)[-1]
        if raw_name in self.columns_by_table.get(base_table, set()):
            return base_table, raw_name
        raise ValueError(f"Unknown semantic filter field: {field}")

    def _filter_signature(self, item: Any, *, base_table: str) -> tuple[Any, ...]:
        table, column = self._filter_binding(item.field, base_table=base_table)
        value = item.value
        if isinstance(value, list):
            value_key: Any = frozenset((type(member).__name__, member) for member in value)
        else:
            value_key = (type(value).__name__, value)
        return table, column, str(item.op), value_key

    def _filter_dimension(
        self,
        field: str,
        *,
        base_table: str,
    ) -> SemanticDimension | None:
        candidates = self._dimension_candidates(field)
        if not candidates:
            return None
        return self._prefer_base_table(
            candidates,
            name=field,
            base_table=base_table,
            kind="dimension",
        )

    def _dimension_candidates(self, name: str) -> list[SemanticDimension]:
        exact = self.dimensions_by_ref.get(str(name or "").strip())
        if exact is not None:
            return [exact]
        return list(self.dimensions_by_name.get(str(name or "").strip(), []))

    def _fact_candidates(self, name: str) -> list[SemanticFact]:
        exact = self.facts_by_ref.get(str(name or "").strip())
        if exact is not None:
            return [exact]
        return list(self.facts_by_name.get(str(name or "").strip(), []))

    @staticmethod
    def _prefer_base_table(candidates: list[Any], *, name: str, base_table: str, kind: str) -> Any:
        local = [item for item in candidates if item.table == base_table]
        if len(local) == 1:
            return local[0]
        if len(candidates) == 1:
            return candidates[0]
        refs = ", ".join(sorted(f"{item.table}.{item.name}" for item in candidates))
        raise ValueError(f"Ambiguous semantic {kind}: {name}. Use a qualified reference; candidates: {refs}")

    def _order_by(
        self,
        order_by: list[SemanticQueryOrder],
        metrics: list[SemanticMetric],
        dimensions: list[SemanticDimension],
    ) -> str:
        aliases = {metric.key: metric.key for metric in metrics}
        for dimension in dimensions:
            for ref in (
                dimension.name,
                f"{dimension.table}.{dimension.name}",
                dimension.dimension_id,
            ):
                aliases[ref] = dimension.name
        if order_by:
            unknown = [item.field for item in order_by if item.field not in aliases]
            if unknown:
                raise ValueError(f"Unknown semantic order field: {', '.join(unknown)}")
            return ", ".join(
                f"{self._quote(aliases[item.field])} {item.direction.upper()}" for item in order_by
            )
        time_dimensions = [dimension for dimension in dimensions if dimension.type == "time"]
        if time_dimensions:
            return f"{self._quote(time_dimensions[0].name)} ASC"
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
