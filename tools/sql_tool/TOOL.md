---
id: sql_tool
name: SQL Tool
kind: tool
tool_key: sql_tool
description: >
  Run read-only SQL over connected databases and uploaded CSV/XLSX tables materialized in DuckDB.
  Use exact table/column names from schema. In DuckDB/Postgres, double-quote identifiers
  with spaces, Cyrillic/non-ASCII, punctuation, or leading digits and preserve their spelling.
  Select raw numeric aggregates with stable aliases, for example AVG(value) AS avg_value;
  apply presentation precision after the returned table is materialized. For row-encoded
  scenarios, conditionally aggregate in a CTE grouped only by final dimensions, then compute
  deltas, ranking, and limits in the outer SELECT. For independent comparison dimensions,
  aggregate each slice separately and UNION ALL one normalized final evidence table.
enabled_by_default: true
triggers:
  - sql
  - database
  - db
  - table
  - tables
  - query
  - duckdb
  - csv
  - xlsx
  - join
  - aggregation
---

## Purpose

Use `sql_tool` for table discovery, schema inspection, joins, raw numeric
aggregations, and read-only analytical SELECT queries.

### API

```json
{"mode": "catalog_tables"}
{"mode": "describe_table", "table_names": ["orders", "customers"]}
{"mode": "execute_sql", "sql": "SELECT customer_id, SUM(amount) AS total_amount FROM orders GROUP BY customer_id", "artifact_name": "order_totals"}
{"mode": "semantic_query", "metrics": ["metric_key"], "dimensions": ["dimension"], "limit": 100}
```

### Final result protocol

The tool returns a table artifact directly. The returned artifact name is the sandbox variable name for later `pandas_tool` or `plotly_tool` calls. When the rows already have the requested final grain, use the artifact as evidence and let the agent synthesize the answer; do not add Pandas only to sort, round, relabel, or format it.

### Runtime contract

- When semantic context reports `metric_resolution=resolved`, use `semantic_query` with its
  `confirmed_metric_keys` only when their `allowed_dimensions` support the requested grain;
  otherwise choose a compatible complete top-k candidate already in context, and resolve only
  when none is complete and unambiguous. Do not inspect coverage or rebuild a resolved metric in
  raw SQL. Use
  `execute_sql` only when no confirmed contract covers the calculation, or when resolved
  metrics span base tables and semantic `execution_mode` requires it.
- Use `catalog_tables` and `describe_table` only when semantic context or prior evidence does not
  already resolve the required physical tables and fields.
- Copy table names, column names, types, and nullability from the returned schema.
- Use those observed columns verbatim; synthesized convenience totals are not source fields.
- Build complete read-only `SELECT` or `WITH` statements from those observed fields.
- For top-N over repeated rows, group to the final period/slice/item grain before `ROW_NUMBER` or `LIMIT`.
- Return raw numeric aggregates with stable aliases; runtime presentation or final prose
  handles display precision. Do not add Pandas only for rounding.
- In PostgreSQL, do not call `ROUND(double precision, digits)`; return the raw value and round later.
- PostgreSQL has no `UNPIVOT` syntax. With `CROSS JOIN LATERAL (VALUES ...)`,
  project the value-table alias columns into the enclosing `SELECT`. After one
  alias or syntax failure, switch to explicit `UNION ALL` branches instead of
  retrying an equivalent LATERAL query.
- Unpivot peer entity columns inside their source table. Join a lookup only when
  the fact row exposes a verified key; an `OR` of nonzero measures is never a join condition.
- For a latest-observed window, derive the anchor from `MAX(source_date)`; today's
  date sets the action/report date, not the source's data availability.
- When relative periods are requested without a year, inspect coverage once, use
  the latest complete year containing every compared period, and reuse it throughout.
- A latest snapshot filters rows to that maximum date; do not average all history
  while merely selecting `MAX(date)` beside the aggregate. Qualify every joined
  field with the alias of the table that declares it.
- Derive period buckets in each source row; do not join a bucket projection back
  to facts on a non-unique date alone.
- When the request explicitly compares independent dimensions, aggregate each slice
  inside one final query and `UNION ALL` rows shaped as dimension type, member, period, value,
  baseline, and delta. For absolute slices, preserve the requested measures without
  inventing a comparison. Do not form joint member combinations across unrelated wide
  families. Each member label maps only to its corresponding value column.
- Never resend failed SQL unchanged. For a missing-column error, project the column
  from the immediate upstream SELECT or remove its downstream reference before retrying.
- Sandbox artifact names are Python dataframes, not database relations. Use
  `pandas_tool` to transform them or repeat the source SQL with needed filters.
- When mutually exclusive scenarios are values in a discriminator column, filter
  the measured value column instead of inventing separate source columns.
- When comparing several scenario values, conditionally aggregate them in a CTE
  and exclude the scenario discriminator from `GROUP BY`; calculate derived aliases,
  ordering, and limits in the outer query.
- A downstream CTE or outer query can reference only columns and aliases exposed
  by its immediate input CTE; source-column names renamed upstream no longer exist.
- Execute only a semantic metric selected from the retrieved candidates; keep
  ambiguous candidates as context until the main agent disambiguates them.
- `semantic_query` supports metrics on one compatible base table. When requested metrics use
  different base tables, aggregate each source in a separate CTE to the declared relationship
  join grain, then join the CTEs in one `execute_sql` call. Never join raw fact rows from both
  sources directly because that multiplies aggregates.
- Pass semantic fields (`metrics`, `dimensions`, `time_dimension`, `time_grain`,
  `filters`, `order_by`, `limit`) directly as top-level typed arguments, with
  arrays and objects kept in their native JSON form.
- Metric-defined filters are compiled automatically. Do not repeat them as query
  filters, and use only dimensions allowed by every selected metric.
- `time_grain` controls grouping only. Encode every requested temporal boundary
  as typed filters.
- Portable aggregate shape: `SELECT segment, AVG(value) AS avg_value FROM source GROUP BY segment`.
- The chosen `artifact_name` becomes the exact sandbox variable for subsequent actions.
