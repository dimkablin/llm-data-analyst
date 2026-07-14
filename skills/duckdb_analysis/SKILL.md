---
name: DuckDB Large File Analysis
description: SQL analysis of large CSV, XLSX-derived, and Parquet tables via DuckDB; supports joins across multiple uploaded files.
enabled_by_default: true
triggers: duckdb, large file, large csv, parquet, sql on file, multiple files, join files, out of memory, large dataset, gb file
---

## DuckDB Large File Analysis

Use this skill when the user asks for SQL analysis over uploaded tables or large files.

In this project, uploaded CSV/XLSX files are registered as DuckDB tables in one session database. Use `sql_tool` with structured modes:

```json
{"mode": "catalog_tables"}
{"mode": "describe_table", "table_names": ["table_a", "table_b"]}
{"mode": "execute_sql", "sql": "SELECT ...", "artifact_name": "joined_result"}
```

If the user explicitly provides a file path that is not registered as an uploaded table, DuckDB SQL may use `read_csv_auto`, `read_csv`, or `read_parquet` inside an `execute_sql` query.

### Algorithm

1. Discover available tables with `catalog_tables`.
2. Inspect relevant schemas with `describe_table`.
3. For joins, aggregate raw tables to the intended grain first when many-to-many duplication is possible.
4. Execute a read-only SQL `SELECT`/`WITH` query with a stable `artifact_name`.
5. Use the returned artifact variable name in `pandas_tool` or `plotly_tool`; do not assume it is named `df`.
6. In later pandas/plotly code, use only the artifact's output columns. Source table columns from `describe_table` must be selected or aliased into the SQL result first.

### Правила

- Never guess table or column names; inspect them first.
- Prefer SQL for joins and aggregations across uploaded files.
- Use `LIMIT` for initial previews.
- Keep SQL read-only.
- Quote non-ASCII, spaced, or symbol-heavy column names with double quotes. If a query/code path fails repeatedly, inspect the real schema/sample rows or switch approach instead of rerunning the same query under a new artifact name.
