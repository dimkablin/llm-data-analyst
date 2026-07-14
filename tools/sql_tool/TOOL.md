---
id: sql_tool
name: SQL Tool
kind: tool
tool_key: sql_tool
description: >
  Run read-only SQL over connected databases and uploaded CSV/XLSX tables materialized in DuckDB.
  Use exact table/column names from schema. In DuckDB/Postgres, double-quote identifiers
  with spaces, Cyrillic/non-ASCII, punctuation, or leading digits; never replace spaces
  with underscores.
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

Use `sql_tool` for table discovery, schema inspection, joins, aggregations, and read-only analytical SELECT queries.

### API

```json
{"mode": "catalog_tables"}
{"mode": "describe_table", "table_names": ["orders", "customers"]}
{"mode": "execute_sql", "sql": "SELECT ...", "artifact_name": "joined_orders"}
{"mode": "nl_query", "question": "Join orders and customers by customer_id"}
```

### Final result protocol

The tool returns a table artifact directly. The returned artifact name is the sandbox variable name for later `pandas_tool` or `plotly_tool` calls.

### Runtime rules

- Prefer `catalog_tables` before choosing table names.
- Prefer `describe_table` before joins or filters.
- Use `execute_sql` only for complete read-only `SELECT` or `WITH` queries.
- Select or alias source columns before handing data to pandas.
