## DuckDB analysis details

Use DuckDB-backed `sql_tool` when analysis spans multiple uploaded CSV/XLSX files, large files, or tables that need joins/aggregations.

### Uploaded table workflow

```json
{"mode": "catalog_tables"}
```

Find the available DuckDB tables.

```json
{"mode": "describe_table", "table_names": ["actuals", "plan"]}
```

Inspect exact columns and data types.

```json
{
  "mode": "execute_sql",
  "artifact_name": "plan_fact_by_project",
  "sql": "WITH actuals_agg AS (...), plan_agg AS (...) SELECT ... FROM actuals_agg a JOIN plan_agg p ON ..."
}
```

Create the analytical result. The returned artifact name, for example `plan_fact_by_project`, becomes the sandbox variable for later tools.

### Direct file SQL

When a file path is explicitly available and not already registered as an uploaded table, DuckDB functions can be used in read-only SQL:

```sql
SELECT * FROM read_csv_auto('/path/file.csv') LIMIT 10;
SELECT * FROM read_parquet('/path/*.parquet') LIMIT 10;
```

For non-standard CSV options:

```sql
SELECT *
FROM read_csv('/path/file.csv', encoding='cp1251', delim=';', header=true, auto_detect=true)
LIMIT 10;
```

### Join guidance

- Inspect both schemas before writing the join.
- Normalize join keys in SQL when possible.
- Aggregate each side to the requested grain before joining if raw tables may be many-to-many.
- Preserve join lineage by producing one SQL artifact from all source tables.

### Handoff to pandas_tool or plotly_tool

After `sql_tool`, use the returned artifact variable name:
The artifact has the SQL output columns only. Do not use source-table column names in pandas/plotly unless the SQL result selected them with those names.
If another source column is needed for a calculation, create a new SQL artifact that includes it.

```python
# sql_tool returned artifact `plan_fact_by_project`
result = plan_fact_by_project.copy()
tool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"result": result}}
tool_result
```

Do not assume the SQL result is named `df`.
