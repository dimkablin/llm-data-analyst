## sql_tool details

`sql_tool` is the canonical table-analysis tool for uploaded CSV/XLSX files in the DuckDB session and for connected databases.

### Modes

```json
{"mode": "catalog_tables"}
```

Returns a catalog table with available schemas/tables.

```json
{"mode": "describe_table", "table_names": ["fact_march", "plan_march"]}
```

Returns column metadata for one or more tables. The aliases `table`, `table_name`, and a single string `table_names` are accepted and normalized to `table_names`.

```json
{"mode": "execute_sql", "sql": "SELECT project_id, SUM(actual) - SUM(plan) AS variance FROM project_values GROUP BY project_id", "artifact_name": "plan_fact_result"}
```

Executes a read-only `SELECT`/`WITH` query. Use this for explicit joins and aggregations after you know table and column names.

### Result handoff

The tool returns a table artifact and stores it in the sandbox under the artifact name returned in the tool response.
That artifact is a pandas DataFrame with its own output schema. Later pandas/plotly code must use this artifact's exact output columns, not the original DuckDB source columns unless they were selected with the same names.
`describe_table` output is only source metadata; it is not the schema of every later SQL result.

Correct follow-up:

```python
# pandas_tool or plotly_tool after sql_tool returned artifact `plan_fact_result`
summary = plan_fact_result.copy()
tool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"summary": summary}}
tool_result
```

Do not assume the SQL result is named `df`. `df` may refer to the initially loaded primary dataframe, not the latest SQL artifact.

### Recommended workflow

1. `catalog_tables`
2. `describe_table` for relevant tables
3. `execute_sql` for SQL joins and aggregations written by the main agent
4. Use the returned artifact variable name in `pandas_tool` or `plotly_tool`

### SQL rules

- Use only read-only `SELECT` or `WITH`.
- Quote identifiers when they contain spaces, Cyrillic characters, punctuation, or start with a digit.
- Aggregate before joining when raw tables can be many-to-many.
- Give important outputs stable `artifact_name` values in snake_case.
