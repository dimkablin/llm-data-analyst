## pandas_tool details

Input is Python code executed in an isolated staging namespace. After a successful
call, validated `tool_result.items` are published to the session sandbox by their
exact item names.

### DataFrame / DuckDB contract

- DuckDB table names are source tables, not pandas variables.
- `describe_table` shows source table columns only. It does not make those columns available in pandas unless a SQL artifact selected them.
- In pandas code, every column reference must exist in the current DataFrame variable being used.
- After `sql_tool` aliases or aggregations, use the returned artifact's output columns, not the original source column names.
- If a needed source column is absent from the current DataFrame, call `sql_tool(mode="execute_sql")` to materialize a new artifact first.
- A pandas code block contains one dataframe transformation. Data acquisition, visualization, and other capabilities run as separate top-level actions between pandas calls.

### Scope

- `df` is the current session DataFrame when one exists. Do not call `pd.read_csv()` or `pd.read_excel()`.
- `pd` and `np` are available.
- Items published by previous successful `tool_result` envelopes are available by exact name.
- A failed call leaves the published session sandbox unchanged.
- Table previews contain the published artifact rows; compute aggregates from the named dataframe.

### Result contract

```python
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"table_name": result_df},
}
tool_result
```

`items` must be a dict. Every value in `items` must be a pandas DataFrame or Series.

### Recovery rules

- On `KeyError`, inspect actual columns with `.columns`, `.head()`, and `.dtypes`.
- Do not rerun the same failing code under a new artifact name.
- Declare helper functions in the same code block before using them.
- Use exact variable names from prior tool results; do not invent aliases from chat history.
- When additional source data is required, complete the current pandas action and request data acquisition as the next top-level tool call.

### Example

```python
agg = df.groupby("category", dropna=False)["revenue"].sum().reset_index()
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"revenue_by_category": agg},
}
tool_result
```
