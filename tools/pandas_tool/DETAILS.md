## pandas_tool details

Input is Python code executed in the session sandbox. Variables persist between tool calls.

### DataFrame / DuckDB contract

- DuckDB table names are source tables, not pandas variables.
- `describe_table` shows source table columns only. It does not make those columns available in pandas unless a SQL artifact selected them.
- In pandas code, every column reference must exist in the current DataFrame variable being used.
- After `sql_tool` aliases or aggregations, use the returned artifact's output columns, not the original source column names.
- If a needed source column is absent from the current DataFrame, call `sql_tool(mode="execute_sql")` to materialize a new artifact first.
- Never call `sql_tool`, `plotly_tool`, `pandas_tool`, `database_tool`, or any other tool from inside pandas code. Tool orchestration happens between tool calls, not inside the sandbox code block.

### Scope

- `df` is the current session DataFrame when one exists. Do not call `pd.read_csv()` or `pd.read_excel()`.
- `pd` and `np` are available.
- Variables created by previous successful tool calls are available by exact name.
- If a previous tool call failed, its variables and helper functions were not persisted.
- Table artifacts include automatic numeric `summary_rows` (`__sum__` and `__mean__` per numeric column), and tool observations may show them as `numeric_summary_rows_appended`. Use those rows for column sums/means instead of recomputing the same totals.

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
- If you need a SQL query, stop using pandas code and request a separate `sql_tool` call; do not try to access tool APIs through Python variables.

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
