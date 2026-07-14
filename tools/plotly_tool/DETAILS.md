## plotly_tool details

Input is Python code. Output is a Plotly `Figure` wrapped through `chart.result(fig, artifact_name="...")`.

### Scope

- `df` is the current session DataFrame when one exists.
- `px`, `go`, `make_subplots`, `chart`, `pd`, and `np` are available.
- Variables from previous successful tool calls are available by exact name.

### Result contract

```python
fig = px.bar(df, x="category", y="revenue", title="Revenue by category")
tool_result = chart.result(fig, artifact_name="revenue_by_category")
tool_result
```

- `fig` must be a real Plotly `Figure`.
- `artifact_name` should be a short snake_case slug without spaces.
- Last line must be `tool_result`.

### Rules

- Do not read files directly.
- Do not query the database here; use `sql_tool` first when data must be fetched.
- Do not use matplotlib, seaborn, or pandas `.plot`.
- Always set a readable title and axis labels.
- For large data, aggregate or sample before plotting.

### Existing artifact example

```python
# `monthly_sales` was returned by pandas_tool or sql_tool.
fig = px.line(
    monthly_sales,
    x="month",
    y="sales",
    title="Monthly sales",
    labels={"month": "Month", "sales": "Sales"},
    markers=True,
)
tool_result = chart.result(fig, artifact_name="monthly_sales_trend")
tool_result
```

### Common errors

| Error | Cause | Fix |
|---|---|---|
| `NameError` | variable is not in sandbox scope | inspect available variables or use the artifact returned by previous tools |
| `KeyError` | wrong column name | inspect exact DataFrame columns before retrying |
| `artifact_type: plot expected Figure` | returned a DataFrame or dict instead of `fig` | build `fig`, then call `chart.result(fig, ...)` |
