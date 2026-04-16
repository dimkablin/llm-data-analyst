---
name: Plotly Tool
description: Построение интерактивных графиков через Plotly. Единственный инструмент для визуализации — используй всегда когда нужен chart/plot/diagram.
kind: tool
tool_key: plotly_tool
triggers: график, графики, графика, диаграмм, диаграмма, визуализац, визуализация, plotly, chart, charts, plot, scatter, bar, line, pie, histogram, heatmap, столбчат, линейн
---

## plotly_tool — interactive charts

Entry: Python code. Exit: Plotly Figure wrapped via `chart.result(fig, artifact_name="...")`.

### API
```python
chart.result(fig: go.Figure, artifact_name: str) -> tool_result
db.query_dataframe(sql: str) -> pd.DataFrame  # if DB connected
```

### Scope
`px`, `go`, `chart`, `df`, `pd`, `np` always available.
`db`, `db_connection` — when DB session active.
All variables from prior tool calls available by name (see sandbox block in system prompt).

### Final result protocol
The last expression must be `tool_result` — the sandbox captures only the last expression.

```python
tool_result = chart.result(fig, artifact_name="chart_name")
tool_result
```

A print or assignment as the final line produces a silent empty result.

### Rules
- Never call `pd.read_csv()` — `df` is already in scope
- `len(df) > 5000` → `df.sample(5000, random_state=42)` first
- Always set `title` and axis `labels` for readability
- Never use matplotlib, seaborn, or `.plot()`
