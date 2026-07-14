---
id: plotly_tool
name: Plotly Tool
kind: tool
tool_key: plotly_tool
description: Build Plotly chart artifacts from dataframe variables already present in the session sandbox.
enabled_by_default: true
triggers:
  - plot
  - chart
  - charts
  - plotly
  - visualization
  - bar
  - line
  - scatter
  - histogram
---

## Purpose

Use `plotly_tool` to create visual artifacts after data is available in the sandbox.

### API

Input is Python code executed in the session sandbox. Available common names include `px`, `go`, `make_subplots`, `chart`, `pd`, `np`, `df`, and prior artifact variables.

### Final result protocol

Create a Plotly `fig`, then return:

```python
tool_result = chart.result(fig, artifact_name="chart_name")
tool_result
```

### Runtime rules

- Do not fetch data from the database here.
- Use `sql_tool` or `pandas_tool` first when data preparation is needed.
- Do not use matplotlib or `df.plot`.
