---
id: plotly_tool
name: Plotly Tool
kind: tool
tool_key: plotly_tool
description: Build one Plotly Figure when requested or when it materially clarifies a comparison or trend; explicit chart bans win. Use a current-turn sandbox variable directly and omit input_artifacts. Only for a durable artifact from history pass input_artifacts as alias to the stable artifact_id shown in history, never an artifact name. Use the exact columns from the latest observation; `chart` is already bound, so never import plotly_tool; order categorical axes with categoryorder, never category_order; return the figure with chart.result.
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

Use `plotly_tool` for one chart action after a successful table-producing action.

### API

Input is Python code executed in the session sandbox. Start from the exact named
dataframe and columns in the latest successful tool observation. Available common names
`px`, `go`, `make_subplots`, `chart`, `pd`, and `np` are already bound; named artifacts
are available directly.

For a dataframe named by the latest successful tool observation, use that sandbox variable
directly and omit `input_artifacts`. Only for an artifact listed in durable history, pass
`input_artifacts={"source": "artifact_id"}` and build from `source`. The value must be the
stable ID shown in history, never the artifact name.

### Final result protocol

Create one Plotly `fig`, then complete the code with:

```python
tool_result = chart.result(fig, artifact_name="chart_name")
tool_result
```

### Runtime contract

- Table preparation completes in a preceding top-level SQL or pandas action.
- Without an explicit chart request, do not duplicate a sufficient final table as a chart.
- The dataframe contains the exact fields used for chart axes and traces.
- `chart.result` receives the Plotly Figure and publishes the named plot artifact.
