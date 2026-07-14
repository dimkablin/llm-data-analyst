---
id: pandas_tool
name: Pandas Tool
kind: tool
tool_key: pandas_tool
description: Transform, clean, refine, and inspect dataframe artifacts already present in the session sandbox.
enabled_by_default: true
triggers:
  - pandas
  - dataframe
  - aggregation
  - filter
  - pivot
  - statistics
  - table transform
---

## Purpose

Use `pandas_tool` for Python/Pandas operations over existing sandbox dataframe variables.

### API

Input is Python code executed in the session sandbox. Available common names include `df`, `pd`, `np`, and variables created by earlier tool calls.

Do not call `sql_tool`, `plotly_tool`, `pandas_tool`, `database_tool`, or any other tool from inside this code. Tool calls happen only outside the pandas sandbox. If you need new source columns or a new SQL aggregation, stop and call `sql_tool` as a separate tool invocation before using `pandas_tool`.

### Final result protocol

Last line must be `tool_result`. For tables:

```python
tool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"name": result_df}}
tool_result
```

For inspection or diagnostics, return findings as a compact table artifact through `tool_result`.

### Runtime rules

- Do not read files directly.
- Do not query databases directly.
- Do not call or import other tools from pandas code.
- Start from exact dataframe variable and column names from existing artifacts.
- Use existing table artifact `summary_rows` / `numeric_summary_rows_appended` for column sums and means when those rows are already present; do not recompute the same sums/means unless the user needs a different grouping or filter.
- On runtime errors, inspect actual data and choose a different approach before retrying.
