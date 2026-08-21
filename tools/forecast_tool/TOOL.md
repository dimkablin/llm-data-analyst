---
id: forecast_tool
name: Forecast Tool
kind: tool
tool_key: forecast_tool
description: Call the structured forecast tool with a question and horizon; it returns forecast table/plot artifacts.
enabled_by_default: true
triggers:
  - forecast
  - prediction
  - future values
  - time series forecast
  - horizon
---

## Purpose

Use `forecast_tool` when the user asks for future values from a historical time series.

### API

Call the registered `forecast_tool` directly with structured arguments:

```json
{
  "question": "Prepare monthly dt and y for attrition and forecast the next 12 months.",
  "horizon": 12
}
```

This is a tool call, not Python code. Do not use imports or instantiate a
Python forecasting class in a code sandbox.

### Final result protocol

The tool returns table and plot artifacts automatically. Do not create
`tool_result` manually and do not route the call through `pandas_tool`.

### Runtime rules

- The predict service prepares/validates the forecast input.
- The historical input must have `dt` and `y`.
- Do not compute forecast values manually in SQL or pandas.
