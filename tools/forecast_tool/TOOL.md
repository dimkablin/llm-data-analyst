---
id: forecast_tool
name: Forecast Tool
kind: tool
tool_key: forecast_tool
description: Run external forecasting over compact prepared time series and return forecast table/plot artifacts.
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

Input is Python code that calls the helper:

```python
tool_result = forecast.forecast_result(question="Prepare a historical series with dt and y.", horizon=12)
tool_result
```

### Final result protocol

Return the helper output as `tool_result`. The helper creates table and plot artifacts.

### Runtime rules

- The predict service prepares/validates the forecast input.
- The historical input must have `dt` and `y`.
- Do not compute forecast values manually in SQL or pandas.
