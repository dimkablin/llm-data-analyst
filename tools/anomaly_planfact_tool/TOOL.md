---
id: anomaly_planfact_tool
name: Anomaly Plan-Fact Tool
kind: tool
tool_key: anomaly_planfact_tool
description: Run external anomaly and plan-fact analysis over aligned plan/fact time series.
enabled_by_default: true
triggers:
  - anomaly
  - plan fact
  - plan vs fact
  - deviation
  - variance
  - actual worse than plan
---

## Purpose

Use `anomaly_planfact_tool` when the user asks for plan/fact deviations, anomalies, or unusual gaps over time.

### API

Input is Python code that calls the helper:

```python
tool_result = anomaly_planfact.analyze_result(question="Prepare a plan/fact dataset with dt, plan, fact.")
tool_result
```

### Final result protocol

Return the helper output as `tool_result`. The helper creates table and plot artifacts.

### Runtime rules

- The predict service prepares/validates the input dataset.
- The input must have `dt`, `plan`, and `fact`.
- Do not pre-filter only worst rows before anomaly detection.
