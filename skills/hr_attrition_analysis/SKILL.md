---
name: HR Attrition Analysis
description: Synthetic contact-center employee attrition analysis, aggregate forecasting, risk indicators, driver evidence, charts, and retention actions.
enabled_by_default: true
triggers: отток сотрудников, текучесть операторов, риск увольнения, прогноз увольнений, удержание сотрудников, HR attrition, employee churn, contact center turnover
---

## HR Attrition Analysis

Use this skill for the synthetic contact-center attrition demo. It coordinates
existing tools. Choose the branches requested by the user and complete them from
the current session evidence.

### Algorithm

1. **Validate source** -> use `data_catalog_tool` to confirm the available table,
   date range, row count, and required columns. Treat blank `attrition_90d` as the
   current scoring population and non-blank values as historical outcomes.
2. **Historical evidence** -> use `sql_tool` for observed attrition counts/rates by
   month, project, contact center, and requested dimensions, calculated from rows
   with known outcomes.
3. **Future forecast when requested** -> prepare valid historical
   `termination_date` counts as one ordered monthly `dt, y` series, resolving
   missing periods from source completeness; then call the bound `forecast`
   capability with its typed schema. For `{dt, y}` rows use `time_column="dt"` and
   `targets=[{"name": "metric", "column": "y", "aggregation": "none"}]`. A month
   maps to one period and a quarter to three.
4. **Risk indicators** -> use `pandas_tool` to compare historical attrition rates and
   feature distributions for workload, overtime, absence, lateness, adherence,
   productivity, quality, engagement, manager change, training, tenure, and shift.
   For current rows, report an attention list based on observed indicators. A
   predicted probability is supported by a predict-service result containing an
   employee identifier and score; `risk_score` is evidence only when supplied by
   that result.
5. **Visual evidence** -> use a provider-published `plot.figure` for forecast trends.
   Use `plotly_tool` for requested comparisons and ranked driver evidence built from
   table artifacts. Keep tables compact and show period and denominator.
6. **Retention actions** -> connect each action to a measured factor. Label statistical
   relationships as associations and synthetic-data impact as a hypothesis.
7. **Final response** -> lead with the answer, then evidence, limitations, and actions.
   Always include the synthetic-data notice below.

### Evidence rules

- Every number in prose must exist in a tool result or artifact.
- Forecast values must come from the active `forecast` capability with provider provenance.
- Historical relationships are reported as associations.
- Individual probabilities cite the predict-service result that returned them.
- Empty targets identify current rows and remain empty.

### Rules

- Forecast values come from the active forecast provider.
- Current employees are described through observed risk indicators.
- `termination_date` supplies historical monthly attrition counts.
- Month means `horizon=1`; quarter means `horizon=3` at monthly granularity.
- The demo scope is synthetic and separate from T2 deployment or employee data.
- End every answer with: **Демонстрация выполнена на синтетических данных. Результаты не относятся к реальным сотрудникам и не подтверждают фактическое внедрение в Т2.**
