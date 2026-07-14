---
name: Business Domain Analysis
description: General schema-first business analysis for market, portfolio, sales, operations, promotion, risk, forecast, anomaly, and reporting questions without prompt-specific hardcoding.
enabled_by_default: true
triggers: business analysis, domain analysis, market analysis, portfolio analysis, sales analysis, risk analysis, root cause, promotion analysis, forecast, anomaly, management memo, инвестиционный анализ, портфельный анализ, продажи, риск, первопричина, промо, прогноз, аномалии, управленческая записка
---

## Business Domain Analysis

Use this skill for business-domain analytical questions over tabular sources
when the user asks about market instruments, portfolios, sales, operations,
promotions, risks, drivers, forecasts, anomalies, or a final management note.
The skill is intentionally domain-general: it derives semantics from schema,
data values, user wording, and tool output rather than from customer names,
example source names, fixed filters, or memorized examples.

### Algorithm
1. **Schema discovery** -> use `data_catalog_tool` and, when needed,
   `sql_tool`/`database_tool` to identify available sources, tables, columns,
   periods, row counts, and representative values. Do not filter on a value
   before verifying it exists.
2. **Role mapping** -> build a compact role map from real columns:
   - time role: dates, periods, snapshots, horizons;
   - metric role: revenue, volume, price, value, weight, return, yield, plan,
     fact, traffic, conversion, discount, or other numeric measures;
   - dimension role: channel, category, region, segment, account, risk class,
     security type, sector, issuer, brand, campaign, or other grouping fields;
   - entity role: position, instrument, customer, product, location, or other
     primary objects.
3. **Question classification** -> select the method by intent, not by prompt
   text: descriptive slice, comparison, ranking/screener, concentration/risk,
   driver/root-cause analysis, promotion uplift, forecast, anomaly/plan-fact,
   or report synthesis.
4. **Evidence table** -> use `sql_tool` to create `business_evidence_table`
   with only the rows and columns needed for the selected method. Alias columns
   to clear semantic names while preserving original field names in the output
   or explanation.
5. **Metric computation** -> use `pandas_tool` to create
   `business_metric_table`: aggregates, deltas, shares, contribution, uplift,
   rankings, risk-return scores, or other method-specific calculations.
6. **Visualization** -> use `plotly_tool` to create `business_analysis_chart`
   for trends, comparisons, contribution, concentration, or ranking when a chart
   improves the answer.
7. **Specialized routing** -> when the selected method requires external
   services, call the relevant instructions and tool instead of reimplementing:
   `forecast_tool` for future values, `anomaly_planfact_tool` for deviations,
   `generate_summary_tool` for session-grounded synthesis, and
   `generate_report_tool` only for explicit file export.
8. **Final synthesis** -> answer with the calculation basis, artifact names,
   strongest findings, uncertainty, missing evidence, and concrete next action.

### Required tools
- `data_catalog_tool`
- `sql_tool`
- `pandas_tool`
- `plotly_tool`

### Required artifacts
- table: business_evidence_table
- table: business_metric_table
- plot: business_analysis_chart

### Evidence rules
- Every filter value, role mapping, metric, and dimension must come from schema discovery or tool output.
- Do not encode customer names, example source names, fixed tickers, brands, categories, regions, months, or baseline labels in the skill.
- If the user names a value, verify it against the data before using it as a filter; if absent, report the mismatch.
- Forecasts must use `forecast_tool`; anomaly and plan-fact detection must use `anomaly_planfact_tool`.
- Report export must use `generate_report_tool` only after evidence exists in persisted session history or artifacts.

### Rules
- Prefer semantic role names over hardcoded column names; show the mapping when it affects the conclusion.
- Keep domain interpretation separate from computed facts and state which artifacts support each conclusion.
- If a required role is missing, state the unsupported part and continue with the valid portion of the analysis.
- Do not invent external facts, market context, recommendations, or chart artifacts that are not present in tool output.
