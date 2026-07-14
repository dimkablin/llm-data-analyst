---
name: Cohort Analysis
description: User retention and LTV analysis by cohorts (date of first event).
enabled_by_default: false
triggers: cohort, retention, ltv, churn, когорт, удержание, отток
---

## Cohort Analysis

User retention and LTV analysis grouped by cohorts (period of first event).

### Algorithm (3 steps)
1. **Cohorts + retention + LTV** → `pandas_tool`: auto-detect user/date columns, compute `cohort_month`, `period_number`, `retention_pct` matrix. Build `ltv_cumulative` if revenue column exists.
2. **Retention heatmap** → `plotly_tool`: `px.imshow` with Blues scale; Y-axis labels include cohort size `(n=X)`.
3. **Cohort comparison** → `pandas_tool`: rank cohorts by period-1 retention, surface best and worst.

### Rules
- ALWAYS auto-detect `user_col` and `date_col` from names and types — never hardcode
- ALWAYS auto-select granularity: range ≤ 90 days → `W`, else → `M`
- ALWAYS validate period_number=0 = 100% per cohort; warn if not
- ALWAYS show `(n=X)` in Y-axis labels
- Compute `nunique(user_col)` per cohort first, then divide by `cohort_sizes` — do not pre-aggregate
