---
name: Cohort Analysis
description: User retention and LTV analysis by cohorts (date of first event).
triggers: cohort, retention, ltv, churn, когорт, удержание, отток
---

## Cohort Analysis

Use when asked for cohort analysis, retention, LTV, or churn.

### Step 1 — pandas_tool: define cohorts, retention, LTV
```python
# Auto-detect user and date columns from names and types
user_col = next(
    (c for c in df.columns if "user" in c.lower() or "client" in c.lower() or "id" in c.lower()),
    df.columns[0]
)
date_col = next(
    (c for c in df.columns if df[c].dtype == "datetime64[ns]" or "date" in c.lower() or "time" in c.lower()),
    None
)
if date_col is None:
    raise ValueError(f"Date column not found. Available columns: {list(df.columns)}")

df["date"] = pd.to_datetime(df[date_col])

# Auto-granularity: ≤90 days → weekly, else → monthly
date_range = (df["date"].max() - df["date"].min()).days
freq = "W" if date_range <= 90 else "M"
freq_label = "week" if freq == "W" else "month"

# Cohort = period of first event per user
cohort_df = df.groupby(user_col)["date"].min().reset_index()
cohort_df.columns = [user_col, "cohort_month"]
cohort_df["cohort_month"] = cohort_df["cohort_month"].dt.to_period(freq)

merged = df.merge(cohort_df, on=user_col)
merged["period"] = pd.to_datetime(merged["date"]).dt.to_period(freq)
merged["period_number"] = (merged["period"] - merged["cohort_month"]).apply(lambda x: x.n)

cohort_sizes = merged.groupby("cohort_month")[user_col].nunique()
retention = merged.groupby(["cohort_month", "period_number"])[user_col].nunique().reset_index()
retention = retention.rename(columns={user_col: "users"})
retention_pivot = retention.pivot(index="cohort_month", columns="period_number", values="users")
retention_pct = retention_pivot.divide(cohort_sizes, axis=0).round(3)

# Validate: period_number=0 should be 100% for every cohort
if 0 in retention_pct.columns:
    bad_cohorts = retention_pct[retention_pct[0] != 1.0].index.tolist()
    if bad_cohorts:
        print(f"WARNING: cohorts {bad_cohorts} have day-0 retention != 100% — possible data error")

# LTV: build cumulative if revenue/amount/sum column exists
revenue_col = next(
    (c for c in df.columns if "revenue" in c.lower() or "amount" in c.lower() or "sum" in c.lower()),
    None
)
if revenue_col is not None:
    ltv = merged.groupby(["cohort_month", "period_number"])[revenue_col].sum().reset_index()
    ltv_pivot = ltv.pivot(index="cohort_month", columns="period_number", values=revenue_col)
    ltv_cumulative = ltv_pivot.cumsum(axis=1)
else:
    ltv_cumulative = None

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {
        "retention_matrix": retention_pct.reset_index().astype(str),
        "cohort_sizes": cohort_sizes.reset_index().rename(columns={0: "size"}).astype(str),
    }
}
tool_result
```

### Step 2 — plotly_tool: retention heatmap with cohort size labels
```python
import plotly.express as px

# retention_pct, cohort_sizes, freq_label available from step 1
y_labels = [f"{str(i)} (n={cohort_sizes.get(i, '?')})" for i in retention_pct.index]

fig = px.imshow(
    retention_pct.values,
    labels=dict(x=f"Period ({freq_label})", y="Cohort", color="Retention"),
    x=[str(c) for c in retention_pct.columns],
    y=y_labels,
    color_continuous_scale="Blues",
    title=f"Cohort Retention Matrix (granularity: {freq_label})",
    text_auto=".0%",
)
fig.update_layout(height=max(300, len(retention_pct) * 40 + 100))
tool_result = chart.result(fig, artifact_name="cohort_retention")
tool_result
```

### Step 3 — pandas_tool: cohort comparison by period-1 retention
```python
# retention_pct, cohort_sizes available from step 1
if 1 in retention_pct.columns:
    month1 = retention_pct[1].dropna().sort_values(ascending=False)
    comparison = pd.DataFrame({
        "cohort": month1.index.astype(str),
        "retention_period_1_pct": (month1 * 100).round(1).values,
        "cohort_size": [cohort_sizes.get(c, None) for c in month1.index],
    })
    best = comparison.iloc[0]
    worst = comparison.iloc[-1]
    interpretation = (
        f"Best cohort: {best['cohort']} — retention {best['retention_period_1_pct']}% "
        f"(n={best['cohort_size']}). "
        f"Worst: {worst['cohort']} — retention {worst['retention_period_1_pct']}% "
        f"(n={worst['cohort_size']})."
    )
else:
    comparison = pd.DataFrame(columns=["cohort", "retention_period_1_pct", "cohort_size"])
    interpretation = "Period 1 absent from data — insufficient repeat events for comparison."

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"cohort_comparison": comparison},
    "summary": interpretation,
}
tool_result
```

### Rules
- ALWAYS auto-detect `user_col` and `date_col` from names and types — never hardcode column names
- ALWAYS auto-select granularity: range ≤90 days → weeks (`W`), else → months (`M`)
- ALWAYS validate day-0 (period_number=0) = 100% per cohort; print a warning if not
- ALWAYS return `cohort_sizes` as a separate key and show `(n=X)` in Y-axis labels
- DO NOT aggregate metrics before computing retention — `nunique` on `user_col` first, then divide by `cohort_sizes`
- LTV: auto-build `ltv_cumulative` if revenue/amount/sum column exists; skip silently if not
