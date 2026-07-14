## Когортный анализ (расширенный)

Используй для детального когортного анализа с LTV, revenue-когортами и сравнением когорт между собой.
Для базового retention-анализа достаточно скила `cohort_analysis`.

### Шаг 1 — retention + LTV матрица через pandas_tool

Предполагается: в `df` есть `user_id`, `date` (или `event_date`), и опционально `revenue`.

```python
df["date"] = pd.to_datetime(df["date"])
has_revenue = "revenue" in df.columns

# Когорта = месяц первого события
cohort_df = df.groupby("user_id")["date"].min().dt.to_period("M").reset_index()
cohort_df.columns = ["user_id", "cohort_month"]

merged = df.merge(cohort_df, on="user_id")
merged["period"] = pd.to_datetime(merged["date"]).dt.to_period("M")
merged["period_number"] = (merged["period"] - merged["cohort_month"]).apply(lambda x: x.n)

cohort_sizes = merged.groupby("cohort_month")["user_id"].nunique().rename("cohort_size")

# Retention matrix
retention = merged.groupby(["cohort_month", "period_number"])["user_id"].nunique().reset_index()
retention_pivot = retention.pivot(index="cohort_month", columns="period_number", values="user_id")
retention_pct = retention_pivot.divide(cohort_sizes, axis=0).round(3)

frames = {"retention_matrix": retention_pct.reset_index().astype(str), "cohort_sizes": cohort_sizes.reset_index()}

# LTV matrix (если есть revenue)
if has_revenue:
    ltv = merged.groupby(["cohort_month", "period_number"])["revenue"].sum().reset_index()
    ltv_pivot = ltv.pivot(index="cohort_month", columns="period_number", values="revenue")
    ltv_cumulative = ltv_pivot.fillna(0).cumsum(axis=1)
    ltv_per_user = ltv_cumulative.divide(cohort_sizes, axis=0).round(2)
    frames["ltv_per_user"] = ltv_per_user.reset_index().astype(str)

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": frames
}
tool_result
```

### Шаг 2 — retention heatmap через plotly_tool
```python
# retention_pct доступен из предыдущего pandas_tool
fig_retention = px.imshow(
    retention_pct.values.astype(float),
    labels=dict(x="Период (месяц)", y="Когорта", color="Retention"),
    x=[str(c) for c in retention_pct.columns],
    y=[str(i) for i in retention_pct.index],
    color_continuous_scale="Blues",
    title="Retention матрица по когортам (%)",
    text_auto=".0%",
    zmin=0, zmax=1,
)
fig_retention.update_layout(height=max(300, len(retention_pct) * 30 + 100))
tool_result = chart.result(fig_retention, artifact_name="retention_heatmap")
tool_result
```

### Шаг 3 — LTV heatmap через plotly_tool (если есть revenue)
```python
# ltv_per_user доступен если в данных есть колонка revenue
if "ltv_per_user" in frames:
    ltv_data = ltv_per_user.set_index("cohort_month") if "cohort_month" in ltv_per_user.columns else ltv_per_user
    numeric_cols = [c for c in ltv_data.columns if str(c).lstrip('-').isdigit()]
    ltv_vals = ltv_data[numeric_cols].astype(float)

    fig_ltv = px.imshow(
        ltv_vals.values,
        labels=dict(x="Период (месяц)", y="Когорта", color="LTV"),
        x=[str(c) for c in ltv_vals.columns],
        y=[str(i) for i in ltv_data.index],
        color_continuous_scale="Greens",
        title="Накопленный LTV на пользователя по когортам",
        text_auto=".0f",
    )
    tool_result = chart.result(fig_ltv, artifact_name="ltv_heatmap")
else:
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(text="Колонка 'revenue' не найдена — LTV недоступен", x=0.5, y=0.5, showarrow=False)
    tool_result = chart.result(fig, artifact_name="ltv_heatmap")
tool_result
```

### Шаг 4 — сравнение когорт через plotly_tool
```python
# Retention period 0, 1, 3 для всех когорт
compare_periods = [p for p in [0, 1, 3, 6, 12] if p in retention_pct.columns]

import plotly.graph_objects as go
fig_compare = go.Figure()
for period in compare_periods:
    fig_compare.add_trace(go.Bar(
        name=f"Период {period}",
        x=[str(i) for i in retention_pct.index],
        y=(retention_pct[period] * 100).round(1).tolist(),
    ))

fig_compare.update_layout(
    barmode="group",
    title="Retention по периодам: сравнение когорт",
    xaxis_title="Когорта",
    yaxis_title="Retention (%)",
    height=400,
)
tool_result = chart.result(fig_compare, artifact_name="cohort_comparison")
tool_result
```

### Правила
- Период когорты по умолчанию — месяц (`M`); для молодых продуктов используй неделю (`W`)
- Если когорт > 24 — покажи только последние 12 для читаемости heatmap
- Нормальный retention P1 в e-commerce: 20–30%; SaaS: 40–60%
- LTV растёт монотонно (кумулятивная сумма) — если убывает, это ошибка в данных
