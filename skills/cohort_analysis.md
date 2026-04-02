---
name: Когортный анализ
description: Анализ удержания и LTV пользователей по когортам (дата первого события)
triggers: когорт, retention, удержание, ltv, отток, churn
---

## Когортный анализ

Используй, когда пользователь просит анализ когорт, удержания, LTV или оттока.

### Шаг 1 — определи когорту через pandas_tool
```python
# Предполагается: df содержит колонки user_id, date (или event_date), и опционально revenue
df["date"] = pd.to_datetime(df["date"])
cohort_df = df.groupby("user_id")["date"].min().reset_index()
cohort_df.columns = ["user_id", "cohort_month"]
cohort_df["cohort_month"] = cohort_df["cohort_month"].dt.to_period("M")

merged = df.merge(cohort_df, on="user_id")
merged["period"] = pd.to_datetime(merged["date"]).dt.to_period("M")
merged["period_number"] = (merged["period"] - merged["cohort_month"]).apply(lambda x: x.n)

cohort_sizes = merged.groupby("cohort_month")["user_id"].nunique()
retention = merged.groupby(["cohort_month", "period_number"])["user_id"].nunique().reset_index()
retention = retention.rename(columns={"user_id": "users"})
retention_pivot = retention.pivot(index="cohort_month", columns="period_number", values="users")
retention_pct = retention_pivot.divide(cohort_sizes, axis=0).round(3)

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"retention_matrix": retention_pct.reset_index().astype(str)}
}
tool_result
```

### Шаг 2 — визуализируй через plotly_tool
```python
import plotly.express as px
# retention_pct доступен из предыдущего pandas_tool (sandbox сохраняет все переменные)
fig = px.imshow(
    retention_pct.values,
    labels=dict(x="Период (месяц)", y="Когорта", color="Retention"),
    x=[str(c) for c in retention_pct.columns],
    y=[str(i) for i in retention_pct.index],
    color_continuous_scale="Blues",
    title="Retention матрица по когортам",
    text_auto=".0%",
)
tool_result = chart.result(fig, artifact_name="cohort_retention")
tool_result
```

### Правила
- Если в df нет явного `user_id` — найди идентификатор пользователя по контексту колонок
- Для LTV: суммируй `revenue` по когорте вместо `nunique` по `user_id`
- Период когорты можно брать по неделям (`W`) если данных мало
