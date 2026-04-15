---
name: Автоматический EDA
description: Системный разведочный анализ данных — распределения, корреляции, выбросы, аномалии типов.
triggers: eda, разведочный анализ, исследование данных, exploratory, корреляция, распределение, выбросы, профилинг, полный анализ
---

## Автоматический EDA

Используй для глубокого первичного анализа датасета. Охватывает распределения, корреляции, выбросы и типовые проблемы данных.

### Шаг 1 — числовые статистики → **pandas_tool** (НЕ plotly_tool)
> Этот шаг возвращает TableArtifact. Используй **только pandas_tool**.

```python
num_cols = df.select_dtypes(include="number").columns.tolist()

stats_rows = []
for col in num_cols:
    s = df[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    outliers = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
    stats_rows.append({
        "column": col,
        "mean": round(s.mean(), 3),
        "median": round(s.median(), 3),
        "std": round(s.std(), 3),
        "min": round(s.min(), 3),
        "max": round(s.max(), 3),
        "skew": round(s.skew(), 3),
        "outliers_iqr": int(outliers),
        "outliers_pct": round(outliers / len(df) * 100, 1),
    })

num_stats = pd.DataFrame(stats_rows)
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"numeric_stats": num_stats}
}
tool_result
```

### Шаг 2 — корреляционная матрица через plotly_tool
```python
num_cols = df.select_dtypes(include="number").columns.tolist()
if len(num_cols) >= 2:
    corr = df[num_cols].corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Корреляционная матрица",
        aspect="auto",
    )
    tool_result = chart.result(fig, artifact_name="correlation_matrix")
else:
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(text="Недостаточно числовых колонок для корреляции", x=0.5, y=0.5, showarrow=False)
    tool_result = chart.result(fig, artifact_name="correlation_matrix")
tool_result
```

### Шаг 3 — распределения числовых колонок через plotly_tool
```python
num_cols = df.select_dtypes(include="number").columns.tolist()[:6]
from plotly.subplots import make_subplots
import plotly.graph_objects as go

cols_per_row = 3
rows = (len(num_cols) + cols_per_row - 1) // cols_per_row
fig = make_subplots(rows=rows, cols=cols_per_row, subplot_titles=num_cols)

for i, col in enumerate(num_cols):
    r, c = divmod(i, cols_per_row)
    fig.add_trace(go.Histogram(x=df[col].dropna(), name=col, showlegend=False), row=r+1, col=c+1)

fig.update_layout(title="Распределения числовых колонок", height=300 * rows)
tool_result = chart.result(fig, artifact_name="distributions")
tool_result
```

### Шаг 4 — категориальные колонки → **pandas_tool** (НЕ plotly_tool)
> Этот шаг возвращает TableArtifact. Используй **только pandas_tool**.

```python
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

cat_rows = []
for col in cat_cols:
    n_unique = df[col].nunique()
    top_val = df[col].value_counts().index[0] if n_unique > 0 else None
    top_pct = round(df[col].value_counts().iloc[0] / len(df) * 100, 1) if n_unique > 0 else None
    cat_rows.append({
        "column": col,
        "unique_values": n_unique,
        "top_value": str(top_val),
        "top_pct": top_pct,
        "null_pct": round(df[col].isna().mean() * 100, 1),
        "likely_type": "ID" if n_unique == len(df) else ("binary" if n_unique == 2 else ("low_cardinality" if n_unique <= 10 else "high_cardinality")),
    })

cat_stats = pd.DataFrame(cat_rows) if cat_rows else pd.DataFrame(columns=["column", "unique_values", "top_value", "top_pct", "null_pct", "likely_type"])
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"categorical_stats": cat_stats}
}
tool_result
```

### Правила
- Анализируй максимум 6 числовых колонок в гистограммах — для большего числа выбирай наиболее важные
- Выбросы по IQR: более 5% — предупреди пользователя
- Skewness > 2 или < -2 — отметь как сильно скошенное распределение
- После EDA сформулируй 3–5 ключевых наблюдений в тексте
- Если колонка `likely_type == "ID"` — исключи из корреляции и статистик
