---
name: CSV Summarizer
description: Быстрый автоматический обзор CSV-датасета — типы, пропуски, статистика, топ-значения и базовые визуализации.
triggers: обзор, резюме, summary, опиши датасет, покажи структуру, что в файле, первичный анализ, csv summarizer, overview
---

## CSV Summarizer — быстрый обзор датасета

Используй, когда пользователь загрузил новый файл и хочет понять что в нём, или когда нужен быстрый первичный анализ без глубокого EDA.

### Шаг 1 — структура и типы через pandas_tool
```python
info_df = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.values,
    "non_null": df.notna().sum().values,
    "null_pct": (df.isna().mean() * 100).round(1).values,
    "unique": df.nunique().values,
})

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"schema": info_df}
}
tool_result
```

### Шаг 2 — описательная статистика через pandas_tool
```python
desc = df.describe(include="all").transpose().reset_index()
desc.columns = ["column"] + list(desc.columns[1:])

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"describe": desc}
}
tool_result
```

### Шаг 3 — топ-значения по категориальным колонкам через pandas_tool
```python
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
rows = []
for col in cat_cols[:5]:
    top = df[col].value_counts().head(3)
    for val, cnt in top.items():
        rows.append({"column": col, "value": str(val), "count": cnt, "pct": round(cnt / len(df) * 100, 1)})

top_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["column", "value", "count", "pct"])
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"top_values": top_df}
}
tool_result
```

### Шаг 4 — визуализация пропусков через plotly_tool
```python
null_pct = (df.isna().mean() * 100).sort_values(ascending=False)
null_pct = null_pct[null_pct > 0]
if len(null_pct) > 0:
    fig = px.bar(
        x=null_pct.index,
        y=null_pct.values,
        labels={"x": "Колонка", "y": "% пропусков"},
        title="Пропуски по колонкам (%)",
        color=null_pct.values,
        color_continuous_scale="Reds",
    )
    tool_result = chart.result(fig, artifact_name="missing_values")
else:
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(text="Пропуски отсутствуют ✓", x=0.5, y=0.5, showarrow=False, font=dict(size=18))
    fig.update_layout(title="Пропуски по колонкам")
    tool_result = chart.result(fig, artifact_name="missing_values")
tool_result
```

### Правила
- Анализируй максимум 5 категориальных колонок в шаге 3 — не перегружай вывод
- Если датасет > 100 колонок, сначала покажи схему, затем спроси пользователя какие колонки важны
- Числовые колонки с `unique == len(df)` — вероятно ID, укажи это в комментарии
- Дата-подобные колонки с dtype=object — предупреди, что их стоит привести к datetime
