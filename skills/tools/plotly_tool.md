---
name: Plotly Tool
description: Построение интерактивных графиков через Plotly. Единственный инструмент для визуализации — используй всегда когда нужен chart/plot/diagram.
kind: tool
tool_key: plotly_tool
triggers: график, графики, графика, диаграмм, диаграмма, визуализац, визуализация, plotly, chart, charts, plot, scatter, bar, line, pie, histogram, heatmap, столбчат, линейн
---

## plotly_tool — интерактивные графики

Вход: Python-код. Выход: Plotly `Figure`, обёрнутый через `chart.result(fig, artifact_name="...")`.

### Примеры запросов пользователя

Вызывай этот инструмент, когда пользователь просит:
- "Построй график продаж по месяцам"
- "Покажи диаграмму распределения"
- "Визуализируй корреляцию X и Y"
- "Сделай scatter plot"
- "Нарисуй столбчатую диаграмму"
- "Покажи линейный тренд"
- "chart по категориям"
- "histogram по возрасту"

IMPORTANT: если пользователь просит визуализацию — обязательно вызови plotly_tool и доведи до `chart.result(fig, ...)`. Не подменяй график только `pandas_tool` или `value_tool`.

### Переменные в scope

Sandbox сессии сохраняет все переменные между вызовами инструментов.
При каждом вызове в scope доступны:
- `df` — DataFrame текущей сессии (уже загружен, **не вызывай** `pd.read_csv`)
- `px` — `plotly.express`
- `go` — `plotly.graph_objects`
- `chart` — хелпер для возврата результата
- `pd` — pandas
- `np` — numpy
- Все переменные из предыдущих tool-вызовов (список доступных переменных указан в системном промпте)

Если подключена БД (режим `db_connection`):
- `db` — хелпер для SQL-запросов (`db.query_dataframe(sql)`)
- `db_connection` — объект подключения

### Контракт результата

CRITICAL: последняя строка кода **обязана** возвращать `tool_result`.

```python
fig = px.bar(df, x="col_x", y="col_y", title="Заголовок")
tool_result = chart.result(fig, artifact_name="slug_без_пробелов")
tool_result
```

- `fig` должен быть настоящим `plotly.graph_objects.Figure`
- `artifact_name` — короткий slug (≤ 32 символа, без пробелов)
- `chart.result` принимает либо `chart.result(fig, artifact_name="name")`, либо `chart.result(fig, "name")`

### Правила

- NEVER вызывай `pd.read_csv()`, `pd.read_excel()` — `df` уже в scope
- NEVER используй matplotlib, `.plot()`, seaborn
- NEVER обращайся к `globals()`, `locals()`, `os`, `sys`, `subprocess`
- ALWAYS указывай `title` и `labels` для осей чтобы график был читаем
- Если `len(df) > 5000`: сначала сэмплируй: `d = df.sample(5000, random_state=42)`
- Числа без избыточной точности: используй `round()` или `tickformat`

---

### Сценарий 1: данные из `df` напрямую

Используй когда данные уже в сессионном DataFrame.

```python
fig = px.bar(
    df,
    x="category",
    y="revenue",
    title="Выручка по категориям",
    labels={"category": "Категория", "revenue": "Выручка, руб."},
)
tool_result = chart.result(fig, artifact_name="revenue_by_category")
tool_result
```

---

### Сценарий 2: данные из переменной предыдущего tool-вызова

Используй когда предыдущий `pandas_tool` создал агрегированный датафрейм.
Имена переменных и их колонки видны в блоке «Доступные переменные в sandbox» системного промпта.

IMPORTANT: не пересчитывай агрегацию заново — используй готовую переменную.

```python
# agg доступен из предыдущего pandas_tool
# Колонки указаны в системном промпте, например: ['region', 'total_sales']
fig = px.bar(
    agg,
    x="region",
    y="total_sales",
    title="Продажи по регионам",
    labels={"region": "Регион", "total_sales": "Продажи"},
)
tool_result = chart.result(fig, artifact_name="sales_by_region")
tool_result
```

Если нужно несколько графиков из одних данных:
```python
fig1 = px.bar(agg, x="region", y="total_sales", title="Продажи по регионам")
fig2 = px.line(agg, x="region", y="avg_order", title="Средний чек по регионам")
chart.result(fig1, "sales_bar")
chart.result(fig2, "avg_order_line")
# tool_result — последний вызов chart.result или любой из них
tool_result = chart.result(fig2, "avg_order_line")
tool_result
```

---

### Сценарий 3: данные из БД через `db.query_dataframe()` (режим db_connection)

Используй когда сессия подключена к БД и нужно сделать SQL-выборку прямо внутри графика.
Это эффективнее чем sql_tool → plotly_tool, потому что данные не покидают scope.

IMPORTANT: `db.query_dataframe(sql)` возвращает `pd.DataFrame`. Передавай его прямо в `px.*`.

```python
df_agg = db.query_dataframe("""
    SELECT region, SUM(revenue) AS total_revenue
    FROM sales
    WHERE year = 2024
    GROUP BY region
    ORDER BY total_revenue DESC
    LIMIT 20
""")
fig = px.bar(
    df_agg,
    x="region",
    y="total_revenue",
    title="Выручка по регионам 2024",
    labels={"region": "Регион", "total_revenue": "Выручка"},
)
tool_result = chart.result(fig, artifact_name="revenue_2024")
tool_result
```

---

### Сценарий 4: временной ряд (line chart)

```python
fig = px.line(
    df,
    x="date",
    y="value",
    color="category",      # разные линии по категориям (если нужно)
    title="Динамика показателя",
    labels={"date": "Дата", "value": "Значение", "category": "Категория"},
    markers=True,
)
tool_result = chart.result(fig, artifact_name="trend_line")
tool_result
```

---

### Сценарий 5: scatter plot (корреляция)

```python
fig = px.scatter(
    df,
    x="col_x",
    y="col_y",
    color="group_col",    # опционально — цвет по группе
    size="size_col",      # опционально — размер точки
    hover_data=["id_col"],
    title="Корреляция X и Y",
    trendline="ols",      # линия тренда (опционально)
)
tool_result = chart.result(fig, artifact_name="correlation_scatter")
tool_result
```

---

### Типичные ошибки и как их избежать

| Ошибка | Причина | Исправление |
|---|---|---|
| `NameError: result_df` | переменная из истории диалога, не из scope | используй `df`, переменные из sandbox или `db.query_dataframe()` |
| `artifact_type: plot — ожидается Figure` | передан DataFrame или dict вместо fig | `fig = px.bar(...); chart.result(fig, ...)` |
| `KeyError: 'col_name'` | неверное имя колонки | проверь точные имена в блоке «Доступные переменные в sandbox» или через `df.columns` |
| `tool_result не найден` | нет последней строки `tool_result` | добавь `tool_result` как последнюю строку |
