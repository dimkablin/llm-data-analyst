---
name: SQL Table Tool
description: Аналитические запросы к БД и CSV в DuckDB. Возвращает табличный артефакт.
kind: tool
tool_key: sql_table_tool
triggers: sql, база данных, таблица, запрос, query, database, db, выборка, джойн, join, агрегация
---

## sql_table_tool — SQL-запросы к данным

Вход: один аргумент `question` — вопрос на естественном языке.
Инструмент сам выбирает таблицу, генерирует безопасный SELECT и возвращает табличный артефакт.

### Когда использовать

- Разведка таблиц в подключённой БД: `"Покажи первые 10 строк таблицы orders"`
- Аналитические агрегации: `"Посчитай выручку по регионам за 2024 год"`
- JOIN между таблицами: `"Соедини orders и clients по client_id, покажи топ-10 клиентов"`
- Работа с CSV, загруженным в DuckDB-сессию

### Как формулировать вопрос

IMPORTANT: вопрос должен быть конкретным. Неконкретные вопросы дают плохой SQL.

Хорошие вопросы:
- `"Покажи первые 10 строк таблицы titanic"`
- `"Средний возраст по колонке Age в таблице titanic"`
- `"Количество строк в таблице bank_churn_clients"`
- `"Топ-5 категорий по сумме revenue в таблице sales"`

Плохие вопросы:
- `"Покажи данные"` — неясно какие
- `"Проанализируй"` — слишком абстрактно
- `"Сделай отчёт"` — нет конкретики

### Ограничения

- Read-only: INSERT, UPDATE, DELETE, DROP недоступны
- Максимум 200 строк в результате

---

### Как использовать результат в plotly_tool

После выполнения `sql_table_tool` результат автоматически сохраняется в sandbox под именем артефакта.
Имя переменной указано в тексте ответа: `"Результаты сохранены в переменных sandbox: \`revenue_by_region\`"`.

**IMPORTANT: используй именно то имя переменной, которое вернул sql_table_tool.**
Не придумывай новые имена (`sql_dataset`, `data`, `result` и т.п.) — они не существуют в scope.

**Путь A (рекомендуется при наличии `db`): SQL прямо внутри plotly_tool**

Один вызов вместо двух — данные и график в одном шаге:

```python
# plotly_tool — получаем данные и сразу строим график
df_agg = db.query_dataframe("""
    SELECT region, SUM(revenue) AS total_revenue
    FROM sales
    GROUP BY region
    ORDER BY total_revenue DESC
""")
fig = px.bar(df_agg, x="region", y="total_revenue", title="Выручка по регионам")
tool_result = chart.result(fig, artifact_name="revenue_by_region")
tool_result
```

**Путь Б: sql_table_tool сохраняет переменную, plotly_tool использует её по имени**

После `sql_table_tool` с артефактом `revenue_by_region` переменная уже в scope:

```python
# plotly_tool — используем переменную по имени из ответа sql_table_tool
fig = px.bar(revenue_by_region, x="region", y="total_revenue", title="Выручка по регионам")
tool_result = chart.result(fig, artifact_name="revenue_chart")
tool_result
```

**Путь В: pandas_tool создаёт переменную, затем plotly_tool использует её**

```python
# pandas_tool
agg = df.groupby("region")["revenue"].sum().reset_index()
tool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"agg": agg}}
tool_result
```

```python
# plotly_tool — переменная agg уже в scope
fig = px.bar(agg, x="region", y="revenue", title="Выручка по регионам")
tool_result = chart.result(fig, artifact_name="revenue_by_region")
tool_result
```

---

### Пример вызова

```python
# Вызов sql_table_tool (аргумент question):
"Посчитай количество заказов и среднюю сумму по каждому региону в таблице orders"
```
