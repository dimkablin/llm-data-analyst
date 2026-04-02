---
name: Forecast Tool
description: Прогнозирование по компактным временным рядам.
kind: tool
tool_key: forecast_tool
triggers: прогноз, forecast, предсказание, тренд
---

## forecast_tool — прогнозирование

Вход: Python-код. Выполняется в **sandbox сессии** — все переменные сохраняются между вызовами.

### Переменные в scope
- `df` — DataFrame текущей сессии
- `forecast` — хелпер для прогнозирования
- `pd`, `np` — pandas, numpy
- Все переменные из предыдущих tool-вызовов

Если подключена БД:
- `db` — хелпер для SQL-запросов (`db.query_dataframe(sql)`)

### Контракт
```python
history = df[["date_col", "value_col"]].sort_values("date_col")
tool_result = forecast.forecast_result(
    history, time_col="date_col", value_col="value_col",
    horizon=3, artifact_name="forecast"
)
tool_result
```

### Правила
- Поддерживается один временной ряд (time_col + value_col).
- Данные должны быть отсортированы по времени.
- Последняя строка: `tool_result`.
