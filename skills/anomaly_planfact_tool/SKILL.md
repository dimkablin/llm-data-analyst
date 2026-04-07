---
name: Anomaly Plan-Fact Tool
description: Анализ отклонений и план-факт по временным рядам.
kind: tool
tool_key: anomaly_planfact_tool
triggers: аномалия, план-факт, отклонение, anomaly, planfact
---

## anomaly_planfact_tool — план-факт анализ

Вход: Python-код. Выполняется в **sandbox сессии** — все переменные сохраняются между вызовами.

### Переменные в scope
- `df` — DataFrame текущей сессии
- `anomaly_planfact` — хелпер для план-факт анализа
- `pd`, `np` — pandas, numpy
- Все переменные из предыдущих tool-вызовов

Если подключена БД:
- `db` — хелпер для SQL-запросов (`db.query_dataframe(sql)`)

### Контракт
```python
history = df[["period", "plan", "fact"]].sort_values("period")
tool_result = anomaly_planfact.analyze_result(
    history, time_col="period", plan_col="plan", fact_col="fact",
    artifact_name="planfact"
)
tool_result
```

### Правила
- Один план-факт ряд (time_col + plan_col + fact_col).
- Данные отсортированы по времени.
- Последняя строка: `tool_result`.
