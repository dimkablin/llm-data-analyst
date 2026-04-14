---
name: Аудит качества данных
description: Комплексная проверка данных — дубли, пропуски, выбросы, несоответствия типов, ссылочная целостность.
triggers: качество данных, data quality, дубли, дублирование, пропуски, выбросы, аномалии в данных, проверка данных, dq, audit, целостность
---

## Аудит качества данных

Используй для систематической проверки качества датасета перед анализом или при подозрении на проблемы в данных.

### Шаг 1 — сводный DQ-отчёт через pandas_tool
```python
dq_rows = []
for col in df.columns:
    s = df[col]
    null_cnt = s.isna().sum()
    dup_in_col = s.duplicated().sum()
    n_unique = s.nunique()
    dtype = str(s.dtype)

    issues = []
    severity = "ok"

    if null_cnt / len(df) > 0.5:
        issues.append("критические пропуски (>50%)")
        severity = "critical"
    elif null_cnt / len(df) > 0.1:
        issues.append(f"пропуски {round(null_cnt/len(df)*100,1)}%")
        severity = "warning" if severity == "ok" else severity

    if n_unique == 1:
        issues.append("константа (1 уникальное значение)")
        severity = "warning" if severity == "ok" else severity

    if n_unique == len(df) and dtype == "object":
        issues.append("вероятно ID-колонка")

    if dtype in ("float64", "int64"):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = ((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum()
        if outliers > 0:
            issues.append(f"экстремальные выбросы: {outliers} строк")
            severity = "warning" if severity == "ok" else severity

    dq_rows.append({
        "column": col,
        "dtype": dtype,
        "null_count": null_cnt,
        "null_pct": round(null_cnt / len(df) * 100, 1),
        "unique_values": n_unique,
        "severity": severity,
        "issues": "; ".join(issues) if issues else "—",
    })

dq_report = pd.DataFrame(dq_rows).sort_values("severity", key=lambda x: x.map({"critical": 0, "warning": 1, "ok": 2}))

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"dq_report": dq_report}
}
tool_result
```

### Шаг 2 — дубликаты через pandas_tool
```python
total_dups = df.duplicated().sum()
dup_rows = df[df.duplicated(keep=False)].head(20) if total_dups > 0 else pd.DataFrame()

summary_dup = pd.DataFrame([{
    "total_rows": len(df),
    "duplicate_rows": total_dups,
    "duplicate_pct": round(total_dups / len(df) * 100, 2),
    "unique_rows": len(df) - total_dups,
}])

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"duplicates_summary": summary_dup, "duplicate_samples": dup_rows}
}
tool_result
```

### Шаг 3 — тепловая карта проблем через plotly_tool
```python
# dq_report доступен из шага 1
severity_map = {"ok": 0, "warning": 1, "critical": 2}
dq_viz = dq_report.copy()
dq_viz["severity_score"] = dq_viz["severity"].map(severity_map)

fig = px.bar(
    dq_viz.head(20),
    x="column",
    y="null_pct",
    color="severity",
    color_discrete_map={"ok": "#2ecc71", "warning": "#f39c12", "critical": "#e74c3c"},
    title="Аудит качества данных по колонкам",
    labels={"null_pct": "% пропусков", "column": "Колонка"},
    hover_data=["issues", "unique_values"],
)
fig.update_layout(xaxis_tickangle=-45)
tool_result = chart.result(fig, artifact_name="data_quality_audit")
tool_result
```

### Правила
- Severity: `critical` — блокирует анализ, `warning` — требует внимания, `ok` — норма
- После аудита явно сообщи: "Данные пригодны для анализа" или "Требуется очистка"
- Перечисли конкретные колонки с `critical` severity и предложи действия
- Не исправляй данные автоматически — только диагностика; предложи пользователю принять решение
- Выбросы по 3×IQR (а не 1.5×IQR) — это экстремальные выбросы, которые точно требуют внимания
