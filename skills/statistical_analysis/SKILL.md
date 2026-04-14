---
name: Статистический анализ
description: Гипотезы, регрессия, ANOVA, корреляции — полный статистический пакет с интерпретацией результатов.
triggers: статистика, гипотеза, регрессия, regression, anova, корреляция pearson, spearman, t-test, chi-square, хи-квадрат, нормальность, линейная зависимость, статистический тест
---

## Статистический анализ

Используй для строгой статистической проверки гипотез, моделирования зависимостей и сравнения групп.

### Тест нормальности через pandas_tool
```python
from scipy import stats

num_cols = df.select_dtypes(include="number").columns.tolist()
normality_rows = []
for col in num_cols[:8]:
    s = df[col].dropna()
    if len(s) < 8:
        continue
    # Shapiro-Wilk для n < 5000, иначе D'Agostino
    if len(s) <= 5000:
        stat, p = stats.shapiro(s.sample(min(len(s), 5000), random_state=42))
        test_name = "Shapiro-Wilk"
    else:
        stat, p = stats.normaltest(s)
        test_name = "D'Agostino"
    normality_rows.append({
        "column": col,
        "test": test_name,
        "statistic": round(stat, 4),
        "p_value": round(p, 5),
        "is_normal": p > 0.05,
        "recommendation": "параметрические тесты" if p > 0.05 else "непараметрические тесты",
    })

normality_df = pd.DataFrame(normality_rows)
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"normality_tests": normality_df}
}
tool_result
```

### Корреляционный анализ через pandas_tool
```python
from scipy import stats

num_cols = df.select_dtypes(include="number").columns.tolist()
corr_rows = []
for i, col1 in enumerate(num_cols):
    for col2 in num_cols[i+1:]:
        pair = df[[col1, col2]].dropna()
        if len(pair) < 5:
            continue
        r_p, p_p = stats.pearsonr(pair[col1], pair[col2])
        r_s, p_s = stats.spearmanr(pair[col1], pair[col2])
        strength = (
            "сильная" if abs(r_p) >= 0.7 else
            "умеренная" if abs(r_p) >= 0.4 else
            "слабая"
        )
        corr_rows.append({
            "col1": col1, "col2": col2,
            "pearson_r": round(r_p, 4), "pearson_p": round(p_p, 5),
            "spearman_r": round(r_s, 4), "spearman_p": round(p_s, 5),
            "strength": strength,
            "significant": p_p < 0.05,
        })

corr_df = pd.DataFrame(corr_rows).sort_values("pearson_r", key=abs, ascending=False) if corr_rows else pd.DataFrame()
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"correlations": corr_df}
}
tool_result
```

### Линейная регрессия через pandas_tool
```python
from scipy import stats as sp_stats

# Пример: предсказать target_col по predictor_col
# Адаптируй под реальный вопрос пользователя
num_cols = df.select_dtypes(include="number").columns.tolist()
if len(num_cols) >= 2:
    y_col = num_cols[-1]   # последняя числовая — целевая
    x_col = num_cols[0]    # первая числовая — предиктор

    pair = df[[x_col, y_col]].dropna()
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(pair[x_col], pair[y_col])

    reg_summary = pd.DataFrame([{
        "predictor": x_col,
        "target": y_col,
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "r_squared": round(r_value**2, 4),
        "p_value": round(p_value, 6),
        "std_err": round(std_err, 4),
        "significant": p_value < 0.05,
    }])

    tool_result = {
        "schema_version": "1.0",
        "artifact_type": "table",
        "items": {"regression": reg_summary}
    }
else:
    tool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"note": pd.DataFrame([{"message": "Нужно минимум 2 числовые колонки"}])}}
tool_result
```

### Визуализация регрессии через plotly_tool
```python
# pair, x_col, y_col, slope, intercept доступны из предыдущего pandas_tool
fig = px.scatter(pair, x=x_col, y=y_col, opacity=0.6, title=f"Регрессия: {y_col} ~ {x_col}")
x_range = [pair[x_col].min(), pair[x_col].max()]
fig.add_trace(__import__("plotly.graph_objects", fromlist=["Scatter"]).Scatter(
    x=x_range,
    y=[slope * x + intercept for x in x_range],
    mode="lines",
    name=f"y = {round(slope,3)}x + {round(intercept,3)}",
    line=dict(color="red", dash="dash"),
))
tool_result = chart.result(fig, artifact_name="regression_plot")
tool_result
```

### Правила
- Всегда начинай с теста нормальности — он определяет какие тесты применять дальше
- R² < 0.3 — слабая модель, предупреди пользователя
- p-value интерпретируй осторожно: p > 0.05 означает "нет доказательств эффекта", не "эффекта нет"
- Для регрессии с несколькими предикторами используй `sklearn.linear_model.LinearRegression` через pandas_tool
