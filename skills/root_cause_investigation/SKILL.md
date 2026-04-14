---
name: Анализ первопричин
description: Систематическое расследование изменений метрики — z-score валидация, drill-down по измерениям, hypothesis testing, waterfall.
triggers: первопричина, root cause, почему упало, почему выросло, что изменилось, drill down, вклад сегментов, изменение метрики, расследование, decomposition
---

## Анализ первопричин

Используй, когда нужно понять ПОЧЕМУ метрика изменилась. Сначала подтверждает что изменение реально (z-score), затем находит сегменты-виновники через drill-down и проверяет гипотезы.

### Шаг 1 — валидация изменения (z-score vs 30-дневный базис) через pandas_tool

Предполагается: в `df` есть `date` (или `period`), числовая метрика и одно или несколько измерений.

```python
date_col = next((c for c in df.columns if "date" in c.lower() or df[c].dtype == "datetime64[ns]"), df.columns[0])
metric_col = df.select_dtypes(include="number").columns[0]
dim_cols = [c for c in df.columns if c not in [date_col, metric_col] and df[c].dtype == object]

df[date_col] = pd.to_datetime(df[date_col])
df_sorted = df.sort_values(date_col)

# Делим на baseline (первые 2/3) и investigation (последняя 1/3)
cutoff = df_sorted[date_col].quantile(0.67)
baseline = df_sorted[df_sorted[date_col] < cutoff][metric_col]
current = df_sorted[df_sorted[date_col] >= cutoff][metric_col]

baseline_mean = baseline.mean()
baseline_std = baseline.std()
current_mean = current.mean()

absolute_change = current_mean - baseline_mean
pct_change = round(absolute_change / baseline_mean * 100, 2) if baseline_mean != 0 else None
z_score = round((current_mean - baseline_mean) / baseline_std, 2) if baseline_std > 0 else None

significance = (
    "🔴 Значимое (|Z| > 2)" if z_score and abs(z_score) > 2
    else "🟡 Умеренное (|Z| 1–2)" if z_score and abs(z_score) > 1
    else "🟢 В пределах нормы"
)

validation_df = pd.DataFrame([{
    "baseline_mean": round(baseline_mean, 3),
    "current_mean": round(current_mean, 3),
    "absolute_change": round(absolute_change, 3),
    "pct_change": pct_change,
    "z_score": z_score,
    "significance": significance,
    "investigation_cutoff": str(cutoff.date()) if hasattr(cutoff, "date") else str(cutoff),
}])

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"change_validation": validation_df}
}
tool_result
```

### Шаг 2 — drill-down вклада сегментов через pandas_tool
```python
mid_date = df_sorted[date_col].quantile(0.67)
prev_period = df_sorted[df_sorted[date_col] < mid_date]
curr_period = df_sorted[df_sorted[date_col] >= mid_date]
total_delta = curr_period[metric_col].sum() - prev_period[metric_col].sum()

contribution_frames = {}
for dim in dim_cols[:3]:
    prev_agg = prev_period.groupby(dim)[metric_col].sum().rename("prev")
    curr_agg = curr_period.groupby(dim)[metric_col].sum().rename("curr")
    pivot = pd.concat([prev_agg, curr_agg], axis=1).fillna(0)
    pivot["delta"] = pivot["curr"] - pivot["prev"]
    pivot["delta_pct"] = (pivot["delta"] / pivot["prev"].replace(0, np.nan) * 100).round(1)
    pivot["contribution_pct"] = (pivot["delta"] / abs(total_delta) * 100).round(1) if total_delta != 0 else 0
    contribution_frames[dim] = pivot.reset_index().sort_values("delta", ascending=False)

if contribution_frames:
    tool_result = {
        "schema_version": "1.0",
        "artifact_type": "table",
        "items": {f"contribution_{dim}": frame for dim, frame in contribution_frames.items()}
    }
else:
    tool_result = {
        "schema_version": "1.0",
        "artifact_type": "table",
        "items": {"note": pd.DataFrame([{"message": "Нет категориальных измерений для drill-down"}])}
    }
tool_result
```

### Шаг 3 — проверка гипотез через pandas_tool
```python
hypotheses = []

# Гипотеза 1: mix shift — изменился состав сегментов
for dim in dim_cols[:2]:
    if dim not in prev_period.columns:
        continue
    prev_dist = prev_period[dim].value_counts(normalize=True)
    curr_dist = curr_period[dim].value_counts(normalize=True)
    all_vals = set(prev_dist.index) | set(curr_dist.index)
    max_shift = max(abs(curr_dist.get(v, 0) - prev_dist.get(v, 0)) for v in all_vals)
    if max_shift > 0.05:
        top_shifted = max(all_vals, key=lambda v: abs(curr_dist.get(v, 0) - prev_dist.get(v, 0)))
        hypotheses.append({
            "hypothesis": f"Mix shift в '{dim}'",
            "evidence": f"Доля '{top_shifted}' изменилась на {max_shift*100:.1f}%",
            "strength": "Сильная" if max_shift > 0.10 else "Умеренная",
            "confirmed": True,
        })

# Гипотеза 2: volume change — изменился объём событий
vol_change = (len(curr_period) - len(prev_period)) / len(prev_period) if len(prev_period) > 0 else 0
if abs(vol_change) > 0.10:
    hypotheses.append({
        "hypothesis": "Изменение объёма",
        "evidence": f"Количество строк: {vol_change:+.1%}",
        "strength": "Сильная" if abs(vol_change) > 0.20 else "Умеренная",
        "confirmed": True,
    })

# Гипотеза 3: per-unit quality change — изменилась метрика на единицу
if len(dim_cols) > 0:
    id_col = dim_cols[0]
    prev_per_unit = prev_period[metric_col].sum() / prev_period[id_col].nunique() if prev_period[id_col].nunique() > 0 else 0
    curr_per_unit = curr_period[metric_col].sum() / curr_period[id_col].nunique() if curr_period[id_col].nunique() > 0 else 0
    per_unit_change = (curr_per_unit - prev_per_unit) / prev_per_unit if prev_per_unit > 0 else 0
    if abs(per_unit_change) > 0.05:
        hypotheses.append({
            "hypothesis": f"Изменение метрики на единицу '{id_col}'",
            "evidence": f"{metric_col} per {id_col}: {per_unit_change:+.1%}",
            "strength": "Сильная" if abs(per_unit_change) > 0.15 else "Умеренная",
            "confirmed": True,
        })

hyp_df = pd.DataFrame(hypotheses) if hypotheses else pd.DataFrame([{
    "hypothesis": "Явные гипотезы не подтверждены",
    "evidence": "Изменения в составе и объёме в пределах нормы",
    "strength": "—",
    "confirmed": False,
}])

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"hypotheses": hyp_df}
}
tool_result
```

### Шаг 4 — waterfall-визуализация вклада через plotly_tool
```python
import plotly.graph_objects as go

if contribution_frames:
    main_dim = list(contribution_frames.keys())[0]
    contrib = contribution_frames[main_dim].head(8)

    segments = contrib[main_dim].astype(str).tolist()
    deltas = contrib["delta"].tolist()

    fig = go.Figure(go.Waterfall(
        name="Вклад",
        orientation="v",
        measure=["relative"] * len(segments) + ["total"],
        x=segments + ["Итого"],
        y=deltas + [sum(deltas)],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#2ecc71"}},
        decreasing={"marker": {"color": "#e74c3c"}},
        totals={"marker": {"color": "#3498db"}},
    ))
    fig.update_layout(
        title=f"Waterfall: вклад сегментов '{main_dim}' в изменение {metric_col}",
        height=450,
    )
else:
    fig = go.Figure()
    fig.add_annotation(text="Нет данных для waterfall", x=0.5, y=0.5, showarrow=False)

tool_result = chart.result(fig, artifact_name="waterfall_contribution")
tool_result
```

### Правила
- Начинай с z-score: |Z| < 1 — изменение в пределах нормы, не ищи первопричины без подтверждения
- Анализируй максимум 3 измерения — фокусируйся на тех, где вклад наибольший
- Вклад > 80% от одного сегмента — главный виновник найден, сообщи пользователю явно
- Проверяй три типа гипотез: mix shift (изменился состав), volume (изменился объём), quality (изменилась метрика на единицу)
- После drill-down формулируй 2–3 конкретные гипотезы с подтверждающими цифрами
- Если данных меньше двух периодов — попроси уточнить что сравнивать
