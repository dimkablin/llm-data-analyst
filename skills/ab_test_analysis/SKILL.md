---
name: A/B тест анализ
description: Статистически строгий анализ A/B тестов — проверка SRM, proportion z-test, power analysis (Cohen's h), guardrail-метрики.
triggers: a/b тест, ab тест, ab test, контрольная группа, тестовая группа, statistical significance, конверсия, эксперимент, hypothesis test, значимость, srm, sample ratio mismatch
---

## A/B тест анализ

Используй для проверки гипотез по результатам A/B экспериментов. Проверяет SRM, вычисляет статистическую значимость, power и даёт чёткую рекомендацию SHIP / DO_NOT_SHIP / INCONCLUSIVE.

### Шаг 1 — SRM-проверка и базовые метрики через pandas_tool

Предполагается: в `df` есть колонка-группа (например `variant`: "control"/"treatment") и числовая метрика (например `converted`: 0/1 или `revenue`).

```python
from scipy import stats

# Авто-определение колонок
group_col = next(
    (c for c in df.columns if df[c].nunique() == 2 and df[c].dtype == object),
    df.columns[0]
)
metric_col = df.select_dtypes(include="number").columns[0]

g0, g1 = df[group_col].unique()[0], df[group_col].unique()[1]
control_data = df[df[group_col] == g0][metric_col].dropna()
treatment_data = df[df[group_col] == g1][metric_col].dropna()
n_control, n_treatment = len(control_data), len(treatment_data)
total = n_control + n_treatment

# SRM check — chi-square at alpha=0.001 (critical value 10.828)
expected_n = total * 0.5
chi2_stat = ((n_control - expected_n)**2 / expected_n +
             (n_treatment - expected_n)**2 / expected_n)
p_srm = 1 - stats.chi2.cdf(chi2_stat, df=1)
srm_detected = chi2_stat > 10.828

rate_ctrl = control_data.mean()
rate_trt = treatment_data.mean()
se_ctrl = np.sqrt(rate_ctrl * (1 - rate_ctrl) / n_control)
se_trt = np.sqrt(rate_trt * (1 - rate_trt) / n_treatment)

metrics_df = pd.DataFrame([
    {"variant": str(g0), "n": n_control, "conversions": int(control_data.sum()),
     "rate": round(rate_ctrl, 4),
     "ci_lower": round(rate_ctrl - 1.96 * se_ctrl, 4),
     "ci_upper": round(rate_ctrl + 1.96 * se_ctrl, 4)},
    {"variant": str(g1), "n": n_treatment, "conversions": int(treatment_data.sum()),
     "rate": round(rate_trt, 4),
     "ci_lower": round(rate_trt - 1.96 * se_trt, 4),
     "ci_upper": round(rate_trt + 1.96 * se_trt, 4)},
])
srm_df = pd.DataFrame([{
    "actual_split": f"{n_control/total*100:.1f}% / {n_treatment/total*100:.1f}%",
    "expected_split": "50% / 50%",
    "chi2_stat": round(chi2_stat, 3),
    "p_value": round(p_srm, 4),
    "srm_detected": srm_detected,
    "status": "⚠️ SRM обнаружен — рандомизация нарушена" if srm_detected else "✅ SRM не обнаружен",
}])

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"variant_metrics": metrics_df, "srm_check": srm_df}
}
tool_result
```

### Шаг 2 — статистическая значимость и power через pandas_tool
```python
from scipy import stats

# Two-proportion z-test (основной тест для конверсий)
pooled_p = (control_data.sum() + treatment_data.sum()) / total
pooled_se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n_control + 1/n_treatment))
diff = rate_trt - rate_ctrl
z_score = diff / pooled_se if pooled_se > 0 else 0
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

# Для непрерывных метрик (revenue, duration) дополнительно применяй t-test:
# t_stat, p_ttest = stats.ttest_ind(control_data, treatment_data, equal_var=False)

relative_uplift = diff / rate_ctrl if rate_ctrl > 0 else 0
se_diff = np.sqrt(rate_ctrl*(1-rate_ctrl)/n_control + rate_trt*(1-rate_trt)/n_treatment)
ci_lower_diff = diff - 1.96 * se_diff
ci_upper_diff = diff + 1.96 * se_diff

# Power (Cohen's h для пропорций)
effect_h = 2 * (np.arcsin(np.sqrt(max(0, min(1, rate_trt)))) - np.arcsin(np.sqrt(max(0, min(1, rate_ctrl)))))
z_crit = stats.norm.ppf(0.975)
ncp = abs(diff) / se_diff if se_diff > 0 else 0
power = float(1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp))

if srm_detected:
    recommendation = "DO_NOT_SHIP — SRM обнаружен"
elif power < 0.70:
    recommendation = f"INCONCLUSIVE — недостаточная мощность ({power:.0%})"
elif p_value < 0.05 and diff > 0:
    recommendation = "SHIP ✅"
elif p_value < 0.05 and diff < 0:
    recommendation = "DO_NOT_SHIP ❌"
else:
    recommendation = "INCONCLUSIVE ⚠️"

sig_df = pd.DataFrame([{
    "absolute_uplift": round(diff, 4),
    "relative_uplift_pct": round(relative_uplift * 100, 2),
    "ci_95_diff": f"[{round(ci_lower_diff, 4)}, {round(ci_upper_diff, 4)}]",
    "z_score": round(z_score, 3),
    "p_value": round(p_value, 5),
    "significant": p_value < 0.05,
    "cohens_h": round(effect_h, 3),
    "achieved_power": round(power, 3),
    "recommendation": recommendation,
}])

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"significance_and_power": sig_df}
}
tool_result
```

### Шаг 3 — визуализация через plotly_tool
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=["Conversion Rate (95% CI)", "Относительный эффект"])

colors = ["#3498db", "#2ecc71" if diff > 0 else "#e74c3c"]
for i, (gname, rate, se) in enumerate([(str(g0), rate_ctrl, se_ctrl), (str(g1), rate_trt, se_trt)]):
    fig.add_trace(go.Bar(
        name=gname, x=[gname], y=[rate],
        error_y=dict(type="data", array=[1.96 * se]),
        marker_color=colors[i],
        text=[f"{rate:.2%}"], textposition="outside",
    ), row=1, col=1)

fig.add_trace(go.Bar(
    x=[relative_uplift * 100], y=["Uplift"], orientation="h",
    marker_color="#2ecc71" if diff > 0 else "#e74c3c",
    text=[f"{relative_uplift*100:+.1f}%  p={p_value:.4f}"],
    textposition="outside",
), row=1, col=2)
fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=2)

fig.update_layout(title=f"A/B анализ: {metric_col}", height=400, showlegend=False)
tool_result = chart.result(fig, artifact_name="ab_test_results")
tool_result
```

### Правила
- Всегда проверяй SRM первым — при `srm_detected=True` результаты ненадёжны; расследуй рандомизацию
- Для бинарных метрик (конверсия) — proportion z-test; для непрерывных (revenue, duration) — Welch t-test
- `SHIP` только при: нет SRM + `p < 0.05` + `power > 0.70` + `diff > 0`
- Power < 0.70 — тест недостаточно мощный; нужно больше данных или пересмотреть MDE
- Относительный лифт = `(treatment - control) / control × 100%`
- Если `n < 100` в любой группе — предупреди о малой выборке
- p-value близко к 0.05 (0.04–0.06) — проверь CI: если он пересекает ноль, доверие низкое
