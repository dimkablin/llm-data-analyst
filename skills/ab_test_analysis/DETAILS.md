## A/B Test Analysis

Checks SRM, computes statistical significance and power, gives SHIP / DO_NOT_SHIP / INCONCLUSIVE recommendation.

### Step 1 — SRM and base metrics → pandas_tool

```python
from scipy import stats

# Auto-detect group column (2 unique values) and primary metric
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

# SRM check: expected 50/50 split
expected_ratio = 0.5
expected_n_ctrl = total * expected_ratio
expected_n_trt = total * (1 - expected_ratio)
chi2_stat = ((n_control - expected_n_ctrl)**2 / expected_n_ctrl +
             (n_treatment - expected_n_trt)**2 / expected_n_trt)
p_srm = 1 - stats.chi2.cdf(chi2_stat, df=1)
srm_detected = chi2_stat > 10.828  # p < 0.001 threshold

actual_ratio = n_control / total
ratio_warning = abs(actual_ratio - 0.5) > 0.05

rate_ctrl = control_data.mean()
rate_trt = treatment_data.mean()
se_ctrl = np.sqrt(max(rate_ctrl * (1 - rate_ctrl), 0) / n_control) if n_control > 0 else 0
se_trt = np.sqrt(max(rate_trt * (1 - rate_trt), 0) / n_treatment) if n_treatment > 0 else 0

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
    "expected_split": f"{expected_ratio*100:.0f}% / {(1-expected_ratio)*100:.0f}%",
    "chi2_stat": round(chi2_stat, 3),
    "p_value": round(p_srm, 4),
    "srm_detected": srm_detected,
    "status": "⚠️ SRM detected" if srm_detected else "✅ No SRM",
    "ratio_warning": "⚠️ Split far from 50/50" if ratio_warning else "—",
}])

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"variant_metrics": metrics_df, "srm_check": srm_df}
}
tool_result
```

### Step 2 — statistical significance and power → pandas_tool
```python
from scipy import stats

unique_vals = df[metric_col].dropna().unique()
is_binary = set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False})

if is_binary:
    # Proportion z-test for binary metrics (conversion rate)
    test_type = "proportion_z_test"
    pooled_p = (control_data.sum() + treatment_data.sum()) / total
    pooled_se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n_control + 1/n_treatment))
    diff = rate_trt - rate_ctrl
    z_score = diff / pooled_se if pooled_se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    relative_uplift = diff / rate_ctrl if rate_ctrl > 0 else 0
    se_diff = np.sqrt(rate_ctrl*(1-rate_ctrl)/n_control + rate_trt*(1-rate_trt)/n_treatment)
    ci_lower_diff = diff - 1.96 * se_diff
    ci_upper_diff = diff + 1.96 * se_diff
    effect_h = 2 * (np.arcsin(np.sqrt(max(0, min(1, rate_trt)))) - np.arcsin(np.sqrt(max(0, min(1, rate_ctrl)))))
    z_crit = stats.norm.ppf(0.975)
    ncp = abs(diff) / se_diff if se_diff > 0 else 0
    power = float(1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp))
else:
    # Welch t-test for continuous metrics
    test_type = "welch_t_test"
    t_stat, p_value = stats.ttest_ind(control_data, treatment_data, equal_var=False)
    diff = treatment_data.mean() - control_data.mean()
    relative_uplift = diff / control_data.mean() if control_data.mean() != 0 else 0
    ci_lower_diff = diff - 1.96 * np.sqrt(control_data.var()/n_control + treatment_data.var()/n_treatment)
    ci_upper_diff = diff + 1.96 * np.sqrt(control_data.var()/n_control + treatment_data.var()/n_treatment)
    z_score = t_stat
    pooled_std = np.sqrt(((n_control-1)*control_data.var() + (n_treatment-1)*treatment_data.var()) / (total-2))
    effect_h = diff / pooled_std if pooled_std > 0 else 0
    ncp = abs(diff) / np.sqrt(control_data.var()/n_control + treatment_data.var()/n_treatment)
    z_crit = stats.norm.ppf(0.975)
    power = float(1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp))

# Final recommendation: SHIP requires no SRM + significant + powered + positive
if srm_detected:
    recommendation = "DO_NOT_SHIP — SRM detected"
elif power < 0.70:
    recommendation = f"INCONCLUSIVE — insufficient power ({power:.0%})"
elif p_value < 0.05 and diff > 0:
    recommendation = "SHIP ✅"
elif p_value < 0.05 and diff < 0:
    recommendation = "DO_NOT_SHIP ❌"
else:
    recommendation = "INCONCLUSIVE ⚠️"

effect_label = "cohens_h" if is_binary else "cohens_d"
sig_df = pd.DataFrame([{
    "test_type": test_type,
    "absolute_uplift": round(diff, 4),
    "relative_uplift_pct": round(relative_uplift * 100, 2),
    "ci_95_diff": f"[{round(ci_lower_diff, 4)}, {round(ci_upper_diff, 4)}]",
    "z_score": round(z_score, 3),
    "p_value": round(p_value, 5),
    "significant": p_value < 0.05,
    effect_label: round(effect_h, 3),
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

### Step 2.5 — Bonferroni correction (only if multiple numeric metrics) → pandas_tool

```python
numeric_cols = df.select_dtypes(include="number").columns.tolist()
n_metrics = len(numeric_cols)

if n_metrics > 1:
    alpha_bonferroni = 0.05 / n_metrics
    bonferroni_df = pd.DataFrame([{
        "n_metrics_tested": n_metrics,
        "original_alpha": 0.05,
        "bonferroni_alpha": round(alpha_bonferroni, 5),
        "primary_metric_passes_bonferroni": bool(p_value < alpha_bonferroni),
        "warning": f"⚠️ Testing {n_metrics} metrics — use α={alpha_bonferroni:.4f}",
    }])
else:
    bonferroni_df = pd.DataFrame([{"note": "Single metric — no correction needed"}])

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"bonferroni_correction": bonferroni_df}
}
tool_result
```

### Step 3 — visualization → plotly_tool
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2, subplot_titles=["Conversion Rate (95% CI)", "Relative Effect"])

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

fig.update_layout(title=f"A/B Analysis: {metric_col}", height=400, showlegend=False)
tool_result = chart.result(fig, artifact_name="ab_test_results")
tool_result
```

### Rules
- Check SRM first — if `srm_detected=True` results are unreliable
- Test selection: binary 0/1 → proportion z-test, otherwise → Welch t-test
- Run Step 2.5 (Bonferroni) when there are multiple numeric columns
- `SHIP` requires: no SRM + `p < 0.05` + `power > 0.70` + `diff > 0`
- Warn if `n < 100` in either group — small sample
- SRM assumes 50/50; adjust `expected_ratio` if the intended split is different
