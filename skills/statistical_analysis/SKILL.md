---
name: Statistical Analysis
description: Hypothesis testing, regression, ANOVA, correlations — full statistical toolkit with result interpretation.
triggers: statistics, hypothesis, regression, anova, correlation, pearson, spearman, t-test, chi-square, normality, linear dependence, statistical test, статистика, гипотеза, регрессия, корреляция, нормальность
---

## Statistical Analysis

Use for rigorous hypothesis testing, dependency modeling, and group comparison.

Order: normality first → if normal → ANOVA + t-test + Pearson; if not → Kruskal-Wallis + Mann-Whitney + Spearman.

### Normality test → pandas_tool

```python
from scipy import stats

num_cols = df.select_dtypes(include="number").columns.tolist()
normality_rows = []
for col in num_cols[:8]:
    s = df[col].dropna()
    if len(s) < 8:
        continue
    # Shapiro-Wilk for n≤5000, D'Agostino for larger samples
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
        "recommendation": "parametric tests" if p > 0.05 else "non-parametric tests",
    })

normality_df = pd.DataFrame(normality_rows)
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"normality_tests": normality_df}
}
tool_result
```

### Correlation analysis → pandas_tool

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
        strength = "strong" if abs(r_p) >= 0.7 else ("moderate" if abs(r_p) >= 0.4 else "weak")
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

### One-Way ANOVA → pandas_tool

```python
from scipy import stats

cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = df.select_dtypes(include="number").columns.tolist()

anova_results = []
if cat_cols and num_cols:
    group_col = cat_cols[0]
    metric_col = num_cols[0]
    groups = [group[metric_col].dropna().values for _, group in df.groupby(group_col) if len(group) >= 5]

    if len(groups) >= 2:
        f_stat, p_anova = stats.f_oneway(*groups)
        anova_results.append({
            "test": "One-Way ANOVA",
            "group_col": group_col,
            "metric_col": metric_col,
            "n_groups": len(groups),
            "f_statistic": round(f_stat, 4),
            "p_value": round(p_anova, 5),
            "significant": p_anova < 0.05,
            "interpretation": (
                f"Significant differences between groups in {group_col}" if p_anova < 0.05
                else f"No significant differences between groups in {group_col}"
            ),
        })

anova_df = pd.DataFrame(anova_results) if anova_results else pd.DataFrame([{"message": "No suitable columns for ANOVA"}])
tool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"anova": anova_df}}
tool_result
```

### Chi-Square → pandas_tool

```python
from scipy import stats

cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
chi2_results = []

for i, col1 in enumerate(cat_cols[:4]):
    for col2 in cat_cols[i+1:4]:
        ct = pd.crosstab(df[col1], df[col2])
        if ct.min().min() >= 5:  # Expected cell frequency ≥ 5
            chi2_stat, p_val, dof, expected = stats.chi2_contingency(ct)
            chi2_results.append({
                "col1": col1, "col2": col2,
                "chi2_statistic": round(chi2_stat, 4),
                "p_value": round(p_val, 5),
                "degrees_of_freedom": dof,
                "significant": p_val < 0.05,
                "interpretation": f"{'Dependent' if p_val < 0.05 else 'Independent'}: {col1} vs {col2}",
            })

chi2_df = pd.DataFrame(chi2_results) if chi2_results else pd.DataFrame([{"message": "No suitable categorical column pairs"}])
tool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"chi2_tests": chi2_df}}
tool_result
```

### Linear regression → pandas_tool

```python
from scipy import stats as sp_stats

num_cols = df.select_dtypes(include="number").columns.tolist()
if len(num_cols) >= 2:
    # Auto-select the pair with highest absolute correlation
    corr_matrix = df[num_cols].corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    max_corr_pair = corr_matrix.stack().idxmax()
    x_col, y_col = max_corr_pair[0], max_corr_pair[1]

    pair = df[[x_col, y_col]].dropna()
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(pair[x_col], pair[y_col])

    reg_summary = pd.DataFrame([{
        "predictor": x_col, "target": y_col,
        "slope": round(slope, 4), "intercept": round(intercept, 4),
        "r_squared": round(r_value**2, 4), "p_value": round(p_value, 6),
        "std_err": round(std_err, 4), "significant": p_value < 0.05,
    }])
    tool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"regression": reg_summary}}
else:
    tool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"note": pd.DataFrame([{"message": "Need at least 2 numeric columns"}])}}
tool_result
```

### Regression visualization → plotly_tool

```python
fig = px.scatter(pair, x=x_col, y=y_col, opacity=0.6, title=f"Regression: {y_col} ~ {x_col}")
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

### Residuals visualization → plotly_tool

```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy import stats as sp_stats

predicted = slope * pair[x_col] + intercept
residuals = pair[y_col] - predicted

fig = make_subplots(rows=1, cols=2, subplot_titles=["Residuals vs Fitted", "Q-Q Plot"])
fig.add_trace(go.Scatter(x=predicted, y=residuals, mode="markers",
    marker=dict(opacity=0.5, color="#3498db"), name="Residuals"), row=1, col=1)
fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

(osm, osr), (slope_qq, intercept_qq, r_qq) = sp_stats.probplot(residuals)
fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Q-Q", marker=dict(color="#2ecc71")), row=1, col=2)
fig.add_trace(go.Scatter(
    x=[min(osm), max(osm)],
    y=[slope_qq * min(osm) + intercept_qq, slope_qq * max(osm) + intercept_qq],
    mode="lines", name="Normal", line=dict(color="red", dash="dash"),
), row=1, col=2)

fig.update_layout(title="Residual Analysis", height=400)
tool_result = chart.result(fig, artifact_name="residuals")
tool_result
```

### Rules
- Start with normality test — it determines which tests to apply
- Regression — only for the pair with highest correlation
- ALWAYS visualize regression residuals
- R² < 0.3 — weak model, warn the user
- p > 0.05 means "no evidence of an effect", not "no effect"
