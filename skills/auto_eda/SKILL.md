---
name: Auto EDA
description: Systematic exploratory data analysis — distributions, correlations, outliers, type anomalies.
triggers: eda, exploratory analysis, data exploration, correlation, distribution, outliers, profiling, full analysis, разведочный анализ, исследование данных
---

## Auto EDA

Use for deep initial dataset analysis: distributions, correlations, outliers, and common data issues.

### Step 1 — numeric statistics → pandas_tool

```python
# Sample large datasets to keep computation fast
sample_note = ""
if len(df) > 50_000:
    df_sample = df.sample(50_000, random_state=42)
    sample_note = f" (sample 50k of {len(df):,} rows)"
else:
    df_sample = df

# Exclude ID-like columns (unique per row) from numeric analysis
num_cols = df_sample.select_dtypes(include="number").columns.tolist()
num_cols = [c for c in num_cols if df_sample[c].nunique() < len(df_sample)]

stats_rows = []
for col in num_cols:
    s = df_sample[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    outliers = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
    stats_rows.append({
        "column": col,
        "mean": round(s.mean(), 3),
        "median": round(s.median(), 3),
        "std": round(s.std(), 3),
        "min": round(s.min(), 3),
        "max": round(s.max(), 3),
        "skew": round(s.skew(), 3),
        "outliers_iqr": int(outliers),
        "outliers_pct": round(outliers / len(df_sample) * 100, 1),
    })

num_stats = pd.DataFrame(stats_rows)
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"numeric_stats": num_stats},
    "sample_note": sample_note,
}
tool_result
```

### Step 2 — correlation matrix → plotly_tool

```python
import plotly.graph_objects as go

if len(df) > 50_000:
    df_sample = df.sample(50_000, random_state=42)
else:
    df_sample = df

num_cols = df_sample.select_dtypes(include="number").columns.tolist()
num_cols = [c for c in num_cols if df_sample[c].nunique() < len(df_sample)]

if len(num_cols) >= 2:
    corr_pearson = df_sample[num_cols].corr(method="pearson").round(2)
    fig = px.imshow(
        corr_pearson, text_auto=True,
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlation Matrix (Pearson)", aspect="auto",
    )
    tool_result = chart.result(fig, artifact_name="correlation_matrix")
else:
    fig = go.Figure()
    fig.add_annotation(text="Not enough numeric columns for correlation", x=0.5, y=0.5, showarrow=False)
    tool_result = chart.result(fig, artifact_name="correlation_matrix")
tool_result
```

### Step 3 — numeric distributions → plotly_tool

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if len(df) > 50_000:
    df_sample = df.sample(50_000, random_state=42)
else:
    df_sample = df

num_cols_all = df_sample.select_dtypes(include="number").columns.tolist()
num_cols_all = [c for c in num_cols_all if df_sample[c].nunique() < len(df_sample)]

# Prioritize columns with highest coefficient of variation (most interesting distributions)
def _cv(col):
    s = df_sample[col].dropna()
    mean = s.mean()
    return s.std() / abs(mean) if mean != 0 else float("inf")

num_cols = sorted(num_cols_all, key=_cv, reverse=True)[:6]

if num_cols:
    cols_per_row = 3
    n_rows = (len(num_cols) + cols_per_row - 1) // cols_per_row
    subplot_titles = [title for col in num_cols for title in (f"{col} (box)", f"{col} (hist)")]
    fig = make_subplots(rows=n_rows * 2, cols=cols_per_row, subplot_titles=subplot_titles)
    for i, col in enumerate(num_cols):
        r_base = (i // cols_per_row) * 2
        c = i % cols_per_row + 1
        data = df_sample[col].dropna()
        fig.add_trace(go.Box(y=data, name=col, showlegend=False, boxpoints="outliers"), row=r_base + 1, col=c)
        fig.add_trace(go.Histogram(x=data, name=col, showlegend=False), row=r_base + 2, col=c)
    fig.update_layout(title="Numeric Distributions (Box + Histogram)", height=400 * n_rows)
else:
    fig = go.Figure()
    fig.add_annotation(text="No numeric columns", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(title="Numeric Distributions")

tool_result = chart.result(fig, artifact_name="distributions")
tool_result
```

### Step 4 — categorical columns → pandas_tool

```python
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

cat_rows = []
for col in cat_cols:
    n_unique = df[col].nunique()
    top_val = df[col].value_counts().index[0] if n_unique > 0 else None
    top_pct = round(df[col].value_counts().iloc[0] / len(df) * 100, 1) if n_unique > 0 else None
    cat_rows.append({
        "column": col,
        "unique_values": n_unique,
        "top_value": str(top_val),
        "top_pct": top_pct,
        "null_pct": round(df[col].isna().mean() * 100, 1),
        "likely_type": (
            "ID" if n_unique == len(df) else
            "binary" if n_unique == 2 else
            "low_cardinality" if n_unique <= 10 else
            "high_cardinality"
        ),
    })

cat_stats = pd.DataFrame(cat_rows) if cat_rows else pd.DataFrame(columns=["column", "unique_values", "top_value", "top_pct", "null_pct", "likely_type"])
tool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"categorical_stats": cat_stats}}
tool_result
```

### Step 5 — automated observations → pandas_tool

```python
import numpy as np

observations = []
num_cols = df.select_dtypes(include="number").columns.tolist()
num_cols = [c for c in num_cols if df[c].nunique() < len(df)]

# Flag columns with >10% missing values
for col in df.columns:
    null_pct = df[col].isna().mean() * 100
    if null_pct > 10:
        observations.append({"type": "missing_values", "column(s)": col, "value": f"{null_pct:.1f}%", "recommendation": "Consider imputation or dropping"})

# Flag highly correlated pairs (Pearson > 0.7)
if len(num_cols) >= 2:
    df_num = df[num_cols].copy()
    if len(df_num) > 50_000:
        df_num = df_num.sample(50_000, random_state=42)
    corr_matrix = df_num.corr(method="pearson").abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if pd.notna(val) and val > 0.7:
                observations.append({"type": "high_correlation", "column(s)": f"{idx} & {col}", "value": f"{val:.2f}", "recommendation": "Check multicollinearity"})

# Flag heavily skewed columns (|skew| > 2)
for col in num_cols:
    skew = df[col].dropna().skew()
    if abs(skew) > 2:
        observations.append({"type": "high_skewness", "column(s)": col, "value": f"{skew:.2f}", "recommendation": "Consider log/sqrt transform"})

# Flag columns with >5% IQR outliers
for col in num_cols:
    s = df[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    outliers_pct = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() * 100
    if outliers_pct > 5:
        observations.append({"type": "outliers", "column(s)": col, "value": f"{outliers_pct:.1f}%", "recommendation": "Check for data errors; consider winsorization"})

observations_df = pd.DataFrame(observations) if observations else pd.DataFrame(columns=["type", "column(s)", "value", "recommendation"])
tool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"observations": observations_df}}
tool_result
```

### Rules
- Sample to 50k rows for Steps 1, 2, 3 when `len(df) > 50_000`
- Exclude `likely_type == "ID"` columns from all numeric analysis
- Step 3: prioritize by coefficient of variation, not column order
- Step 5: return computed `observations_df` table, not a text list
