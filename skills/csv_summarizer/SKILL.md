---
name: CSV Summarizer
description: Fast automatic dataset overview — types, missing values, statistics, top values, and basic visualizations.
triggers: overview, summary, describe dataset, show structure, what's in the file, initial analysis, csv summarizer, обзор, резюме, опиши датасет, покажи структуру, что в файле, первичный анализ
---

## CSV Summarizer — quick dataset overview

Use when the user has uploaded a new file and wants to understand what's in it, or when a quick initial analysis is needed without deep EDA.

### Step 0 — dataset size and sample rows → pandas_tool
```python
n_rows, n_cols = df.shape
mem_mb = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
sample = df.head(5)

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {
        "dataset_size": pd.DataFrame([{
            "rows": n_rows,
            "columns": n_cols,
            "memory_mb": mem_mb,
        }]),
        "sample_rows": sample,
    }
}
tool_result
```

### Step 1 — schema and types → pandas_tool
```python
likely_types = []
for col in df.columns:
    dtype = str(df[col].dtype)
    n_unique = df[col].nunique()
    lt = "unknown"
    if dtype != "object":
        lt = "numeric" if "int" in dtype or "float" in dtype else dtype
        if n_unique == len(df):
            lt = "ID"
    else:
        # Check if object column looks like datetime
        datetime_ratio = pd.to_datetime(df[col], errors="coerce").notna().mean()
        if datetime_ratio > 0.8:
            lt = "datetime-like"
        elif n_unique == len(df):
            lt = "ID"
        elif n_unique == 2:
            lt = "binary"
        elif n_unique <= 10:
            lt = "low_cardinality"
        else:
            lt = "high_cardinality"
    likely_types.append(lt)

info_df = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.values,
    "non_null": df.notna().sum().values,
    "null_pct": (df.isna().mean() * 100).round(1).values,
    "unique": df.nunique().values,
    "likely_type": likely_types,
})

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"schema": info_df}
}
tool_result
```

### Step 2 — descriptive statistics → pandas_tool
```python
desc = df.describe(include="all").transpose().reset_index()
desc.columns = ["column"] + list(desc.columns[1:])

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"describe": desc}
}
tool_result
```

### Step 3 — top values for categorical columns → pandas_tool
```python
import numpy as np

cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

def _col_entropy(series):
    probs = series.value_counts(normalize=True).values
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

# Prioritize top-5 by entropy (most informative columns), not column order
skipped_columns = []
selected_cols = cat_cols
if len(cat_cols) > 5:
    entropies = {col: _col_entropy(df[col].dropna()) for col in cat_cols}
    sorted_cols = sorted(entropies, key=entropies.get, reverse=True)
    selected_cols = sorted_cols[:5]
    skipped_columns = sorted_cols[5:]

rows = []
for col in selected_cols:
    top = df[col].value_counts().head(3)
    for val, cnt in top.items():
        rows.append({"column": col, "value": str(val), "count": cnt, "pct": round(cnt / len(df) * 100, 1)})

top_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["column", "value", "count", "pct"])
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"top_values": top_df},
    "skipped_columns": skipped_columns,
}
tool_result
```

### Step 4 — missing values chart and numeric histograms → plotly_tool
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

null_pct = (df.isna().mean() * 100).sort_values(ascending=False)
null_pct = null_pct[null_pct > 0]

if len(null_pct) > 0:
    fig_missing = px.bar(
        x=null_pct.index,
        y=null_pct.values,
        labels={"x": "Column", "y": "% missing"},
        title="Missing Values by Column (%)",
        color=null_pct.values,
        color_continuous_scale="Reds",
    )
else:
    fig_missing = go.Figure()
    fig_missing.add_annotation(text="No missing values", x=0.5, y=0.5, showarrow=False, font=dict(size=18))
    fig_missing.update_layout(title="Missing Values by Column")

tool_result = chart.result(fig_missing, artifact_name="missing_values")
tool_result
```

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Histograms for up to 4 numeric columns, excluding IDs
num_cols = df.select_dtypes(include="number").columns.tolist()
num_cols = [c for c in num_cols if df[c].nunique() < len(df)][:4]

if len(num_cols) > 0:
    cols_per_row = min(2, len(num_cols))
    n_rows = (len(num_cols) + cols_per_row - 1) // cols_per_row
    fig_hist = make_subplots(rows=n_rows, cols=cols_per_row, subplot_titles=num_cols)
    for i, col in enumerate(num_cols):
        r, c = divmod(i, cols_per_row)
        fig_hist.add_trace(go.Histogram(x=df[col].dropna(), name=col, showlegend=False), row=r+1, col=c+1)
    fig_hist.update_layout(title="Numeric Column Histograms", height=320 * n_rows)
else:
    fig_hist = go.Figure()
    fig_hist.add_annotation(text="No numeric columns", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
    fig_hist.update_layout(title="Numeric Column Histograms")

tool_result = chart.result(fig_hist, artifact_name="numeric_histograms")
tool_result
```

### Rules
- ALWAYS start with Step 0 — the agent needs scale context before proceeding
- If dataset > 500 columns — show only Steps 0 and 1, then ask which blocks are needed
- In Step 3 use top-5 by entropy (not first 5 by order) when there are more than 5 columns; list skipped in `skipped_columns`
- Numeric columns with `unique == len(df)` — likely IDs; mark as such in `likely_type` and exclude from histograms
- Object columns with `likely_type == "datetime-like"` — warn that they should be cast to datetime
