## Data Quality Audit

Use for systematic dataset quality checks before analysis or when data issues are suspected.

### Step 0 — source preparation → general_analytics + sql_tool

Load `get_tool_instructions("general_analytics")` first and follow its source context
and data extraction workflow. If the current sandbox already lists a dataframe artifact,
use that exact variable name. Otherwise inspect the source with `sql_tool` and create a
stable table artifact, then use the returned artifact variable in the examples below.

In snippets below, `df` means the actual dataframe artifact variable from this step.

### Step 1 — DQ report per column → pandas_tool
```python
dq_rows = []
for col in df.columns:
    s = df[col]
    null_cnt = s.isna().sum()
    n_unique = s.nunique()
    dtype = str(s.dtype)

    issues = []
    severity = "ok"

    if null_cnt / len(df) > 0.5:
        issues.append("critical missing (>50%)")
        severity = "critical"
    elif null_cnt / len(df) > 0.1:
        issues.append(f"missing {round(null_cnt/len(df)*100,1)}%")
        severity = "warning" if severity == "ok" else severity

    if n_unique == 1:
        issues.append("constant (1 unique value)")
        severity = "warning" if severity == "ok" else severity

    if n_unique == len(df) and dtype == "object":
        issues.append("likely ID column")

    # Extreme outliers: 3×IQR (more conservative than 1.5×IQR)
    if dtype in ("float64", "int64"):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = ((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum()
        if outliers > 0:
            issues.append(f"extreme outliers: {outliers} rows")
            severity = "warning" if severity == "ok" else severity

    # Detect numeric values stored as object
    if dtype == "object":
        numeric_parseable = pd.to_numeric(s, errors='coerce').notna().sum()
        numeric_ratio = numeric_parseable / len(s.dropna()) if len(s.dropna()) > 0 else 0
        if numeric_ratio > 0.8:
            issues.append(f"likely numeric (dtype=object, {numeric_ratio*100:.0f}% parses as number)")
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

### Step 2 — duplicates → pandas_tool
```python
total_dups = df.duplicated().sum()
dup_rows = df[df.duplicated(keep=False)].head(20) if total_dups > 0 else pd.DataFrame()

summary_dup = pd.DataFrame([{
    "total_rows": len(df),
    "duplicate_rows": total_dups,
    "duplicate_pct": round(total_dups / len(df) * 100, 2),
    "unique_rows": len(df) - total_dups,
}])

# Key-based duplicate check on probable ID columns
id_candidates = [c for c in df.columns if df[c].dtype == "object" and df[c].nunique() > len(df) * 0.9]
key_dup_results = []
for id_col in id_candidates[:3]:
    key_dups = df[df.duplicated(subset=[id_col], keep=False)]
    if len(key_dups) > 0:
        key_dup_results.append({
            "key_column": id_col,
            "duplicate_keys": df[id_col].duplicated().sum(),
            "affected_rows": len(key_dups),
            "severity": "critical" if len(key_dups) > 0 else "ok",
            "sample_key": str(df[id_col][df[id_col].duplicated()].iloc[0]) if df[id_col].duplicated().any() else "—",
        })

key_dup_df = pd.DataFrame(key_dup_results) if key_dup_results else pd.DataFrame([{"message": "No ID columns found"}])

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"duplicates_summary": summary_dup, "duplicate_samples": dup_rows, "key_duplicates": key_dup_df}
}
tool_result
```

### Step 2.5 — cross-column validation → pandas_tool
```python
cross_issues = []

# Date pair check: end must not be before start
date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
start_cols = [c for c in date_cols if "start" in c.lower() or "begin" in c.lower() or "from" in c.lower() or "created" in c.lower()]
end_cols = [c for c in date_cols if "end" in c.lower() or "finish" in c.lower() or "to" in c.lower() or "closed" in c.lower()]

for sc in start_cols:
    for ec in end_cols:
        try:
            s_dt = pd.to_datetime(df[sc], errors='coerce')
            e_dt = pd.to_datetime(df[ec], errors='coerce')
            invalid = ((e_dt < s_dt) & s_dt.notna() & e_dt.notna()).sum()
            if invalid > 0:
                cross_issues.append({
                    "check": f"{ec} < {sc}",
                    "invalid_rows": int(invalid),
                    "pct": round(invalid / len(df) * 100, 2),
                    "severity": "critical" if invalid / len(df) > 0.01 else "warning",
                })
        except Exception:
            pass

# Negative values in obviously positive columns
for col in df.select_dtypes(include="number").columns:
    if any(kw in col.lower() for kw in ["price", "amount", "revenue", "cost", "age", "count", "qty", "quantity"]):
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            cross_issues.append({
                "check": f"{col} >= 0",
                "invalid_rows": int(neg_count),
                "pct": round(neg_count / len(df) * 100, 2),
                "severity": "warning",
            })

cross_df = pd.DataFrame(cross_issues) if cross_issues else pd.DataFrame([{"check": "No violations found", "invalid_rows": 0, "pct": 0.0, "severity": "ok"}])
tool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"cross_column_validation": cross_df}}
tool_result
```

### Step 3 — issue visualization → plotly_tool
```python
from plotly.subplots import make_subplots
import re

severity_map = {"ok": 0, "warning": 1, "critical": 2}
dq_viz = dq_report.head(20).copy()
dq_viz["severity_score"] = dq_viz["severity"].map(severity_map)

def extract_outliers(issues_str):
    m = re.search(r"extreme outliers: (\d+) rows", issues_str)
    return int(m.group(1)) if m else 0

dq_viz["outliers_count"] = dq_viz["issues"].apply(extract_outliers)
has_outliers = dq_viz["outliers_count"].sum() > 0

if has_outliers:
    fig = make_subplots(rows=1, cols=2, subplot_titles=["% Missing by Severity", "Extreme Outliers (rows)"])

    for sev, color in [("critical", "#e74c3c"), ("warning", "#f39c12"), ("ok", "#2ecc71")]:
        sub = dq_viz[dq_viz["severity"] == sev]
        fig.add_trace(
            go.Bar(x=sub["column"], y=sub["null_pct"], name=sev, marker_color=color, showlegend=True),
            row=1, col=1
        )

    outlier_sub = dq_viz[dq_viz["outliers_count"] > 0]
    fig.add_trace(
        go.Bar(x=outlier_sub["column"], y=outlier_sub["outliers_count"], name="outliers",
               marker_color="#9b59b6", showlegend=True),
        row=1, col=2
    )

    fig.update_layout(
        title="Data Quality Audit by Column",
        height=450,
        xaxis=dict(tickangle=-45),
        xaxis2=dict(tickangle=-45),
    )
else:
    fig = go.Figure()
    for sev, color in [("critical", "#e74c3c"), ("warning", "#f39c12"), ("ok", "#2ecc71")]:
        sub = dq_viz[dq_viz["severity"] == sev]
        fig.add_trace(go.Bar(x=sub["column"], y=sub["null_pct"], name=sev, marker_color=color))
    fig.update_layout(
        title="Data Quality Audit — % Missing by Column",
        xaxis=dict(tickangle=-45, title="Column"),
        yaxis=dict(title="% missing"),
        height=400,
    )

tool_result = chart.result(fig, artifact_name="data_quality_audit")
tool_result
```

### Rules
- Severity: `critical` — blocks analysis, `warning` — needs attention, `ok` — normal
- ALWAYS run Step 2.5 (cross-column) — often catches critical errors in dates and amounts
- If duplicates > 5% of rows — stop and warn the user before proceeding with analysis
- Object columns with > 80% numeric values — suggest conversion via `pd.to_numeric`
- Explicitly conclude: "Data is ready for analysis" or "Cleaning required"
- List specific `critical`-severity columns and suggest actions
- DO NOT fix data automatically — diagnostics only; let the user decide
- Outliers use 3×IQR (not 1.5×IQR) — these are extreme outliers that clearly need attention
