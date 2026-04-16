## Root Cause Investigation

Use when you need to understand WHY a metric changed. Validates the change is real (z-score), finds guilty segments via drill-down, tests hypotheses.

### Step 1 — change validation (z-score) → pandas_tool

```python
# Auto-detect date, metric, and categorical dimension columns
date_col = next((c for c in df.columns if "date" in c.lower() or df[c].dtype == "datetime64[ns]"), df.columns[0])
metric_col = df.select_dtypes(include="number").columns[0]
dim_cols = [c for c in df.columns if c not in [date_col, metric_col] and df[c].dtype == object]

df[date_col] = pd.to_datetime(df[date_col])
df_sorted = df.sort_values(date_col)

# Split: last 30 days vs baseline (or last third if range < 60 days)
date_range_days = (df_sorted[date_col].max() - df_sorted[date_col].min()).days
if date_range_days >= 60:
    cutoff = df_sorted[date_col].max() - pd.Timedelta(days=30)
    period_label = "last 30 days"
else:
    cutoff = df_sorted[date_col].quantile(0.67)
    period_label = "last third of data"

baseline = df_sorted[df_sorted[date_col] < cutoff][metric_col]
current = df_sorted[df_sorted[date_col] >= cutoff][metric_col]

baseline_mean = baseline.mean()
baseline_std = baseline.std()
current_mean = current.mean()

absolute_change = current_mean - baseline_mean
pct_change = round(absolute_change / baseline_mean * 100, 2) if baseline_mean != 0 else None
z_score = round((current_mean - baseline_mean) / baseline_std, 2) if baseline_std > 0 else None

significance = (
    "🔴 Significant (|Z| > 2)" if z_score and abs(z_score) > 2
    else "🟡 Moderate (|Z| 1–2)" if z_score and abs(z_score) > 1
    else "🟢 Within normal range"
)

validation_df = pd.DataFrame([{
    "baseline_mean": round(baseline_mean, 3),
    "current_mean": round(current_mean, 3),
    "absolute_change": round(absolute_change, 3),
    "pct_change": pct_change,
    "z_score": z_score,
    "significance": significance,
    "investigation_period": period_label,
    "investigation_cutoff": str(cutoff.date()) if hasattr(cutoff, "date") else str(cutoff),
}])

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"change_validation": validation_df}
}
tool_result
```

### Step 2 — dimension drill-down → pandas_tool
```python
mid_date = cutoff
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
        "items": {"note": pd.DataFrame([{"message": "No categorical dimensions for drill-down"}])}
    }
tool_result
```

### Step 2.5 — top-5 contributors → pandas_tool
```python
all_contributions = []
for dim, frame in contribution_frames.items():
    for _, row in frame.iterrows():
        all_contributions.append({
            "dimension": dim,
            "segment": str(row[dim]),
            "delta": row["delta"],
            "contribution_pct": row["contribution_pct"],
        })

if all_contributions:
    all_contrib_df = (
        pd.DataFrame(all_contributions)
        .sort_values("delta", key=abs, ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    top = all_contrib_df.iloc[0]
    main_culprit = f"{top['dimension']}='{top['segment']}' ({top['contribution_pct']:+.1f}%)"
else:
    all_contrib_df = pd.DataFrame(columns=["dimension", "segment", "delta", "contribution_pct"])
    main_culprit = "not identified"

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {
        "top_contributors": all_contrib_df,
        "main_culprit": pd.DataFrame([{"main_culprit": main_culprit}])
    }
}
tool_result
```

### Step 3 — hypothesis testing → pandas_tool
```python
from scipy import stats as sp_stats

hypotheses = []

# Hypothesis 1: mix shift in categorical dimensions
for dim in dim_cols[:2]:
    if dim not in prev_period.columns:
        continue
    prev_dist = prev_period[dim].value_counts(normalize=True)
    curr_dist = curr_period[dim].value_counts(normalize=True)
    all_vals = set(prev_dist.index) | set(curr_dist.index)
    max_shift = max(abs(curr_dist.get(v, 0) - prev_dist.get(v, 0)) for v in all_vals)
    if max_shift > 0.05:
        top_shifted = max(all_vals, key=lambda v: abs(curr_dist.get(v, 0) - prev_dist.get(v, 0)))
        hyp_entry = {
            "hypothesis": f"Mix shift in '{dim}'",
            "evidence": f"Share of '{top_shifted}' changed by {max_shift*100:.1f}%",
            "strength": "Strong" if max_shift > 0.10 else "Moderate",
            "confirmed": True,
            "chi2_p_value": None,
            "statistically_significant": None,
        }
        prev_counts = prev_period[dim].value_counts()
        curr_counts = curr_period[dim].value_counts()
        all_cats = sorted(set(prev_counts.index) | set(curr_counts.index))
        obs = np.array([
            [prev_counts.get(c, 0) for c in all_cats],
            [curr_counts.get(c, 0) for c in all_cats],
        ])
        if obs.min() >= 5:
            chi2, p_chi2, _, _ = sp_stats.chi2_contingency(obs)
            hyp_entry["chi2_p_value"] = round(p_chi2, 4)
            hyp_entry["statistically_significant"] = bool(p_chi2 < 0.05)
        else:
            hyp_entry["chi2_p_value"] = None
            hyp_entry["statistically_significant"] = False
            hyp_entry["evidence"] += " (⚠️ chi-square not applicable)"
        hypotheses.append(hyp_entry)

# Hypothesis 2: volume change
vol_change = (len(curr_period) - len(prev_period)) / len(prev_period) if len(prev_period) > 0 else 0
if abs(vol_change) > 0.10:
    hypotheses.append({
        "hypothesis": "Volume change",
        "evidence": f"Row count: {vol_change:+.1%}",
        "strength": "Strong" if abs(vol_change) > 0.20 else "Moderate",
        "confirmed": True, "chi2_p_value": None, "statistically_significant": None,
    })

# Hypothesis 3: per-unit metric quality change
if len(dim_cols) > 0:
    id_col = dim_cols[0]
    prev_per_unit = prev_period[metric_col].sum() / prev_period[id_col].nunique() if prev_period[id_col].nunique() > 0 else 0
    curr_per_unit = curr_period[metric_col].sum() / curr_period[id_col].nunique() if curr_period[id_col].nunique() > 0 else 0
    per_unit_change = (curr_per_unit - prev_per_unit) / prev_per_unit if prev_per_unit > 0 else 0
    if abs(per_unit_change) > 0.05:
        hypotheses.append({
            "hypothesis": f"Per-unit quality change '{id_col}'",
            "evidence": f"{metric_col} per {id_col}: {per_unit_change:+.1%}",
            "strength": "Strong" if abs(per_unit_change) > 0.15 else "Moderate",
            "confirmed": True, "chi2_p_value": None, "statistically_significant": None,
        })

hyp_df = pd.DataFrame(hypotheses) if hypotheses else pd.DataFrame([{
    "hypothesis": "No hypotheses confirmed",
    "evidence": "Composition and volume changes within normal range",
    "strength": "—", "confirmed": False, "chi2_p_value": None, "statistically_significant": None,
}])

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"hypotheses": hyp_df}
}
tool_result
```

### Step 4 — waterfall → plotly_tool
```python
import plotly.graph_objects as go

if contribution_frames:
    main_dim = list(contribution_frames.keys())[0]
    contrib = contribution_frames[main_dim].head(8)
    segments = contrib[main_dim].astype(str).tolist()
    deltas = contrib["delta"].tolist()

    fig = go.Figure(go.Waterfall(
        name="Contribution", orientation="v",
        measure=["relative"] * len(segments) + ["total"],
        x=segments + ["Total"],
        y=deltas + [sum(deltas)],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#2ecc71"}},
        decreasing={"marker": {"color": "#e74c3c"}},
        totals={"marker": {"color": "#3498db"}},
    ))
    fig.update_layout(title=f"Waterfall: '{main_dim}' contribution to {metric_col}", height=450)
else:
    fig = go.Figure()
    fig.add_annotation(text="No data for waterfall", x=0.5, y=0.5, showarrow=False)

tool_result = chart.result(fig, artifact_name="waterfall_contribution")
tool_result
```

### Rules
- Start with z-score: |Z| < 1 — change is within normal range, don't hunt for root causes
- Analyze at most 3 dimensions; surface top contributors via Step 2.5
- If one segment contributes > 80%, explicitly write "Main culprit: [segment]"
- Mix shift — ALWAYS validate with chi-square (p < 0.05) in Step 3
- Three hypothesis types: mix shift, volume, quality (metric per unit)
- If fewer than two periods in data — ask the user to clarify what to compare
