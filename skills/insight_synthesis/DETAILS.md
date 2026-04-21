## Insight Synthesis

Use as the final step after completing analysis. Collects key observations, structures them through the So What / Why / Now What framework, prioritizes by impact×confidence, and produces an executive summary.

### Step 1 — collect key metrics → value_tool
```python
# Only metrics with real variation — exclude constants
num_cols = df.select_dtypes(include="number").columns.tolist()

key_metrics = {}

for col in num_cols[:8]:
    s = df[col].dropna()
    if len(s) == 0:
        continue
    cv = s.std() / abs(s.mean()) if s.mean() != 0 else 0
    if cv > 0.01:  # skip constants
        key_metrics[f"{col}_total"] = round(s.sum(), 2)
        key_metrics[f"{col}_mean"] = round(s.mean(), 3)
        key_metrics[f"{col}_median"] = round(s.median(), 3)
        # Signal: mean/median divergence indicates skew or outliers
        if abs(s.mean() - s.median()) / (abs(s.median()) + 1e-9) > 0.2:
            key_metrics[f"{col}_skew_alert"] = f"mean/median diverge by {abs(s.mean()-s.median())/abs(s.median())*100:.0f}%"

tool_result = {"schema_version": "1.0", "artifact_type": "value", "items": key_metrics}
tool_result
```

### Step 2 — programmatic insight extraction → pandas_tool
```python
# Insights are generated PROGRAMMATICALLY from real data — NOT from a template
auto_insights = []

# 1. High missing values
for col in df.columns:
    null_pct = df[col].isna().mean() * 100
    if null_pct > 30:
        auto_insights.append({
            "priority": "🔴 Critical",
            "metric": col,
            "so_what": f"{null_pct:.1f}% missing — over a third of data absent",
            "why": "Possible causes: collection error, optional field, or partial-period data",
            "now_what": "Exclude from analysis or impute with median/mode before use",
            "confidence": "High",
            "expected_outcome": "Reduced aggregation errors",
        })

# 2. Extreme outliers (3×IQR)
for col in df.select_dtypes(include="number").columns[:5]:
    s = df[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    outliers = ((s < q1 - 3*iqr) | (s > q3 + 3*iqr)).sum()
    outlier_pct = outliers / len(df) * 100
    if outlier_pct > 1:
        auto_insights.append({
            "priority": "🟡 Important",
            "metric": col,
            "so_what": f"{outliers} extreme outliers ({outlier_pct:.1f}%) — values beyond 3×IQR",
            "why": "Data entry errors, anomalous events, or special segments",
            "now_what": f"Inspect rows where {col} < {q1 - 3*iqr:.2f} or > {q3 + 3*iqr:.2f}",
            "confidence": "High",
            "expected_outcome": "Improved means and metrics after cleaning",
        })

# 3. Dominant categorical value
for col in df.select_dtypes(include=["object", "category"]).columns[:3]:
    top_pct = df[col].value_counts(normalize=True).iloc[0] * 100 if df[col].nunique() > 0 else 0
    top_val = df[col].value_counts().index[0] if df[col].nunique() > 0 else "—"
    if top_pct > 70:
        auto_insights.append({
            "priority": "🟢 FYI",
            "metric": col,
            "so_what": f"'{top_val}' is {top_pct:.1f}% — heavily dominant category",
            "why": "Dataset may be imbalanced on this field",
            "now_what": "Account for in segmentation; analyzing other categories requires filtering",
            "confidence": "High",
            "expected_outcome": "—",
        })

# Neutral outcome if no significant findings
if not auto_insights:
    auto_insights.append({
        "priority": "🟢 FYI",
        "metric": "Entire dataset",
        "so_what": "No significant deviations found — data within normal range",
        "why": "No critical missing values, outliers, or imbalance",
        "now_what": "Ready to proceed with analytical questions",
        "confidence": "High",
        "expected_outcome": "—",
    })

# Manually append insights from previous steps in this session (root_cause, ab_test, etc.):
# auto_insights.append({
#     "priority": "🔴 Critical",
#     "metric": "<actual field from session>",
#     "so_what": "<actual fact with numbers from session>",
#     "why": "<actual hypothesis from session>",
#     "now_what": "<concrete action>",
#     "confidence": "High",
#     "expected_outcome": "<measurable result>",
# })

insights_df = pd.DataFrame(auto_insights)
priority_map = {"🔴 Critical": 3, "🟡 Important": 2, "🟢 FYI": 1}
insights_df["_rank"] = insights_df["priority"].map(priority_map).fillna(0)
insights_df = insights_df.sort_values("_rank", ascending=False).drop(columns=["_rank"]).head(5)

tool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"insights": insights_df}}
tool_result
```

### Step 3 — insight priority chart → plotly_tool
```python
import plotly.graph_objects as go

colors_map = {"🔴 Critical": "#e74c3c", "🟡 Important": "#f39c12", "🟢 FYI": "#2ecc71"}
priority_score = {"🔴 Critical": 3, "🟡 Important": 2, "🟢 FYI": 1}

fig = go.Figure(go.Bar(
    y=insights_df["metric"],
    x=insights_df["priority"].map(priority_score),
    orientation="h",
    marker_color=[colors_map.get(p, "#95a5a6") for p in insights_df["priority"]],
    text=insights_df["so_what"].str[:60] + "...",
    textposition="inside",
))
fig.update_layout(
    title="Insight Priorities",
    xaxis=dict(tickvals=[1, 2, 3], ticktext=["FYI", "Important", "Critical"], title="Priority"),
    yaxis=dict(title="Metric / Column"),
    height=max(200, len(insights_df) * 60),
)
tool_result = chart.result(fig, artifact_name="insights_summary")
tool_result
```

### Rules
- Step 2 generates insights PROGRAMMATICALLY from real data — don't copy the template literally
- After the programmatic base (auto_insights), manually append insights from prior session steps (root_cause, ab_test, etc.)
- Maximum 5 insights — select the most important by impact × confidence
- If analysis yielded neutral results — explicitly write "No significant deviations found" (template already handles this case)
- Confidence: High — confirmed by data with p<0.05; Medium — correlation without causation; Low — hypothesis
- Each insight: **So What** (concrete numbers) + **Why** (hypothesis) + **Now What** (concrete action)
- Recommendations must be actionable with an owner and timeline, not "conduct further analysis"
- Priorities: 🔴 Critical (immediate action) → 🟡 Important (consider) → 🟢 FYI
