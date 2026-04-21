## Time Series Analysis

Use for data with a time component: trends, seasonality, stationarity, anomalies.

### Step 1 — preparation and moving averages → pandas_tool
```python
# Auto-detect date column and primary numeric metric
date_col = next(
    (c for c in df.columns if df[c].dtype == "datetime64[ns]" or "date" in c.lower() or "time" in c.lower()),
    df.columns[0]
)
metric_col = df.select_dtypes(include="number").columns[0]

df[date_col] = pd.to_datetime(df[date_col])
ts = df.set_index(date_col)[metric_col].sort_index()

# Auto-granularity: ≤90 days → daily, ≤2 years → weekly, else → monthly
date_range_days = (ts.index.max() - ts.index.min()).days
freq = "D" if date_range_days <= 90 else ("W" if date_range_days <= 730 else "ME")

ts_resampled = ts.resample(freq).sum()

if len(ts_resampled) < 12:
    # Too few points — skip ADF, decomposition, ACF
    ts_df = ts_resampled.reset_index()
    ts_df.columns = ["date", "value"]
    tool_result = {
        "schema_version": "1.0",
        "artifact_type": "table",
        "items": {"time_series": ts_df},
        "warning": f"Only {len(ts_resampled)} points — ADF, decomposition, and ACF skipped.",
    }
    tool_result
else:
    # Adaptive MA windows based on frequency
    if freq == "D":
        short_window, long_window = 7, 30
        short_label, long_label = "MA-7d", "MA-30d"
    elif freq == "W":
        short_window, long_window = 4, 12
        short_label, long_label = "MA-4w", "MA-12w"
    else:
        short_window, long_window = 3, 6
        short_label, long_label = "MA-3m", "MA-6m"

    ts_df = ts_resampled.reset_index()
    ts_df.columns = ["date", "value"]
    ts_df["rolling_short"] = ts_resampled.rolling(short_window, min_periods=1).mean().values
    ts_df["rolling_long"] = ts_resampled.rolling(long_window, min_periods=1).mean().values

    tool_result = {
        "schema_version": "1.0",
        "artifact_type": "table",
        "items": {"time_series": ts_df},
    }
    tool_result
```

### Step 2 — stationarity test (ADF) → pandas_tool
```python
is_stationary = None
adf_result_df = pd.DataFrame()

try:
    from statsmodels.tsa.stattools import adfuller
    series = ts_resampled.dropna()
    adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(series, autolag="AIC")
    is_stationary = adf_p <= 0.05
    adf_result_df = pd.DataFrame([{
        "test": "Augmented Dickey-Fuller",
        "adf_statistic": round(adf_stat, 4),
        "p_value": round(adf_p, 5),
        "is_stationary": is_stationary,
        "critical_1pct": round(adf_crit["1%"], 3),
        "critical_5pct": round(adf_crit["5%"], 3),
        "interpretation": (
            "Stationary series" if is_stationary
            else "Non-stationary — trend or seasonality present"
        ),
    }])
except ImportError:
    # Fallback: rolling variance ratio when statsmodels unavailable
    if len(ts_resampled) >= 10:
        mid = len(ts_resampled) // 2
        var_first = ts_resampled.iloc[:mid].var()
        var_second = ts_resampled.iloc[mid:].var()
        ratio = max(var_first, var_second) / min(var_first, var_second) if min(var_first, var_second) > 0 else 1
        is_stationary = ratio < 2.0
        adf_result_df = pd.DataFrame([{
            "test": "Rolling Variance (ADF unavailable)",
            "variance_ratio": round(ratio, 3),
            "is_stationary": is_stationary,
            "interpretation": "Variance stable" if is_stationary else "Variance unstable",
        }])

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"stationarity": adf_result_df}
}
tool_result
```

### Step 2.5 — autocorrelation (ACF) → pandas_tool
```python
max_lags = min(20, len(ts_resampled) // 2)
acf_values = [ts_resampled.autocorr(lag=i) for i in range(max_lags)]
acf_df = pd.DataFrame({
    "lag": range(len(acf_values)),
    "acf": [round(v, 4) for v in acf_values],
})

if len(acf_df) > 1:
    dominant_lag = int(acf_df.loc[1:, "acf"].abs().idxmax())
    dominant_acf = round(acf_df.loc[dominant_lag, "acf"], 4)
else:
    dominant_lag = None
    dominant_acf = None

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"acf": acf_df},
    "dominant_lag": dominant_lag,
    "dominant_acf": dominant_acf,
}
tool_result
```

### Step 3 — decomposition and anomalies → pandas_tool
```python
ts_df_copy = ts_df.copy()
ts_df_copy["date"] = pd.to_datetime(ts_df_copy["date"])
ts_df_copy["month"] = ts_df_copy["date"].dt.month
ts_df_copy["year"] = ts_df_copy["date"].dt.year

trend_slope = np.polyfit(range(len(ts_df_copy)), ts_df_copy["value"].fillna(0), 1)[0]
trend_direction = "rising" if trend_slope > 0 else "falling"

monthly_avg = ts_df_copy.groupby("month")["value"].mean()
seasonal_strength = round(monthly_avg.std() / monthly_avg.mean() * 100, 1) if monthly_avg.mean() != 0 else 0

# Anomaly threshold: Z>2.5 for short series, Z>3.0 for longer
z_threshold = 2.5 if len(ts_df_copy) < 30 else 3.0
mean_val = ts_df_copy["value"].mean()
std_val = ts_df_copy["value"].std()
z_scores = ((ts_df_copy["value"] - mean_val) / std_val).abs() if std_val > 0 else pd.Series(0, index=ts_df_copy.index)
anomalies = ts_df_copy[z_scores > z_threshold][["date", "value"]].copy()
anomalies["z_score"] = z_scores[z_scores > z_threshold].round(2).values

summary = pd.DataFrame([{
    "metric": metric_col,
    "total_periods": len(ts_df_copy),
    "freq": freq,
    "trend": trend_direction,
    "trend_slope_per_period": round(trend_slope, 4),
    "seasonal_variation_pct": seasonal_strength,
    "is_seasonal": seasonal_strength > 20,
    "anomalies_count": len(anomalies),
    "anomaly_z_threshold": z_threshold,
    "dominant_acf_lag": dominant_lag,
    "stationarity": ("stationary" if is_stationary else "non-stationary") if is_stationary is not None else "not checked",
    "mean_value": round(mean_val, 2),
    "max_value": round(ts_df_copy["value"].max(), 2),
    "min_value": round(ts_df_copy["value"].min(), 2),
}])

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"ts_summary": summary, "anomalies": anomalies}
}
tool_result
```

### Step 4 — visualization → plotly_tool
```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["value"], mode="lines", name="Actual",
    line=dict(color="#3498db", width=1), opacity=0.6))
fig.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["rolling_short"], mode="lines", name=short_label,
    line=dict(color="#e74c3c", width=2)))
fig.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["rolling_long"], mode="lines", name=long_label,
    line=dict(color="#2ecc71", width=2, dash="dash")))

if len(anomalies) > 0:
    fig.add_trace(go.Scatter(x=anomalies["date"], y=anomalies["value"],
        mode="markers", name=f"Anomalies (Z>{z_threshold})",
        marker=dict(color="#e74c3c", size=10, symbol="x")))

fig.update_layout(
    title=f"Dynamics: {metric_col}  |  trend: {trend_direction}  |  seasonality: {seasonal_strength:.1f}%",
    xaxis_title="Date", yaxis_title=metric_col, hovermode="x unified", height=420,
)
tool_result = chart.result(fig, artifact_name="time_series_chart")
tool_result
```

### Rules
- Auto-granularity: ≤90 days → `D`, ≤2 years → `W`, else → `ME`
- MA windows adapt to frequency — don't hardcode `rolling_7`/`rolling_30`
- Anomaly threshold: Z>2.5 if points < 30, Z>3.0 if ≥ 30
- < 12 points after resampling — skip ADF, ACF, decomposition, return only chart with warning
- Not for forecasting — use `anomaly_planfact_tool` for plan-vs-actual
- Non-stationary (ADF p > 0.05) — trend/seasonality present
- Seasonal variation > 20% — series is seasonal; < 5% — not significant
