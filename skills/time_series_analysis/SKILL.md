---
name: Анализ временных рядов
description: Декомпозиция трендов и сезонности, тест стационарности, скользящие средние, аномалии, паттерны во временных данных.
triggers: временной ряд, time series, тренд, сезонность, seasonality, динамика, по времени, rolling, скользящее среднее, период, temporal, временная зависимость
---

## Анализ временных рядов

Используй для анализа данных с временной компонентой: тренды, сезонность, стационарность, аномалии.

### Шаг 1 — подготовка и ресемплинг через pandas_tool
```python
# Авто-определение колонок
date_col = next(
    (c for c in df.columns if df[c].dtype == "datetime64[ns]" or "date" in c.lower()),
    df.columns[0]
)
metric_col = df.select_dtypes(include="number").columns[0]

df[date_col] = pd.to_datetime(df[date_col])
ts = df.set_index(date_col)[metric_col].sort_index()

# Авто-гранулярность по длине ряда
date_range_days = (ts.index.max() - ts.index.min()).days
freq = "D" if date_range_days <= 90 else ("W" if date_range_days <= 730 else "ME")

ts_resampled = ts.resample(freq).sum()

ts_df = ts_resampled.reset_index()
ts_df.columns = ["date", "value"]
ts_df["rolling_7"] = ts_resampled.rolling(7, min_periods=1).mean().values
ts_df["rolling_30"] = ts_resampled.rolling(30, min_periods=1).mean().values

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"time_series": ts_df}
}
tool_result
```

### Шаг 2 — тест стационарности через pandas_tool
```python
# Augmented Dickey-Fuller (statsmodels) — при недоступности: rolling variance check
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
            "✅ Стационарный ряд (тренд/сезонность невыражены)" if is_stationary
            else "⚠️ Нестационарный ряд — присутствует тренд или сезонность"
        ),
    }])
except ImportError:
    # Fallback: rolling variance check
    if len(ts_resampled) >= 10:
        mid = len(ts_resampled) // 2
        var_first = ts_resampled.iloc[:mid].var()
        var_second = ts_resampled.iloc[mid:].var()
        ratio = max(var_first, var_second) / min(var_first, var_second) if min(var_first, var_second) > 0 else 1
        is_stationary = ratio < 2.0
        adf_result_df = pd.DataFrame([{
            "test": "Rolling Variance (ADF недоступен)",
            "variance_ratio": round(ratio, 3),
            "is_stationary": is_stationary,
            "interpretation": (
                "✅ Дисперсия стабильна" if is_stationary
                else "⚠️ Дисперсия нестабильна — нестационарный ряд"
            ),
        }])

tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"stationarity": adf_result_df}
}
tool_result
```

### Шаг 3 — декомпозиция тренда и сезонности через pandas_tool
```python
ts_df_copy = ts_df.copy()
ts_df_copy["date"] = pd.to_datetime(ts_df_copy["date"])
ts_df_copy["month"] = ts_df_copy["date"].dt.month
ts_df_copy["year"] = ts_df_copy["date"].dt.year

# Тренд через линейную регрессию
trend_slope = np.polyfit(range(len(ts_df_copy)), ts_df_copy["value"].fillna(0), 1)[0]
trend_direction = "↑ растёт" if trend_slope > 0 else "↓ падает"

# Сезонность через месячные средние
monthly_avg = ts_df_copy.groupby("month")["value"].mean()
seasonal_strength = round(monthly_avg.std() / monthly_avg.mean() * 100, 1) if monthly_avg.mean() != 0 else 0

# Аномалии (Z-score > 3)
mean_val = ts_df_copy["value"].mean()
std_val = ts_df_copy["value"].std()
z_scores = ((ts_df_copy["value"] - mean_val) / std_val).abs() if std_val > 0 else pd.Series(0, index=ts_df_copy.index)
anomalies = ts_df_copy[z_scores > 3][["date", "value"]].copy()
anomalies["z_score"] = z_scores[z_scores > 3].round(2).values

summary = pd.DataFrame([{
    "metric": metric_col,
    "total_periods": len(ts_df_copy),
    "freq": freq,
    "trend": trend_direction,
    "trend_slope_per_period": round(trend_slope, 4),
    "seasonal_variation_pct": seasonal_strength,
    "is_seasonal": seasonal_strength > 20,
    "anomalies_count": len(anomalies),
    "stationarity": ("стационарный" if is_stationary else "нестационарный") if is_stationary is not None else "не проверялось",
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

### Шаг 4 — визуализация динамики через plotly_tool
```python
import plotly.graph_objects as go

# ts_df доступен из шага 1; anomalies из шага 3
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ts_df["date"], y=ts_df["value"],
    mode="lines", name="Факт",
    line=dict(color="#3498db", width=1), opacity=0.6,
))
fig.add_trace(go.Scatter(
    x=ts_df["date"], y=ts_df["rolling_7"],
    mode="lines", name="MA-7",
    line=dict(color="#e74c3c", width=2),
))
fig.add_trace(go.Scatter(
    x=ts_df["date"], y=ts_df["rolling_30"],
    mode="lines", name="MA-30",
    line=dict(color="#2ecc71", width=2, dash="dash"),
))

if len(anomalies) > 0:
    fig.add_trace(go.Scatter(
        x=anomalies["date"], y=anomalies["value"],
        mode="markers", name="Аномалии",
        marker=dict(color="#e74c3c", size=10, symbol="x"),
    ))

fig.update_layout(
    title=f"Динамика: {metric_col}  |  тренд: {trend_direction}  |  сезонность: {seasonal_strength:.1f}%",
    xaxis_title="Дата",
    yaxis_title=metric_col,
    hovermode="x unified",
    height=420,
)
tool_result = chart.result(fig, artifact_name="time_series_chart")
tool_result
```

### Правила
- Авто-гранулярность: ≤90 дней → дни, ≤2 года → недели, иначе → месяцы
- Нестационарный ряд (ADF `p > 0.05`) — тренд или сезонность присутствуют; для прогноза нужно дифференцирование
- Сезонная вариация > 20% — ряд сезонный, укажи явно; < 5% — сезонность незначима
- Аномалии по Z-score > 3 — экстремальные; Z > 2 — заслуживают внимания
- Если данных < 12 точек — тренд ненадёжен, предупреди пользователя
- Для прогноза используй `forecast_tool` — этот скил только для анализа исторических данных
