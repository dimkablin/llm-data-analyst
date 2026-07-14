---
name: Time Series Analysis
description: Trend and seasonality decomposition, stationarity test, moving averages, anomaly detection, temporal patterns.
enabled_by_default: false
triggers: time series, trend, seasonality, rolling, moving average, period, temporal, temporal dependency, временной ряд, тренд, сезонность, скользящее среднее, аномал, отклонени
---

## Time Series Analysis

**НЕ используй этот skill для прогноза будущих периодов.** Если пользователь просит прогноз / forecast / «на N месяцев вперёд» → сразу `get_tool_instructions("forecast_tool")` и `forecast_tool` с `horizon=N`. Не вызывай `statsmodels` / ручной forecast в `pandas_tool`.

Data with a time component: trends, seasonality, stationarity, anomalies.

### Algorithm (4 steps)
1. **Preparation + moving averages** → `pandas_tool`: auto-detect date column, resample by auto-granularity, compute short/long MA.
2. **Stationarity (ADF)** → `pandas_tool`: Augmented Dickey-Fuller; rolling-variance fallback if statsmodels unavailable.
3. **Autocorrelation (ACF)** → `pandas_tool`: up to 20 lags, identify dominant lag.
4. **Decomposition + anomalies + visualization** → `pandas_tool` + `plotly_tool`: trend slope, seasonal variation %, Z-score anomalies, line chart.

### Rules
- Auto-granularity: ≤90 days → `D`, ≤2 years → `W`, else → `ME`
- MA windows adapt to frequency — don't hardcode values
- Anomaly threshold: Z > 2.5 if points < 30, else Z > 3.0
- < 12 points after resampling → skip ADF/ACF/decomposition, return chart with warning
- Not for forecasting — use `forecast_tool` / `anomaly_planfact_tool` instead
- Non-stationary (ADF p > 0.05) → trend/seasonality present; seasonal variation > 20% → series is seasonal
