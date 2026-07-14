---
name: Investment Market Analysis
description: Model-directed investment market analysis that combines instrument snapshot, price history, news impact, and charts through explicit tool evidence.
enabled_by_default: true
triggers: акция, акции, облигация, облигации, тикер, эмитент, сектор, цена акции, почему выросла, почему упала, investment, market analysis, ticker, issuer, stock, bond, sector comparison, price history, market movement
---

## Investment Market Analysis

Use this skill for security, issuer, ticker, sector, market movement, and
price/news impact questions. The model must discover available tables and
compute evidence through tools; this skill must not encode one customer schema,
one ticker, or deterministic summary.

### Algorithm
1. **Source discovery** -> use `database_tool` to list tables and schemas when
   market data locations are unknown. Respect the active connection scope.
2. **Instrument lookup** -> use `sql_tool` to identify the requested security,
   issuer, ticker, sector, or asset class from available snapshot/reference
   tables. Save the result as `market_snapshot_table`.
3. **Price evidence** -> use `sql_tool` to fetch chronological price data for
   the selected instrument or peer group. Save the result as
   `market_price_history_table`.
4. **News/event evidence** -> when the user asks why price changed, or when
   market movement needs explanation, use `sql_tool` to fetch event/news rows.
   Save the result as `market_news_impact_table`.
5. **Computation** -> use `pandas_tool` to calculate return, volatility,
   drawdown, event grouping, sector comparison, or peer ranking from the
   tool-produced tables.
6. **Visualization** -> use `plotly_tool` to create
   `market_price_history_chart` from price history, not from a one-row snapshot.
7. **Final synthesis** -> separate facts from interpretation: snapshot status,
   price dynamics, news/event evidence, comparison context, and caveats.

### Required tools
- `database_tool`
- `sql_tool`
- `pandas_tool`
- `plotly_tool`

### Required artifacts
- table: market_snapshot_table
- table: market_price_history_table
- table: market_news_impact_table
- plot: market_price_history_chart

### Evidence rules
- Use schema preview or table discovery before assuming where snapshot, price, or news evidence lives.
- A company name is not necessarily a ticker; perform lookup before filtering time series or news.
- Price charts must be built from chronological price history, not from a one-row instrument snapshot.
- Movement explanations require both price history and event/news evidence when such data is available.

### Rules
- Do not hardcode ticker symbols, demo schema names, table names, or customer
  reports in the runtime.
- If a required evidence table is unavailable, state exactly which evidence is
  missing and continue only with the supported part of the analysis.
- For sector or peer comparisons, report the selected peer universe and the
  filter used to build it.
- The final answer must cite the tool-produced artifacts used for price and
  event conclusions.
