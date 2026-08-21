---
name: Portfolio Risk Analysis
description: Model-directed portfolio risk, concentration, and contribution analysis using discovered portfolio columns and tool-produced evidence.
enabled_by_default: true
triggers: портфельный риск, риск портфеля, концентрация портфеля, перегруз портфеля, вклад позиций, portfolio risk, risk concentration, concentration risk, portfolio contribution, portfolio positions
---

## Portfolio Risk Analysis

Use this skill for portfolio concentration, risk segmentation, position
contribution, and defensive-vs-risky position analysis. The model must perform
the calculations through tools and artifacts; this skill is instruction and
contract, not executable business logic.

### Algorithm
1. **Schema preview** -> use `sql_tool` to inspect the portfolio table or
   session artifact and identify position id, instrument, segment, risk class,
   value, share, and PnL-like columns from real schema names.
2. **Base portfolio table** -> use `sql_tool` to create
   `portfolio_positions_table` with the columns needed for the requested
   question. Do not assume fixed column names; alias discovered columns to clear
   semantic names in the result artifact.
3. **Concentration metrics** -> use `pandas_tool` to calculate segment shares,
   top positions, contribution to absolute portfolio value, and overload
   thresholds requested by the user or relevant to the data.
4. **Risk segmentation** -> use `pandas_tool` to create
   `risk_concentration_table` with segment, total value, share percent,
   position count, and notable contributors.
5. **Visualization** -> use `plotly_tool` to create
   `risk_concentration_chart` for the strongest risk or concentration dimension.
6. **Final synthesis** -> explain where concentration exists, which positions
   or segments drive it, what the chart shows, and what data limits remain.

### Required capabilities
- `read_only_sql`
- `dataframe_transform`
- `chart`

### Required artifacts
- table: portfolio_positions_table
- table: risk_concentration_table
- plot: risk_concentration_chart

### Evidence rules
- Start from a schema preview before naming portfolio, risk, value, share, or PnL columns.
- Never overwrite a source percent/rate column when deriving monetary impact; create a new derived column.
- When reporting concentration, state whether the share is position-level, instrument-level, or segment-level.
- Do not replace concentration analysis with a simple top-by-value ranking unless the user asked only for top positions.

### Rules
- Use real column names from schema inspection; fixed example or customer schemas
  are not part of the runtime contract.
- If no risk-class column exists, analyze concentration by the closest
  available segment and explicitly state the missing risk dimension.
- If no value or weight column exists, ask for the needed denominator instead of
  fabricating concentration percentages.
- The final answer must reference `risk_concentration_table` and
  `risk_concentration_chart` when the requested analysis includes risk or
  concentration.
