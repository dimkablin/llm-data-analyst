---
name: Retail Sales Analysis
description: Schema-first retail sales analysis for trends, channels, categories, brands, plan-fact, funnel metrics, and period comparisons.
enabled_by_default: true
triggers: розничн, retail, выручк, объём продаж, канал, категори, бренд, план-факт, динамик продаж, по месяцам, sales trend, retail channels, category ranking
---

## Retail Sales Analysis

Use this skill for retail datasets with time periods, commercial metrics,
dimensions, plan/fact columns, and funnel signals. Discover actual schema and
values first; do not rely on customer, file, brand, category, or period names.

### Algorithm
1. **Schema preview** -> use `sql_tool` to identify time, metric, plan, funnel,
   and dimension columns from the actual table.
2. **Slice** -> use `sql_tool` to aggregate by the requested period and
   dimensions. Keep the result compact and name it `retail_sales_slice_table`.
3. **Visualization** -> use `plotly_tool` to create
   `retail_sales_trend_chart` for trend, comparison, or ranking requests.
4. **Synthesis** -> report numbers from artifacts and list the schema roles used.

### Rules
- A source or file name is not a product, brand, category, or filter value.
- Use only columns discovered in the active schema.
- For period comparisons, verify the available date range before filtering.
- Do not substitute columns from a different dataset or language convention.

### Required capabilities
- read_only_sql
- chart

### Required artifacts
- table: retail_sales_slice_table
- plot: retail_sales_trend_chart

### Evidence rules
- Start with a schema and period preview before filtering channels, categories, brands, or months.
- The final answer must reference `retail_sales_slice_table` and `retail_sales_trend_chart` for trend or comparison requests.
