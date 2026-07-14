---
name: Promo Effectiveness
description: Promotion effectiveness analysis using discovered baseline and promotion labels, with sales, volume, conversion, and uplift comparison by dimensions.
enabled_by_default: true
triggers: промо, promo, акци, скидк, baseline, эффективность промо, промоакци, сравни период, uplift, promotion effectiveness
---

## Promo Effectiveness

Use this skill when a dataset has a promotion/campaign field and the user asks
to compare baseline periods against promoted periods. Discover the baseline
label from data values or user wording before grouping rows.

### Algorithm
1. **Promotion map** -> `sql_tool`: list distinct promotion/campaign values,
   row shares, and candidate baseline labels.
2. **Baseline selection** -> use the user's wording when it matches data;
   otherwise infer a baseline only from values that clearly mean no campaign.
3. **Aggregate by flag** -> `sql_tool` or `pandas_tool`: compare baseline vs
   promotion groups for the requested period and dimensions.
4. **Visualization** -> `plotly_tool`: grouped bar or line chart for uplift by
   category, channel, segment, or another discovered dimension.
5. **Interpretation** -> calculate relative uplift against baseline and warn
   about seasonality or mix shifts when the comparison periods differ.

### Required tools
- `sql_tool`
- `pandas_tool`
- `plotly_tool`

### Required artifacts
- table: promo_map_table
- table: promo_effectiveness_table
- plot: promo_effectiveness_chart

### Evidence rules
- Сначала проверь DISTINCT promotion/campaign column; baseline label must come from data or user wording.
- Для сравнения периода явно фильтруй запрошенный диапазон, а не все доступные годы.
- Uplift считай как относительное изменение к baseline и показывай denominator.
- Не сравнивай conversion-like metric, если такой роли нет в источнике.

### Rules
- Не смешивай разные промо в одну кучу без пометки, если пользователь просит детализацию.
- Конверсию сравнивай только если колонка есть в данных.
- Цифры только из tool output; при слабом uplift (<5%) скажи об этом явно.
