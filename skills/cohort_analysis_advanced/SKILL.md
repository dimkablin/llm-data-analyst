---
name: Когортный анализ (расширенный)
description: Расширенный когортный анализ — retention, LTV, revenue cohorts, сравнение когорт и визуализация heatmap.
enabled_by_default: false
triggers: когорт ltv, revenue cohort, ltv когорт, выручка по когортам, сравнение когорт, продвинутый когортный, cohort revenue, lifetime value
---

## Когортный анализ (расширенный)

Для детального когортного анализа с LTV, revenue-когортами и сравнением когорт.
Для базового retention используй `cohort_analysis`.

### Алгоритм (4 шага)
1. **Retention + LTV матрица** → `pandas_tool`: когорта = месяц первого события, `period_number`, `retention_pct`. Если есть `revenue` → строит `ltv_cumulative` и `ltv_per_user`.
2. **Retention heatmap** → `plotly_tool`: `px.imshow`, цветовая шкала Blues.
3. **LTV heatmap** → `plotly_tool`: `px.imshow`, цветовая шкала Greens. Если revenue нет → placeholder.
4. **Сравнение когорт** → `plotly_tool`: grouped bar по периодам 0, 1, 3, 6, 12.

### Правила
- Гранулярность по умолчанию — месяц (`M`); для молодых продуктов — неделя (`W`)
- Когорт > 24 → показывать только последние 12 для читаемости
- LTV монотонно растёт (кумулятивная сумма) — если убывает, ошибка в данных
- Нормальный retention P1: e-commerce 20–30%, SaaS 40–60%
