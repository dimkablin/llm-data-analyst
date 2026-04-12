# 📊 Топ-10 open-source Claude Code скилов для анализа данных

Подборка бесплатных скилов в формате `SKILL.md`, которые можно добавить в папку `skills/` проекта.

---

### 1. `csv-data-summarizer`
🔗 **Репо:** [coffeefuelbump/csv-data-summarizer-claude-skill](https://github.com/coffeefuelbump/csv-data-summarizer-claude-skill)

Автоматически анализирует загруженный CSV: статистика, пропуски, базовые визуализации через pandas.

| | |
|--|--|
| ✅ Plug-and-play, минимальная настройка | ⚠️ Только CSV, не работает с БД |
| ✅ Распознаёт тип данных (sales, time series, financial) | ⚠️ Нет глубокого прогноза |
| ✅ Генерирует code-ready pandas pipeline | ⚠️ Визуализации базовые (matplotlib) |

💡 **Полезность для проекта:** Прямое расширение `pandas_tool` — можно добавить как шаблон инструкций для автоматического EDA при загрузке нового файла.

---

### 2. `programmatic-eda`
🔗 **Репо:** [nimrodfisher/data-analytics-skills](https://github.com/nimrodfisher/data-analytics-skills)

Системный EDA с автоматическими sanity-чеками, профилингом и корреляционным анализом.

| | |
|--|--|
| ✅ Структурированный подход, воспроизводимые отчёты | ⚠️ Требует чёткого schema hint |
| ✅ Выявляет аномалии типов и выбросы | ⚠️ Медленно на больших датасетах |
| ✅ Генерирует markdown-резюме | ⚠️ Нет поддержки вложенных JSON-полей |

💡 **Полезность для проекта:** Идеальный кандидат для скила `auto_eda` в папке `skills/` — агент сможет запускать его первым шагом при любом новом датасете.

---

### 3. `cohort-analysis`
🔗 **Репо:** [nimrodfisher/data-analytics-skills](https://github.com/nimrodfisher/data-analytics-skills)

Когортный анализ с retention-матрицей и визуализацией heatmap.

| | |
|--|--|
| ✅ Готовые шаблоны для retention и LTV | ⚠️ Предполагает наличие `user_id` + `date` |
| ✅ Чёткий контракт входных данных | ⚠️ Требует pandas + matplotlib в sandbox |
| ✅ Объясняет бизнес-интерпретацию результата | ⚠️ Нет поддержки SQL напрямую |

💡 **Полезность для проекта:** Проект уже имеет `cohort_analysis` в папке `skills/` — этот скил можно использовать для обновления инструкций через сравнение с сообществом.

---

### 4. `time-series-analysis`
🔗 **Репо:** [nimrodfisher/data-analytics-skills](https://github.com/nimrodfisher/data-analytics-skills)

Детекция трендов, сезонности, ARIMA-прогноз, STL-декомпозиция.

| | |
|--|--|
| ✅ Охватывает decompose + forecast + anomaly | ⚠️ Зависит от `statsmodels` / `prophet` |
| ✅ Хорошо документированные примеры | ⚠️ Не адаптирован под plan-fact формат |
| ✅ Работает с любым time-indexed DataFrame | |

💡 **Полезность для проекта:** Напрямую дополняет `forecast_tool` — можно включить как расширенную инструкцию для агента при временны́х рядах.

---

### 5. `ab-test-analysis`
🔗 **Репо:** [nimrodfisher/data-analytics-skills](https://github.com/nimrodfisher/data-analytics-skills)

Статистически строгий A/B анализ: t-test, Mann-Whitney, power analysis, multiple testing correction.

| | |
|--|--|
| ✅ Автоматически выбирает тест по типу данных | ⚠️ Нужны чёткие колонки control/variant |
| ✅ Объясняет statistical significance в понятном формате | ⚠️ Не умеет multi-arm тесты |
| ✅ Генерирует интерпретацию для нетехнических стейкхолдеров | |

💡 **Полезность для проекта:** Нет аналога в текущем проекте — готовый скил для добавления в `skills/ab_test/SKILL.md`.

---

### 6. `root-cause-investigation`
🔗 **Репо:** [nimrodfisher/data-analytics-skills](https://github.com/nimrodfisher/data-analytics-skills)

Систематическое расследование падения/роста метрики: drill-down по измерениям, waterfall, contribution analysis.

| | |
|--|--|
| ✅ Структурированный фреймворк для дебага метрик | ⚠️ Требует нескольких измерений в данных |
| ✅ Генерирует гипотезы и их проверку | ⚠️ Медленный (многошаговый pipeline) |
| ✅ Хорошо ложится на `planner_tool` | |

💡 **Полезность для проекта:** Идеально работает в связке с `planner_tool` — планировщик видит этот скил и включает его в план при запросах типа «почему упали продажи».

---

### 7. `duckdb-query`
🔗 **Репо:** [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills)

Запросы к CSV/Parquet напрямую через DuckDB SQL без загрузки в pandas.

| | |
|--|--|
| ✅ Работает с большими файлами (>1GB) без нагрузки на память | ⚠️ Нужен duckdb в sandbox |
| ✅ Поддерживает JOIN через несколько файлов | ⚠️ Дублирует часть `sql_tool` для CSV |
| ✅ Отлично для schema exploration | |

💡 **Полезность для проекта:** Закрывает gap для тяжёлых CSV — текущий `sql_tool` ориентирован на PG/ClickHouse, а этот скил добавляет SQL-интерфейс к любому файлу.

---

### 8. `statistical-analysis`
🔗 **Репо:** [mingrath/awesome-claude-skills](https://github.com/mingrath/awesome-claude-skills)

Гипотезы, регрессия, ANOVA, корреляции — полный стат пакет с интерпретацией.

| | |
|--|--|
| ✅ Широкий охват методов | ⚠️ Требует `scipy` + `statsmodels` |
| ✅ Объясняет результаты человекочитаемо | ⚠️ Нет байесовских методов |
| ✅ Генерирует publication-quality таблицы | |

💡 **Полезность для проекта:** Дополнение к `value_tool` для случаев когда нужна не просто метрика, а статистически подтверждённый вывод.

---

### 9. `data-quality-audit`
🔗 **Репо:** [nimrodfisher/data-analytics-skills](https://github.com/nimrodfisher/data-analytics-skills)

Аудит качества данных: дубли, nulls, outliers, type mismatches, referential integrity.

| | |
|--|--|
| ✅ Генерирует DQ-отчёт с severity уровнями | ⚠️ Только диагностика, не исправляет |
| ✅ Хорошо документирован | ⚠️ Медленный на широких таблицах |
| ✅ Указывает конкретные строки с проблемами | |

💡 **Полезность для проекта:** Можно добавить как автоматический шаг при bind нового датасета (`skills/data_quality/SKILL.md`) — агент будет предупреждать о проблемах с данными до начала анализа.

---

### 10. `insight-synthesis`
🔗 **Репо:** [nimrodfisher/data-analytics-skills](https://github.com/nimrodfisher/data-analytics-skills)

Трансформирует сырые результаты анализа в структурированные бизнес-инсайты с рекомендациями.

| | |
|--|--|
| ✅ Отличный финальный шаг для любого pipeline | ⚠️ Зависит от качества предыдущих шагов |
| ✅ Генерирует executive summary в markdown | ⚠️ Иногда слишком лаконичен |
| ✅ Хорошо работает в связке с `review_tool` | |

💡 **Полезность для проекта:** Напрямую усиливает `_finalize_node` в `runner.py` — агент получает инструкции как правильно упаковать аналитику в финальный ответ.

---

## 🗂️ Приоритет добавления в проект

| Скил | Приоритет | Действие |
|------|-----------|---------|
| `programmatic-eda` | 🔴 Высокий | Добавить как `skills/auto_eda/SKILL.md` |
| `duckdb-query` | 🔴 Высокий | Закрывает gap с большими CSV |
| `data-quality-audit` | 🔴 Высокий | Автозапуск при загрузке датасета |
| `ab-test-analysis` | 🟡 Средний | Новый аналитический сценарий |
| `root-cause-investigation` | 🟡 Средний | Усиливает `planner_tool` |
| `insight-synthesis` | 🟡 Средний | Усиливает `_finalize_node` |
| `csv-data-summarizer` | 🟢 Низкий | Перекрывается с `pandas_tool` |
| `cohort-analysis` | 🟢 Низкий | Уже есть в `skills/` |
| `time-series-analysis` | 🟢 Низкий | Покрыт `forecast_tool` |
| `statistical-analysis` | 🟢 Низкий | Расширение по необходимости |

---

## 🔗 Источники

- [coffeefuelbump/csv-data-summarizer-claude-skill](https://github.com/coffeefuelbump/csv-data-summarizer-claude-skill)
- [nimrodfisher/data-analytics-skills](https://github.com/nimrodfisher/data-analytics-skills)
- [mingrath/awesome-claude-skills](https://github.com/mingrath/awesome-claude-skills)
- [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills)
- [K-Dense-AI/claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills)
- [hesreallyhim/awesome-claude-code](https://github.com/hesreallyhim/awesome-claude-code)
- [BehiSecc/awesome-claude-skills](https://github.com/BehiSecc/awesome-claude-skills)
