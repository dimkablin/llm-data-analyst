---
name: DuckDB анализ больших файлов
description: SQL-анализ больших CSV и Parquet файлов через DuckDB — без загрузки в память, с поддержкой JOIN по нескольким файлам.
triggers: duckdb, большой файл, large csv, parquet, sql по файлу, несколько файлов, join файлов, out of memory, большой датасет, gb файл
---

## DuckDB анализ больших файлов

Используй, когда pandas не справляется с размером файла или нужно выполнить SQL-запросы по CSV/Parquet без загрузки в память.

В этом проекте `sql_tool` использует DuckDB под капотом — он умеет читать CSV и Parquet напрямую.

### Исследование схемы файла через sql_tool

```sql
-- Посмотреть структуру и первые строки
DESCRIBE SELECT * FROM read_csv_auto('/path/to/file.csv');

SELECT * FROM read_csv_auto('/path/to/file.csv') LIMIT 10;
```

### Агрегация без загрузки в память через sql_tool
```sql
-- Группировка и агрегация напрямую из файла
SELECT
    category,
    COUNT(*) AS count,
    SUM(revenue) AS total_revenue,
    AVG(revenue) AS avg_revenue,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY revenue) AS median_revenue
FROM read_csv_auto('/path/to/file.csv')
GROUP BY category
ORDER BY total_revenue DESC
LIMIT 20;
```

### JOIN нескольких файлов через sql_tool
```sql
-- Объединение двух CSV по ключу
SELECT
    a.user_id,
    a.event_date,
    b.segment,
    b.region,
    a.revenue
FROM read_csv_auto('/path/to/events.csv') a
LEFT JOIN read_csv_auto('/path/to/users.csv') b
    ON a.user_id = b.user_id
WHERE a.event_date >= '2024-01-01'
LIMIT 1000;
```

### Временные таблицы для сложных запросов через sql_tool
```sql
-- Создание временной таблицы для повторного использования
CREATE OR REPLACE TABLE tmp_filtered AS
SELECT * FROM read_csv_auto('/path/to/file.csv')
WHERE status = 'active' AND amount > 0;

SELECT
    DATE_TRUNC('month', CAST(created_at AS DATE)) AS month,
    COUNT(*) AS events,
    SUM(amount) AS total
FROM tmp_filtered
GROUP BY 1
ORDER BY 1;
```

### Анализ Parquet файлов через sql_tool
```sql
-- Parquet читается так же, только read_parquet вместо read_csv_auto
SELECT
    year,
    region,
    SUM(sales) AS total_sales
FROM read_parquet('/path/to/data.parquet')
GROUP BY year, region
ORDER BY year, total_sales DESC;
```

### Правила
- Используй `read_csv_auto` — он автоматически определяет типы, разделитель и кодировку
- Для файлов > 1GB всегда добавляй `WHERE` условия и `LIMIT` при первичном исследовании
- `PERCENTILE_CONT` и `PERCENTILE_DISC` — только в DuckDB, не в стандартном SQL
- После получения агрегата через sql_tool — результат доступен как `df` в pandas_tool для дальнейшей обработки
- Если путь к файлу неизвестен — попроси пользователя уточнить или используй `df` который уже загружен в сессию
