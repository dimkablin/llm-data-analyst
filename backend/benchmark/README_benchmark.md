# Бенчмарк LLM Data Analyst

`benchmark_chat.py` прогоняет набор вопросов через backend LLM Data Analyst и сохраняет отчет о качестве, времени, tool calls, ошибках инструментов, артефактах и оценке LLM-судьи.

Главная особенность: все вопросы по умолчанию идут в одном chat session. Это позволяет измерять деградацию качества на растущем контексте.

## Что получается на выходе

В папке `--out` создаются:

- `results.jsonl` - полный лог по каждому вопросу: ответ, raw SSE events, tool payloads, final payload, judge result.
- `summary.csv` - таблица для анализа в Excel/BI.
- `report.md` - короткая markdown-сводка.
- `chat.html` - человекочитаемый чат с Markdown-ответами, артефактами, tool calls, raw API и LLM summary в начале.
- `run_meta.json` - параметры запуска и финальное LLM summary.

## Формат вопросов

Можно использовать обычный TXT/MD:

```text
Сколько всего пассажиров в таблице?
Построй график по категориям.
```

Или YAML/JSON с ожиданиями для LLM-as-a-judge:

```yaml
cases:
  - id: total_passengers
    question: "Сколько пассажиров всего в датасете?"
    must_include_numbers: [891]
    expected_facts:
      - "Всего 891 пассажир."
    expected_insights:
      - "Нужно явно назвать общее количество строк/пассажиров."
```

Если в YAML есть ожидания, judge включается автоматически. Для TXT/MD можно включить generic judge через `--judge`.

## Базовый запуск

```powershell
python backend\benchmark\benchmark_chat.py `
  --base-url http://localhost:8605 `
  --user admin `
  --password tdapadmin `
  --csv backend\benchmark\data\transactions\bank_transactions_demo.csv `
  --questions backend\benchmark\data\transactions\questions.yaml `
  --out .runtime\benchmark_transactions
```

Ограничить число вопросов:

```powershell
python backend\benchmark\benchmark_chat.py `
  --base-url http://localhost:8605 `
  --user admin `
  --password tdapadmin `
  --csv backend\benchmark\data\transactions\bank_transactions_demo.csv `
  --questions backend\benchmark\data\transactions\questions.yaml `
  --out .runtime\benchmark_transactions_1 `
  --limit 1
```

## LLM-as-a-judge

Judge использует OpenAI-compatible `/chat/completions`.

По умолчанию настройки берутся из `.env`:

- `JUDGE_LLM_BASE_URL`, иначе `LLM_MODEL_API_URL`
- `JUDGE_LLM_MODEL`, иначе `LLM_MODEL_NAME`
- `JUDGE_LLM_API_KEY`, иначе `LLM_API_KEY`

Можно переопределить явно:

```powershell
python backend\benchmark\benchmark_chat.py `
  --base-url http://localhost:8605 `
  --user admin `
  --password tdapadmin `
  --csv backend\benchmark\data\transactions\bank_transactions_demo.csv `
  --questions backend\benchmark\data\transactions\questions.yaml `
  --out .runtime\benchmark_transactions `
  --judge-base-url http://localhost:8002/v1 `
  --judge-timeout 240
```

Judge строже всего штрафует за:

- выдуманные числа;
- выводы без подтверждения артефактами;
- упомянутые, но не созданные таблицы/графики;
- запрошенные, но отсутствующие графики/таблицы;
- пропущенные ожидаемые инсайты из YAML.

Ошибки tool calls уже штрафуются эвристикой, поэтому judge учитывает их как вторичный сигнал. Приблизительные коэффициенты и округленные сравнения не считаются большой ошибкой, если абсолютные значения и главный вывод корректны.

## Скоринг

Есть два уровня:

- `heuristic_score` - локальные правила: HTTP/stream errors, пустой ответ, fallback, tool errors, missing plot/artifact, слишком много tool calls, latency.
- `quality_score` - итоговый score. Если judge включен, это смесь `heuristic_score` и `judge_score`, с cap при `judge_pass=false`.

Штраф за время:

- `>30s`: `-5`
- `>40s`: `-10`
- `>60s`: `-18`
- `>90s`: `-25`
- `>120s`: `-35`

## Температура аналитика

В `QueryRequest` backend нет поля temperature. Температуры берутся из пользовательских настроек backend:

- `llm_temperature_chat`
- `llm_temperature_tool`

Раннер умеет временно менять их перед прогоном:

```powershell
python backend\benchmark\benchmark_chat.py `
  --base-url http://localhost:8605 `
  --user admin `
  --password tdapadmin `
  --csv backend\benchmark\data\transactions\bank_transactions_demo.csv `
  --questions backend\benchmark\data\transactions\questions.yaml `
  --out .runtime\benchmark_transactions_temp0 `
  --chat-temperature 0 `
  --tool-temperature 0
```

По умолчанию прежние значения восстанавливаются при завершении процесса. Если нужно оставить новые настройки в backend:

```powershell
--keep-temperature-settings
```

Judge temperature задается отдельно:

```powershell
--judge-temperature 0
```

## Как смотреть HTML

Открой `chat.html` из папки прогона.

В начале будут:

- общая метрика по прогону;
- `LLM run summary` с главными проблемами;
- затем каждый turn: вопрос, ответ с Markdown-форматированием, judge result, tool calls, артефакты и raw API.

Для расследования ошибок полезнее всего смотреть:

- `LLM run summary`;
- блок `llm judge`;
- `Tool output / error`;
- `Raw tool_start payload`;
- `Raw tool_end payload`;
- `All SSE events`.

## Запуск через pytest

Live benchmark можно запускать одной pytest-командой. По умолчанию он скипается, чтобы обычные тесты не ходили в backend и LLM.

```powershell
$env:LIVE_BENCHMARK_TESTS = "1"
python -m pytest -m benchmark
```

Доступные demo-наборы:

- `backend/benchmark/data/sportmaster`
- `backend/benchmark/data/investment`

Полезные переменные:

```powershell
$env:BENCHMARK_LIMIT = "3"
$env:BENCHMARK_MIN_AVG_SCORE = "70"
$env:BENCHMARK_MIN_CASE_SCORE = "50"
$env:BENCHMARK_MIN_JUDGE_SCORE = "60"
```
