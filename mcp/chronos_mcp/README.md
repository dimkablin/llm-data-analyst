# Chronos MCP Server

Typed MCP tools поверх `chronos-forecasting` без LLM, свободного SQL и передачи
DB credentials в tool arguments.

## GitLab dev deploy

Push в ветку `dev` запускает один atomic build+deploy job на runner с тегом
`deploy-shell`. Runtime хранится в `/opt/deploy/chronos-mcp-dev`.

При первом deploy ключ генерируется на dev-хосте. Опциональная masked GitLab CI
variable `CHRONOS_MCP_API_KEY` позволяет задать его явно. Следующие deploy
используют сохранённый secret-файл. Backend из основного dev Compose подключается
по `http://chronos-mcp:8810/mcp` через сеть `llm-data-analyst-dev_default`.

## Что реализовано

- `forecast` — inline/table source, batch series, multi-target, quantiles,
  explicit frequency/missing policy, past/future covariates.
- `backtest` — rolling windows и метрики `MAE`, `RMSE`, `sMAPE`, `MASE`, `WQL`.
- `capabilities` — model family, revision, limits и доступность sources.
- Stable `structuredContent`, `outputSchema` и typed domain errors с
  `isError=true`.
- Lazy loading официального `BaseChronosPipeline`; unit tests не скачивают
  модель.
- Table source через явный read-only Data Gateway contract.

## Архитектура

```text
server.py       MCP transport + composition root
    ↓
application.py  forecast/backtest use cases
    ↓
preparation.py  deterministic series validation and preparation
    ↑
adapters.py     Chronos SDK and HTTP Data Gateway adapters
```

`application.py` и `preparation.py` не импортируют MCP или Chronos SDK.

## Локальный запуск

Python 3.11:

```powershell
cd mcp/chronos_mcp
python -m pip install -e ".[chronos,dev]"
Copy-Item .env.example .env
$env:CHRONOS_MODEL_REVISION="<pinned Hugging Face revision>"
$env:CHRONOS_MCP_API_KEY="<unique secret>"
chronos-mcp
```

Docker использует PyTorch `2.7.1` с CUDA `12.8`/cuDNN `9` и запрашивает GPU
через Compose. Для другой CUDA-сборки задайте `CHRONOS_PYTORCH_IMAGE`.

По умолчанию Streamable HTTP доступен на
`http://127.0.0.1:8810/mcp`. Передавайте ключ в заголовке
`Authorization: Bearer <CHRONOS_MCP_API_KEY>`; HTTP startup без ключа запрещён.

Для stdio:

```powershell
$env:CHRONOS_MCP_TRANSPORT="stdio"
chronos-mcp
```

## Проверка без модели

Тесты используют fake runtime и не требуют GPU/Hugging Face:

```powershell
..\..\.venv\Scripts\python.exe -m pytest -q
..\..\.venv\Scripts\ruff.exe check src tests
```

## Forecast input

```json
{
  "source": {
    "kind": "inline",
    "rows": [
      {"ts": "2026-01-01", "sales": 10},
      {"ts": "2026-02-01", "sales": 12}
    ]
  },
  "time_column": "ts",
  "targets": [
    {"name": "sales", "column": "sales", "aggregation": "none"}
  ],
  "horizon": 3,
  "frequency": "month_start",
  "missing_policy": "error",
  "quantiles": [0.1, 0.5, 0.9]
}
```

Aliases `M/MS/ME`, raw SQL, Python, URLs и credentials в input schema
отсутствуют.

## Table source

Table source включается только при наличии `CHRONOS_DATA_GATEWAY_URL`.
MCP отправляет:

```json
{
  "connection_id": "demo",
  "schema": "analytics",
  "table": "sales",
  "columns": ["ts", "amount"],
  "filter": {"column": "region", "op": "eq", "value": "Moscow"},
  "history_start": null,
  "history_end": null,
  "horizon": 3,
  "frequency": "month_start",
  "future_columns": [],
  "max_rows": 50000
}
```

на `POST {CHRONOS_DATA_GATEWAY_URL}/v1/chronos/rows`.

Gateway обязан:

- авторизовать `connection_id/schema/table`;
- проверить filter types по реальной schema;
- скомпилировать один parameterized read-only `SELECT`;
- вернуть `{"rows": [...], "query": {...}}`;
- не возвращать credentials.

В текущем backend такого endpoint пока нет. Без URL table source возвращает
`SOURCE_UNAVAILABLE`, а inline forecast продолжает работать.

## Production

- Укажите pinned `CHRONOS_MODEL_REVISION`; `main` допустим только для dev.
- Публикуйте MCP endpoint через существующий mTLS/OAuth service gateway.
- Не выставляйте порт напрямую в недоверенную сеть.
- `CHRONOS_DATA_GATEWAY_TOKEN` храните в secret store.
- GPU/CUDA image выбирается окружением; agent не управляет device/dtype.

Сборка:

```powershell
docker build -t chronos-mcp mcp/chronos_mcp
docker run --rm -p 8810:8810 --env-file mcp/chronos_mcp/.env chronos-mcp
```

Полное ТЗ: [../../docs/chronos_mcp_server_spec.md](../../docs/chronos_mcp_server_spec.md).
