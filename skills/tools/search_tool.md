---
name: Search Tool
description: Веб-поиск актуальной информации с возможностью чтения страниц.
kind: tool
tool_key: search_tool
triggers: поиск, найди, новости, search, web, интернет, погода, курс, цена
---

## search_tool — веб-поиск

Вход: Python-код; доступен helper `search`.

### Методы
- `search.search("query")` — dict с keys: query, answer, results, sources
- `search.search_result("query", artifact_name="...")` — готовый табличный артефакт
- `search.fetch(urls)` — полный текст страниц (для точных/свежих данных)

### Когда использовать fetch
- Погода, курсы валют, цены, новости, расписание — любые **актуальные** данные.
- Если сниппеты из search не дают точного ответа — обязательно fetch.
- Паттерн: search → выбрать лучшие URL → fetch → ответить по тексту страниц.

### Контракт
```python
tool_result = search.search_result("запрос", artifact_name="results")
tool_result
```

### Правила
- Последняя строка: `tool_result`.
- Не делай HTTP-запросы вручную — используй helper `search`.
- Для точных фактов: search → pick URLs → fetch → ответ по содержимому.
