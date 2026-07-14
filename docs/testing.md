# Testing Baseline

## Canonical CI Commands

Offline baseline:

```bash
poetry run pytest -m "not live and not e2e" -q
```

Config validation:

```bash
poetry check
```

Lint touched or new Python code:

```bash
poetry run ruff check <paths>
```

Use the offline baseline before runner/runtime refactors. It must stay green so new
regressions are distinguishable from old noise.

## Test Scope

- Offline tests must not require an LLM, network, external DB, browser, or running
  backend/frontend process.
- Tests that require external services must be marked `live`, `e2e`, or both.
- Prefer public behavioral contracts: API responses, Pydantic schemas, service
  outputs, artifact payloads, tool input/output contracts, and runtime events.
- Avoid tests that pin private helper names or temporary RED-test scaffolding unless
  the helper is an intentionally stable internal contract.
