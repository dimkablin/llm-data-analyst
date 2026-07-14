# Domain Extension Runtime Boundary

This project treats the LangGraph/ReAct runner as a generic execution engine.
Domain behavior must be attached through explicit extension contracts, not by
adding one-off branches to `backend/agent/runner.py`.

## Responsibilities

### LangGraph runtime

Owns only generic agent execution:

- state, graph nodes, routing, LLM calls, and the tool loop;
- MCP tool discovery and tool execution orchestration;
- permission checks, runtime events, artifact collection, and final response
  orchestration;
- generic validation of typed contracts.

The runtime must not contain customer-specific scenarios, fixed tickers, demo
schemas, report templates, deterministic summaries for one vertical, or
fallback/recovery branches for one dataset.

### Backend

Owns product infrastructure:

- FastAPI endpoints, auth, sessions, settings, persistence, storage, and
  observability;
- Pydantic API schemas and service contracts;
- source inventory, artifact persistence, and tool permissions.

Backend services may contain product business logic only when it is reusable
across domains. Vertical expertise belongs in domain extensions.

### Frontend

Owns customer-specific UX:

- screens, workflows, settings, and presentation of artifacts;
- typed frontend contracts for API requests and responses;
- workflow-specific UI decisions.

Frontend code must not depend on private runner methods or internal runtime
fallbacks.

### Domain extensions

New investment, sales, risk, forecasting, or customer-specific capability must
follow this path:

```text
MCP server + typed tools + Pydantic schemas + permissions + markdown skills
    -> generic LangGraph runtime
    -> backend API/service contracts
    -> frontend workflow/artifact presentation
```

## Skill Execution Contracts

Analytical skills can declare an execution contract directly in `SKILL.md`:

- `### Required tools`
- `### Required artifacts`
- `### Evidence rules`

`backend.skills.contracts` parses those sections into Pydantic models. Runtime
contract validation must stay domain-neutral: it checks that declared tools and
artifact types/names exist, but it does not know investment, sales, or risk
business rules.

During dispatch, the runner resolves selected and trigger-matched analytical
skills into `SkillExecutionRequirement` records and checks that their declared
tools are allowed by the current permission policy. During finalization, the
runner validates the produced tool calls and artifacts against those same
requirements. This is a generic contract gate, not a domain fallback.

## Domain Extension Manifests

`backend/domain_extensions` contains thin Pydantic manifests for domain MCP
adapters. A manifest links:

- a markdown skill id;
- an MCP server and MCP tool name;
- typed request/response schema names;
- required runtime tool permissions;
- expected artifact contracts.

The manifest layer does not execute domain calculations. It defines the adapter
boundary that future MCP servers must implement.

## Current Domain Skills

- `portfolio_risk_analysis`: portfolio concentration, risk segmentation,
  position contribution, and defensive-vs-risky position analysis.
- `investment_market_analysis`: issuer/ticker/security analysis that combines
  source discovery, snapshot/reference evidence, price history, event/news
  evidence, and charts.
- `retail_sales_analysis`: sales slices, trends, channel/category comparisons,
  and plan-fact evidence over retail datasets.

`general_analytics` remains a generic tabular analytics workflow. It must not
own investment, portfolio, sales, risk, or customer-specific recipes.

## Refactor Rule

When touching `backend/agent/runner.py`, do not add new domain scenarios. Move
domain behavior to one of these places instead:

- backend service: reusable product logic;
- MCP/domain extension: vertical executable operation;
- markdown skill: model instructions and evidence rules;
- typed tool: executable operation with strict input/output schemas.

Existing domain helpers in `runner.py` should be removed incrementally by first
adding contract tests, then routing the workflow through a skill/tool contract,
and only then deleting the legacy fallback.
