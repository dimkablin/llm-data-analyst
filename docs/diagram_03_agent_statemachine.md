# Diagram 3 - Agent State Machine

Current LangGraph topology is intentionally small and domain-neutral.

```mermaid
flowchart TD
    START([START]) --> Prepare["prepare_context\nAgentContextBuilder"]
    Prepare -->|"done or response already set"| Finalize["finalize\nfinal response orchestration"]
    Prepare -->|"needs tool loop"| Agent["agent\ndirect_tool_loop"]
    Agent --> Finalize
    Finalize --> END([END])

    subgraph Context["Context preparation"]
        ToolCtx["tool catalog and allowed tools"]
        SkillCtx["selected skills and requirements"]
        SourceCtx["session/source/DB runtime metadata"]
        FutureCtx["future budget and vector-retrieval policies"]
    end

    subgraph Runtime["Generic runtime responsibilities"]
        State["typed state"]
        Events["runtime events"]
        Artifacts["artifact references"]
        Validation["generic skill/tool contract validation"]
    end

    subgraph Extensions["Domain extensions"]
        Skills["SKILL.md instructions"]
        Tools["typed tools"]
        Manifests["domain extension manifests"]
        MCP["MCP servers"]
    end

    Prepare --> Context
    Agent --> Runtime
    Runtime --> Extensions
```

## Node Responsibilities

- `prepare_context` builds generic context. It does not classify customer
  scenarios by keywords and does not short-circuit chat, summary, or reports.
- `agent` calls the LLM and executes typed tools through the generic tool loop.
- `finalize` performs generic final response orchestration and validation. It
  may append warnings for failed checks, but it must not replace good model
  answers with deterministic domain summaries.

New investment, sales, risk, forecasting, or customer-specific behavior should
enter through skills/tools/domain manifests/MCP, not by changing graph topology.
