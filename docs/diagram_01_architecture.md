# Diagram 1 - System Architecture

High-level view of the current production architecture after the agent runtime
split. The important boundary is that the LangGraph runtime is a generic engine;
domain behavior enters through skills, typed tools, domain extension manifests,
and future MCP servers.

```mermaid
flowchart TD
    User([User])

    subgraph Frontend["React frontend"]
        UI["Customer-specific screens\nchat, sessions, settings, artifacts"]
        Contracts["Typed API contracts"]
    end

    subgraph Backend["FastAPI backend"]
        Routes["Thin API routes"]
        QueryService["QueryExecutionService\nrequest validation, session ownership,\nruntime setup, persistence, streaming"]
        Auth["Auth and user settings"]
        Sessions["SessionStore and artifact storage"]
        Sources["Source inventory and runtime data access"]
    end

    subgraph AgentRuntime["Generic LangGraph runtime"]
        Runner["AgentRunner\ncomposition root + public run/run_query"]
        Graph["StateGraph\nprepare_context -> agent -> finalize"]
        Context["AgentContextBuilder\nsession/source/tool/skill context"]
        ToolLoop["direct_tool_loop\nLLM tool-call loop"]
        Finalize["finalize_node\nfinal response orchestration"]
    end

    subgraph Extensions["Extension layer"]
        Skills["Markdown skills\ninstructions and evidence rules"]
        Tools["Typed tools\nstrict input/output artifacts"]
        DomainRegistry["DomainExtensionRegistry\nMCP/tool/skill manifests"]
        MCP["Future MCP servers"]
    end

    subgraph Providers["External providers"]
        LLM["LLM provider"]
        Search["Search / RAG / forecasting / anomaly services"]
    end

    User --> UI
    UI --> Contracts
    Contracts --> Routes
    Routes --> QueryService
    QueryService --> Auth
    QueryService --> Sessions
    QueryService --> Sources
    QueryService --> Runner
    Runner --> Graph
    Graph --> Context
    Graph --> ToolLoop
    Graph --> Finalize
    Context --> Skills
    Context --> DomainRegistry
    ToolLoop --> Tools
    Tools --> Search
    ToolLoop --> LLM
    Finalize --> Sessions
    DomainRegistry --> MCP
```

## Runtime Boundaries

- `backend/api/routes/query.py` is a thin FastAPI boundary. It delegates
  synchronous and streaming execution to `QueryExecutionService`.
- `QueryExecutionService` owns backend request concerns: auth/session checks,
  user settings, selected skills, tool permissions, source context, persistence,
  streaming callbacks, and fallback API responses.
- `AgentRunner` is the composition root for the generic runtime. It wires
  dependencies, compiles the graph, owns the small query cache, exposes
  `run(AgentRunRequest)`, and keeps `run_query(...)` as a compatibility adapter.
- The graph has three current nodes: `prepare_context`, `agent`, and `finalize`.
  It does not route chat/summary/report through keyword shortcuts.
- Domain behavior belongs in extension contracts: markdown skills, typed tools,
  domain extension manifests, and future MCP servers.
