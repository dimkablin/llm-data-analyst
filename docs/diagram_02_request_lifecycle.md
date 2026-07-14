# Diagram 2 - Request Lifecycle

The query endpoints keep FastAPI concerns at the route boundary and hand the
runtime flow to `QueryExecutionService`.

```mermaid
sequenceDiagram
    actor User
    participant UI as React frontend
    participant Route as query.py route
    participant Service as QueryExecutionService
    participant Runner as AgentRunner
    participant Graph as LangGraph graph
    participant Context as AgentContextBuilder
    participant Tools as Typed tools
    participant Store as SessionStore

    User->>UI: Submit query
    UI->>Route: POST /sessions/{id}/query or /query/stream
    Route->>Service: QueryExecutionRequest or QueryStreamExecutionRequest
    Service->>Service: validate ownership, settings, skills, tools, sources
    Service->>Runner: run(AgentRunRequest)
    Runner->>Graph: invoke(AgentGraphState)
    Graph->>Context: prepare context
    Context-->>Graph: tool context, skill requirements, runtime metadata
    Graph->>Tools: agent node executes tool-call loop
    Tools-->>Graph: typed artifacts and tool messages
    Graph->>Graph: finalize response
    Graph-->>Runner: AgentResponse
    Runner-->>Service: AgentRunResult
    Service->>Store: persist messages, artifacts, runtime effects
    Service-->>Route: QueryResponse or SSE events
    Route-->>UI: final answer and artifacts
    UI-->>User: Render response
```

## Streaming Path

For `/query/stream`, `prepare_stream(...)` runs before `StreamingResponse` is
created, so ownership and request setup errors remain normal HTTP errors. After
that, `stream_events(...)` owns the event loop and emits:

- `start`
- token and reasoning events from callbacks
- `tool_start` / `tool_end`
- `phase` and `execution_graph`
- `final`
- `done`

The route formats these service events as SSE. It does not build runtime state,
instantiate the agent directly, or call private runner methods.
