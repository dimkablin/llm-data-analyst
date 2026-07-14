# Diagram 9 - Thinking and Runtime Events

Streaming events are backend service concerns. The graph and tool loop emit
typed runtime information through callbacks; `QueryExecutionService` converts
that into SSE events for the frontend.

```mermaid
flowchart TD
    UI([React client]) --> Route["query.py\nSSE route"]
    Route --> Service["QueryExecutionService\nprepare_stream + stream_events"]
    Service --> Runner["AgentRunner.run"]
    Runner --> Graph["LangGraph\nprepare_context -> agent -> finalize"]

    Graph --> LLM["LLM callbacks"]
    Graph --> ToolCallbacks["Tool callbacks"]
    Graph --> PhaseCallbacks["Phase callbacks"]

    LLM --> Queue["async event queue"]
    ToolCallbacks --> Queue
    PhaseCallbacks --> Queue
    Service --> Queue
    Queue --> Route
    Route --> UI

    subgraph EventTypes["SSE event types"]
        Start["start"]
        Token["token / reasoning_token"]
        Thinking["thinking_start / thinking_end"]
        Tool["tool_start / tool_end"]
        Phase["phase"]
        GraphEvent["execution_graph"]
        Final["final"]
        Done["done"]
    end

    Queue --> EventTypes
```

## Event Rules

- The frontend consumes public SSE event payloads only. It must not depend on
  private runner methods or graph internals.
- Thinking visibility is controlled by user settings and provider policy.
- Tool events describe typed tool calls and artifact references; they should not
  encode customer-specific business logic.
- Runtime fallback responses are API/service-level safety behavior. Domain
  recovery belongs in typed tools, skills, or extension services.
