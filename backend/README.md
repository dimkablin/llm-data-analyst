# Backend API

## Run locally

```bash
poetry run uvicorn backend.app:app --host 0.0.0.0 --port ${BACKEND_PORT:-8000} --reload
```

## Main endpoints

- `POST /sessions` - create session
- `GET /sessions/{session_id}` - get chat history and artifacts
- `POST /sessions/{session_id}/data` - upload CSV into session
- `POST /sessions/{session_id}/query` - run agent, returns text + artifacts + metrics
- `POST /sessions/{session_id}/evaluate` - run non-persistent evaluation (does not append to chat/artifacts)
- `POST /sessions/{session_id}/query/stream` - SSE token stream + final payload

## Limits and retention

- Session TTL: `BACKEND_SESSION_TTL_DAYS` (default `7` days)
- Dataset max size: `BACKEND_MAX_DATASET_MB` (default `100` MB)
- Session storage path: `BACKEND_STORAGE_DIR`

## External integrations (env)

- `SEARCH_BACKEND_URL` - quick external search backend (search_service)
- `RAG_URL` - knowledge-base / RAG backend base URL
- `RAG_ENABLED` - force on/off RAG integration
- `RAG_QUERY_ENDPOINT` - sync query endpoint (default `/query`)
- `RAG_STREAM_ENDPOINT` - streaming query endpoint (default `/query/stream`)
- `RAG_TIMEOUT_SEC` - RAG request timeout (default `45`)
- `RAG_QUERY_MODE` - default RAG query mode (default `hybrid`)
- `RAG_TOP_K` - default top-k for RAG retrieval (default `5`)
- `RAG_VERIFY_SSL` - verify TLS certs for HTTPS RAG backends

## SSE event types

- `start` - stream opened, includes `session_id`
- `token` - LLM token chunk
- `reasoning` - live reasoning/tool-progress markdown chunk (when `include_reasoning=true`)
- `reasoning_token` - live model-thinking token chunk (streamed token-by-token from `<think>...</think>`)
- `final` - final response payload (same schema as `/query`)
- `error` - execution error payload

## Request flags

- `use_history` - when `true`, previous chat history is passed into the agent
- `include_reasoning` - enables full model reasoning mode (more tokens / thinking) for the current request
  and streams live tool/code progress in SSE mode

## Agent controls (env)

- `AGENT_MAX_STEPS` - max Reason-Action loop steps (`Plan -> Act -> Observe -> Decide`, default `5`)
- `AGENT_STEP_TIMEOUT_SEC` - per-step timeout budget for tool-action stage (default `45`)
- `AGENT_INNER_RECURSION_LIMIT` - max inner LangGraph recursion for one ACT-step (default `6`)
- `AGENT_HISTORY_MAX_MESSAGES` - last N chat messages passed directly into prompt (default `8`)
- `AGENT_HISTORY_SUMMARY_CHARS` - compact summary cap for older history (default `700`)
- `AGENT_PROMPT_MAX_COLUMNS` - max dataset columns included in system prompt profile (default `16`)
- `AGENT_PROMPT_HEAD_ROWS` - head rows included in system prompt (default `3`)
- `AGENT_CACHE_ENABLED` - query cache on/off (default `true`)
- `AGENT_CACHE_SIZE` - LRU query cache size (default `128`)
- `AGENT_CACHE_TTL_SEC` - query cache TTL in seconds (default `900`)

## LLM/tool performance controls (env)

- `LLM_PROVIDER` - `ollama` (default) or `vllm`
- `LLM_OLLAMA_API_URL` / `LLM_OLLAMA_MODEL_NAME` / `LLM_OLLAMA_API_KEY` - Ollama provider settings
- `LLM_VLLM_API_URL` / `LLM_VLLM_MODEL_NAME` / `LLM_VLLM_API_KEY` - vLLM provider settings
- `LLM_CHAT_TEMPLATE_KWARGS_ENABLED` - send `chat_template_kwargs` in request body; default `true` for `ollama`, `false` for `vllm`
- `LLM_TEMPERATURE_CHAT` - temperature for non-thinking chat/final narrative (default `0.7`; overridden to `1.0` when thinking is on per Qwen3.5 spec)
- `LLM_TEMPERATURE_TOOL` - temperature for tool-use steps (default `0.5`)
- `LLM_TOP_P` - nucleus sampling top-p (default `0.95`; `0.8` for non-thinking)
- `LLM_TOP_K` - top-k sampling (default `20`; passed via `extra_body`)
- `LLM_PRESENCE_PENALTY` - presence penalty (default `1.5`; `0` for evaluate)
- `LLM_MAX_TOKENS_DEFAULT` - default token budget per model call (default `1200`)
- `LLM_MAX_TOKENS_REASONING` - token budget when `include_reasoning=true` (default `2200`)
- `TOOL_CACHE_SIZE` - per-tool in-memory code-result cache size (default `48`)
- `LLM_STREAMING_FORCE` - always keep model streaming enabled (default `true`)
- `LLM_WARMUP_ENABLED` - best-effort warmup ping at backend startup (default `true`)
- `LLM_WARMUP_TIMEOUT_SEC` - timeout for warmup ping (default `12`)

## Tool output contract

Each tool call must return `tool_result` using strict JSON schema:

```json
{
  "schema_version": "1.0",
  "artifact_type": "plot|table|value",
  "items": { "name": "<payload>" }
}
```

If schema is violated, tool result is rejected and the agent retries.

## Observability (Phoenix)

- Traces are annotated with OpenInference attributes:
  - `session_id` = chat session id
  - `user_id` = backend auth user id
  - metadata includes `username`, `request_kind`, `use_history`, `include_reasoning`
- Query/Eval/Stream runs are tagged (`request:query`, `request:evaluate`, `request:stream`)
- This enables filtering traces by user and by chat session in Phoenix UI
