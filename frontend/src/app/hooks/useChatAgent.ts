import { useCallback, useEffect, useRef, useState } from "react";

import { deleteLastMessages, getSession, streamQuery } from "../lib/backend-api";
import type {
  ArtifactPayload,
  AssistantBlock,
  ChatMessage,
  ExecutionGraph,
  PhaseEvent,
  QueryResponse,
  SessionState,
  StreamToolCall,
} from "../lib/backend-types";

const META_TOOLS = new Set(["get_tool_instructions", "planner_tool", "review_tool"]);

function parseInputSummary(toolName: string, raw: string): string {
  if (META_TOOLS.has(toolName)) return "";
  const trimmed = raw.trim();
  if (!trimmed) return "";
  try {
    const parsed = JSON.parse(trimmed) as Record<string, unknown>;
    if (typeof parsed.code === "string") return parsed.code.split("\n")[0]?.slice(0, 60) ?? "";
    if (typeof parsed.query === "string") return parsed.query.slice(0, 60);
    if (typeof parsed.path === "string") return parsed.path;
    if (typeof parsed.file_path === "string") return parsed.file_path;
    if (typeof parsed.dataset === "string") return parsed.dataset;
    if (typeof parsed.alias === "string") return parsed.alias;
    if (typeof parsed.pattern === "string") return `"${parsed.pattern}"`;
    if (typeof parsed.command === "string") return parsed.command.split("\n")[0]?.slice(0, 60) ?? "";
  } catch {
    /* raw string fallback */
  }
  return trimmed.slice(0, 60).replace(/\n/g, " ");
}

type LiveReasoningSnapshot = {
  fingerprint: string;
  liveReasoningTrace: string | null;
  livePhases: PhaseEvent[];
  tools?: StreamToolCall[];
};

function liveReasoningStorageKey(sessionId: string): string {
  return `llm_live_reasoning_snapshot_${sessionId}`;
}

function normalizeFingerprintPart(value: string | null | undefined): string {
  return String(value ?? "").trim().replace(/\s+/g, " ").slice(0, 4000);
}

function buildAssistantMessageFingerprint(message: {
  role?: string;
  content?: string | null;
  artifacts?: ArtifactPayload[] | undefined;
}): string | null {
  if (message.role !== "assistant") {
    return null;
  }
  const content = normalizeFingerprintPart(message.content);
  if (!content) {
    return null;
  }
  const artifactIds = (message.artifacts ?? [])
    .map((artifact) => String(artifact.id || artifact.text || artifact.type || "").trim())
    .filter(Boolean)
    .join("|");
  return `${content}::${artifactIds}`;
}

function saveLiveReasoningSnapshot(
  sessionId: string,
  message: {
    role?: string;
    content?: string | null;
    artifacts?: ArtifactPayload[] | undefined;
  },
  liveReasoningTrace: string | null | undefined,
  livePhases: PhaseEvent[] | undefined,
  tools?: StreamToolCall[] | undefined,
): void {
  if (typeof window === "undefined" || !sessionId) {
    return;
  }
  const fingerprint = buildAssistantMessageFingerprint(message);
  const normalizedTrace = String(liveReasoningTrace ?? "").trim();
  const normalizedPhases = Array.isArray(livePhases) ? livePhases.filter(Boolean) : [];
  const normalizedTools = Array.isArray(tools) && tools.length > 0 ? tools : undefined;
  if (!fingerprint || (!normalizedTrace && normalizedPhases.length === 0 && !normalizedTools)) {
    return;
  }
  const snapshot: LiveReasoningSnapshot = {
    fingerprint,
    liveReasoningTrace: normalizedTrace || null,
    livePhases: normalizedPhases,
    tools: normalizedTools,
  };
  try {
    window.sessionStorage.setItem(
      liveReasoningStorageKey(sessionId),
      JSON.stringify(snapshot),
    );
  } catch {
    // Ignore storage failures; live reasoning remains available in memory.
  }
}

function loadLiveReasoningSnapshot(sessionId: string): LiveReasoningSnapshot | null {
  if (typeof window === "undefined" || !sessionId) {
    return null;
  }
  try {
    const raw = window.sessionStorage.getItem(liveReasoningStorageKey(sessionId));
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw) as LiveReasoningSnapshot;
    if (!parsed || typeof parsed !== "object") {
      return null;
    }
    if (!String(parsed.fingerprint || "").trim()) {
      return null;
    }
    return {
      fingerprint: String(parsed.fingerprint),
      liveReasoningTrace: String(parsed.liveReasoningTrace ?? "").trim() || null,
      livePhases: Array.isArray(parsed.livePhases) ? parsed.livePhases : [],
      tools: Array.isArray(parsed.tools) && parsed.tools.length > 0 ? parsed.tools : undefined,
    };
  } catch {
    return null;
  }
}

function applyLiveReasoningSnapshot(
  sessionId: string,
  messages: ChatMessage[],
): ChatMessage[] {
  const snapshot = loadLiveReasoningSnapshot(sessionId);
  if (!snapshot || messages.length === 0) {
    return messages;
  }

  let lastAssistantIndex = -1;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "assistant") {
      lastAssistantIndex = index;
      break;
    }
  }
  if (lastAssistantIndex < 0) {
    return messages;
  }

  const candidate = messages[lastAssistantIndex];
  const fingerprint = buildAssistantMessageFingerprint(candidate);
  if (!fingerprint || fingerprint !== snapshot.fingerprint) {
    return messages;
  }

  const copy = [...messages];
  copy[lastAssistantIndex] = {
    ...candidate,
    liveReasoningTrace: snapshot.liveReasoningTrace,
    livePhases: snapshot.livePhases.length > 0 ? snapshot.livePhases : undefined,
    tools: snapshot.tools ?? candidate.tools,
  };
  return copy;
}

function toChatMessages(
  sessionId: string,
  history: Array<{
    id?: string;
    role: string;
    content: string;
    timestamp: string;
    reasoning?: string | null;
    artifacts?: ArtifactPayload[];
  }>,
): ChatMessage[] {
  const messages = history.map((item, index) => ({
    // Prefer the backend UUID as the React key when available for stability.
    id: item.id ?? `${item.timestamp}-${index}`,
    backendId: item.id,
    timestamp: item.timestamp,
    role: item.role === "user" ? "user" : ("assistant" as const),
    content: item.content,
    reasoning: item.reasoning ?? null,
    artifacts: item.artifacts ?? [],
  }));
  return applyLiveReasoningSnapshot(sessionId, messages as ChatMessage[]);
}

function buildStreamingReasoning(
  reasoning: string,
): string | null {
  const cleanReasoning = reasoning.trim();
  return cleanReasoning || null;
}

function mergeReasoning(
  primary: string | null | undefined,
  fallback: string | null | undefined,
): string | null {
  const normalized: string[] = [];
  const seen = new Set<string>();
  for (const item of [primary, fallback]) {
    const clean = String(item ?? "").trim();
    if (!clean || seen.has(clean)) {
      continue;
    }
    seen.add(clean);
    normalized.push(clean);
  }
  return normalized.length > 0 ? normalized.join("\n\n") : null;
}

type UseChatAgentArgs = {
  sessionId: string;
  includeReasoning: boolean;
  useHistory: boolean;
  analysisDepth?: string;
};

type UseChatAgentResult = {
  messages: ChatMessage[];
  artifacts: ArtifactPayload[];
  isStreaming: boolean;
  streamingSessionId: string | null;
  streamDraft: string;
  streamReasoning: string;
  streamPhases: PhaseEvent[];
  streamTools: StreamToolCall[];
  streamBlocks: AssistantBlock[];
  streamGraph: ExecutionGraph | null;
  error: string | null;
  lastQuery: string | null;
  hydrate: (
    session: SessionState,
    options?: { preserveStreamingForSessionId?: string | null },
  ) => void;
  sendQuery: (query: string) => Promise<void>;
  retryLast: () => Promise<void>;
  stopStreaming: () => void;
  reset: () => void;
  clearError: () => void;
  setErrorMessage: (value: string | null) => void;
};

export function useChatAgent({
  sessionId,
  includeReasoning,
  useHistory,
  analysisDepth,
}: UseChatAgentArgs): UseChatAgentResult {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [artifacts, setArtifacts] = useState<ArtifactPayload[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingSessionId, setStreamingSessionId] = useState<string | null>(null);
  const [streamDraft, setStreamDraft] = useState("");
  const [streamReasoning, setStreamReasoning] = useState("");
  const [streamPhases, setStreamPhases] = useState<PhaseEvent[]>([]);
  const [streamTools, setStreamTools] = useState<StreamToolCall[]>([]);
  const [streamBlocks, setStreamBlocks] = useState<AssistantBlock[]>([]);
  const [streamGraph, setStreamGraph] = useState<ExecutionGraph | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastQuery, setLastQuery] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const messagesRef = useRef<ChatMessage[]>([]);
  messagesRef.current = messages;

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const isStreamingRef = useRef(isStreaming);
  isStreamingRef.current = isStreaming;

  const hydrate = useCallback((
    session: SessionState,
    options?: { preserveStreamingForSessionId?: string | null },
  ) => {
    const preserveStreaming =
      isStreamingRef.current &&
      options?.preserveStreamingForSessionId &&
      options.preserveStreamingForSessionId === session.session_id;
    const hydratedMessages = toChatMessages(session.session_id, session.chat_history);
    const hasArtifactMessages = hydratedMessages.some((item) => (item.artifacts?.length ?? 0) > 0);
    if (!hasArtifactMessages && session.artifacts.length > 0) {
      hydratedMessages.push({
        id: `restored-${Date.now()}`,
        timestamp: new Date().toISOString(),
        role: "assistant",
        content: "Артефакты восстановлены из сохраненной сессии.",
        artifacts: session.artifacts,
      });
    }
    if (!preserveStreaming) {
      setMessages(hydratedMessages);
    }
    setArtifacts(session.artifacts);
    setError(null);
    if (!preserveStreaming) {
      setStreamDraft("");
      setStreamReasoning("");
      setStreamPhases([]);
      setStreamTools([]);
      setStreamBlocks([]);
      setIsStreaming(false);
      setLastQuery(null);
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const setErrorMessage = useCallback((value: string | null) => {
    setError(value);
  }, []);

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
      setMessages([]);
      setArtifacts([]);
      setIsStreaming(false);
      setStreamingSessionId(null);
      setStreamDraft("");
    setStreamReasoning("");
    setStreamPhases([]);
    setStreamTools([]);
    setStreamBlocks([]);
    setError(null);
    setLastQuery(null);
  }, []);

  const sendQuery = useCallback(
    async (query: string) => {
      const prompt = query.trim();
      if (!sessionId || !prompt || isStreaming) {
        return;
      }

      setError(null);
      setLastQuery(prompt);
      setStreamDraft("");
      setStreamReasoning("");
      setStreamPhases([]);
      setStreamTools([]);
      setStreamBlocks([]);
      setStreamGraph(null);
      setIsStreaming(true);
      setStreamingSessionId(sessionId);
      setMessages((prev) => [
        ...prev,
        {
          id: `u-${Date.now()}`,
          timestamp: new Date().toISOString(),
          role: "user",
          content: prompt,
        },
      ]);

      const streamState: { finalPayload?: QueryResponse; streamedText: string } = {
        streamedText: "",
      };
      const collectedPhases: PhaseEvent[] = [];
      const collectedTools: StreamToolCall[] = [];
      const collectedBlocks: AssistantBlock[] = [];
      let blockCounter = 0;
      const nextBlockId = () => `blk-${++blockCounter}`;
      // Accumulated complete thinking blocks (one per LLM call, filled by thinking_end)
      let collectedReasoning = "";
      // The last complete thinking block waiting to be attached to the next tool call
      let pendingThinkingBlock = "";
      // Track accumulated visible text between tool calls for text blocks
      let pendingIntentText = "";
      let aborted = false;
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await streamQuery(
          sessionId,
          prompt,
          includeReasoning,
          useHistory,
          {
            onToken: (token) => {
              streamState.streamedText += token;
              pendingIntentText += token;
              setStreamDraft((prev) => prev + token);
            },
            onFinal: (payload) => {
              streamState.finalPayload = payload;
            },
            onReasoning: (reasoningChunk, mode) => {
              if (!includeReasoning || !reasoningChunk) return;
              if (mode === "token") {
                // Live display only — complete block arrives via onThinkingEnd
                setStreamReasoning((prev) => prev + reasoningChunk);
              }
              // chunk mode (from emit_live_reasoning progress events) — ignored,
              // tool activity is shown via tool_start / tool_end events
            },
            onThinkingStart: () => {
              // New thinking block started — clear live display
              setStreamReasoning("");
            },
            onThinkingEnd: (text) => {
              // Complete thinking block for this LLM call
              const trimmed = text.trim();
              if (trimmed) {
                pendingThinkingBlock = trimmed;
                collectedReasoning = collectedReasoning
                  ? `${collectedReasoning}\n\n${trimmed}`
                  : trimmed;
                // Add thinking block to block timeline
                collectedBlocks.push({
                  type: "thinking",
                  id: nextBlockId(),
                  content: trimmed,
                });
                setStreamBlocks([...collectedBlocks]);
              }
              setStreamReasoning("");
            },
            onPhase: (phaseEvent) => {
              const mergedEvent = phaseEvent;
              if (mergedEvent.id) {
                const idx = collectedPhases.findIndex((p) => p.id === mergedEvent.id);
                if (idx >= 0) {
                  collectedPhases[idx] = mergedEvent;
                } else {
                  collectedPhases.push(mergedEvent);
                }
              } else {
                collectedPhases.push(mergedEvent);
              }
              setStreamPhases((prev) => {
                if (mergedEvent.id) {
                  const idx = prev.findIndex((p) => p.id === mergedEvent.id);
                  if (idx >= 0) {
                    const copy = [...prev];
                    copy[idx] = mergedEvent;
                    return copy;
                  }
                }
                return [...prev, mergedEvent];
              });
            },
            onToolStart: (event) => {
              if (META_TOOLS.has(event.tool_name)) return;
              const callId = `tool-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
              const inputSummary = event.input_summary || parseInputSummary(event.tool_name, event.input_preview ?? "");
              // Attach the last complete thinking block to this tool call
              const preReasoning = pendingThinkingBlock;
              pendingThinkingBlock = "";

              // Flush pending intent text as a text block (pre-tool narration)
              const intentText = pendingIntentText.trim();
              pendingIntentText = "";
              if (intentText) {
                collectedBlocks.push({
                  type: "text",
                  id: nextBlockId(),
                  content: intentText,
                });
              }

              // Add tool_use block
              const toolBlockId = nextBlockId();
              collectedBlocks.push({
                type: "tool_use",
                id: toolBlockId,
                tool_name: event.tool_name,
                input_summary: inputSummary,
                input_code: event.input_code || undefined,
                input_preview: event.input_preview || undefined,
                status: "running",
                started_at: Date.now(),
              });
              setStreamBlocks([...collectedBlocks]);

              const entry: StreamToolCall = {
                id: callId,
                tool_name: event.tool_name,
                input_summary: inputSummary,
                input_preview: event.input_preview || undefined,
                status: "running",
                started_at: Date.now(),
                pre_reasoning: preReasoning || undefined,
              };
              collectedTools.push(entry);
              setStreamTools((prev) => [...prev, entry]);
            },
            onToolEnd: (event) => {
              if (META_TOOLS.has(event.tool_name)) return;
              const endPatch = {
                status: (event.status === "ok" ? "done" : "error") as "done" | "error",
                artifact_keys: event.artifact_keys ?? [],
                ...(event.output_preview ? { output_preview: event.output_preview } : {}),
              };
              // Update collectedTools in-place
              for (let i = collectedTools.length - 1; i >= 0; i--) {
                if (collectedTools[i]!.tool_name === event.tool_name && collectedTools[i]!.status === "running") {
                  collectedTools[i] = { ...collectedTools[i]!, ...endPatch };
                  break;
                }
              }

              // Update tool_use block status in collectedBlocks
              for (let i = collectedBlocks.length - 1; i >= 0; i--) {
                const blk = collectedBlocks[i]!;
                if (blk.type === "tool_use" && blk.tool_name === event.tool_name && blk.status === "running") {
                  collectedBlocks[i] = {
                    ...blk,
                    status: event.status === "ok" ? "done" : "error",
                    input_code: blk.input_code,
                  };
                  // Add tool_result block right after
                  collectedBlocks.push({
                    type: "tool_result",
                    id: nextBlockId(),
                    tool_use_id: blk.id,
                    tool_name: event.tool_name,
                    status: (event.status === "ok" ? "ok" : "error") as "ok" | "error",
                    result_summary: event.result_summary || "",
                    output_preview: event.output_preview,
                    artifact_keys: event.artifact_keys,
                  });
                  break;
                }
              }
              setStreamBlocks([...collectedBlocks]);

              setStreamTools((prev) => {
                const copy = [...prev];
                for (let i = copy.length - 1; i >= 0; i--) {
                  if (copy[i]!.tool_name === event.tool_name && copy[i]!.status === "running") {
                    copy[i] = { ...copy[i]!, ...endPatch };
                    break;
                  }
                }
                return copy;
              });
            },
            onGraphUpdate: (graph) => {
              setStreamGraph(graph);
            },
            onError: (streamError) => {
              setError(streamError);
            },
          },
          controller.signal,
          analysisDepth,
        );
      } catch (err) {
        if ((err as Error)?.name === "AbortError") {
          aborted = true;
        } else {
          setError(String(err));
        }
      } finally {
        abortRef.current = null;
        setIsStreaming(false);
        setStreamingSessionId(null);
        setStreamDraft("");
        setStreamReasoning("");
        setStreamPhases([]);
        setStreamTools([]);
        setStreamBlocks([]);
      }

      if (aborted) {
        const partialReasoning = buildStreamingReasoning(
          collectedReasoning,
        );
        if (streamState.streamedText.trim() || partialReasoning) {
          setMessages((prev) => [
            ...prev,
            {
              id: `a-aborted-${Date.now()}`,
              timestamp: new Date().toISOString(),
              role: "assistant",
              content:
                streamState.streamedText.trim() ||
                "_Генерация остановлена пользователем до появления итогового текста._",
              reasoning: partialReasoning,
              phases: collectedPhases.length > 0 ? [...collectedPhases] : undefined,
              tools: collectedTools.length > 0 ? [...collectedTools] : undefined,
              blocks: collectedBlocks.length > 0 ? [...collectedBlocks] : undefined,
            },
          ]);
        }
        return;
      }

      const finalPayload = streamState.finalPayload;
      const savedPhases = collectedPhases.filter((p) => p.status === "done");
      const fallbackReasoning = buildStreamingReasoning(
        collectedReasoning,
      );
      if (finalPayload) {
        const finalReasoning = mergeReasoning(
          finalPayload.reasoning,
          fallbackReasoning,
        );
        const livePhases = collectedPhases.length > 0 ? [...collectedPhases] : undefined;
        saveLiveReasoningSnapshot(
          sessionId,
          {
            role: "assistant",
            content: finalPayload.text,
            artifacts: finalPayload.artifacts,
          },
          fallbackReasoning,
          livePhases,
          collectedTools.length > 0 ? [...collectedTools] : undefined,
        );
        setMessages((prev) => [
          ...prev,
          {
            id: `a-${Date.now()}`,
            timestamp: new Date().toISOString(),
            role: "assistant",
            content: finalPayload.text,
            reasoning: finalReasoning,
            phases: savedPhases.length > 0 ? savedPhases : undefined,
            tools: collectedTools.length > 0 ? [...collectedTools] : undefined,
            blocks: collectedBlocks.length > 0 ? [...collectedBlocks] : undefined,
            liveReasoningTrace: fallbackReasoning,
            livePhases,
            metrics: finalPayload.metrics,
            artifacts: finalPayload.artifacts,
            executionGraph: (finalPayload as Record<string, unknown>).execution_graph as ExecutionGraph | undefined,
          },
        ]);
        setArtifacts((prev) => [...prev, ...finalPayload.artifacts]);
        return;
      }

      try {
        const recoveredSession = await getSession(sessionId);
        const recoveredHistory = toChatMessages(
          recoveredSession.session_id,
          recoveredSession.chat_history,
        );
        const lastRecoveredAssistant = [...recoveredHistory]
          .reverse()
          .find((message) => message.role === "assistant");
        if (lastRecoveredAssistant) {
          saveLiveReasoningSnapshot(
            sessionId,
            lastRecoveredAssistant,
            fallbackReasoning,
            collectedPhases.length > 0 ? [...collectedPhases] : undefined,
            collectedTools.length > 0 ? [...collectedTools] : undefined,
          );
        }
        const hydratedWithLiveTrace = applyLiveReasoningSnapshot(
          recoveredSession.session_id,
          recoveredHistory,
        );
        if (hydratedWithLiveTrace.length > 0) {
          setMessages(hydratedWithLiveTrace);
          setArtifacts(recoveredSession.artifacts);
          return;
        }
      } catch {
        // Keep local fallback.
      }

      if (streamState.streamedText.trim()) {
        setMessages((prev) => [
          ...prev,
          {
            id: `a-${Date.now()}`,
            timestamp: new Date().toISOString(),
            role: "assistant",
            content: streamState.streamedText,
            reasoning: fallbackReasoning,
            phases: savedPhases.length > 0 ? savedPhases : undefined,
          },
        ]);
        return;
      }

      setMessages((prev) => [
        ...prev,
        {
          id: `a-fallback-${Date.now()}`,
          timestamp: new Date().toISOString(),
          role: "assistant",
          content:
            "Ответ был сформирован на backend, но не доставлен в потоковом канале. Обновите чат или повторите запрос.",
        },
      ]);
    },
    [analysisDepth, includeReasoning, isStreaming, sessionId, useHistory],
  );

  const retryLast = useCallback(async () => {
    if (isStreaming || !sessionId) {
      return;
    }
    const msgs = messagesRef.current;
    // lastQuery is only populated during the current browser session.
    // Fall back to the content of the last user message so the button works
    // after a page reload or when the session is restored from the backend.
    let lastUserMsg: ChatMessage | undefined;
    for (let i = msgs.length - 1; i >= 0; i -= 1) {
      if (msgs[i].role === "user") {
        lastUserMsg = msgs[i];
        break;
      }
    }
    const query = lastQuery ?? lastUserMsg?.content ?? null;
    if (!query) {
      return;
    }
    // Delete from backend by the exact message ID so history stays consistent.
    if (lastUserMsg?.backendId) {
      try {
        await deleteLastMessages(sessionId, lastUserMsg.backendId);
      } catch {
        // Best-effort: continue even if the message wasn't persisted yet.
      }
    }
    // Mirror the deletion in local state (remove last user + assistant pair).
    setMessages((prev) => (prev.length >= 2 ? prev.slice(0, -2) : []));
    await sendQuery(query);
  }, [isStreaming, lastQuery, sendQuery, sessionId]);

  return {
    messages,
    artifacts,
    isStreaming,
    streamingSessionId,
    streamDraft,
    streamReasoning,
    streamPhases,
    streamTools,
    streamBlocks,
    streamGraph,
    error,
    lastQuery,
    hydrate,
    sendQuery,
    retryLast,
    stopStreaming,
    reset,
    clearError,
    setErrorMessage,
  };
}
