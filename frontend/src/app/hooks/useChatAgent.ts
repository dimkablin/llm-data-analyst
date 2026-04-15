import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";

import { deleteLastMessages, getSession, streamQuery } from "../lib/backend-api";
import type {
  ArtifactPayload,
  AssistantBlock,
  ChatMessage,
  ExecutionGraph,
  PersistedToolCall,
  PhaseEvent,
  QueryResponse,
  SessionState,
  StreamToolCall,
} from "../lib/backend-types";

const META_TOOLS = new Set<string>();

function parseInputSummary(toolName: string, raw: string): string {
  if (META_TOOLS.has(toolName)) return "";
  if (toolName === "get_tool_instructions") {
    try {
      const parsed = JSON.parse(raw.trim()) as Record<string, unknown>;
      return typeof parsed.tool_name === "string" ? parsed.tool_name : raw.trim().slice(0, 40);
    } catch {
      return raw.trim().slice(0, 40);
    }
  }
  const trimmed = raw.trim();
  if (!trimmed) return "";
  try {
    const parsed = JSON.parse(trimmed) as Record<string, unknown>;
    if (typeof parsed.code === "string") return parsed.code.split("\n")[0]?.slice(0, 60) ?? "";
    if (typeof parsed.query === "string") return parsed.query.slice(0, 60);
    if (typeof parsed.question === "string") return parsed.question.slice(0, 60);
    if (typeof parsed.answer === "string") return parsed.answer.split("\n")[0]?.slice(0, 60) ?? "";
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

function persistedToolToStream(p: PersistedToolCall, idx: number): StreamToolCall {
  return {
    id: `hist-${p.tool_name}-${idx}`,
    tool_name: p.tool_name,
    input_summary: p.input_summary ?? p.tool_name,
    input_preview: p.input_preview,
    status: p.status === "error" ? "error" : "done",
    artifact_keys: p.artifact_keys,
    started_at: p.started_at ? Date.parse(p.started_at) : 0,
    output_preview: p.error ? `Error: ${p.error}` : undefined,
    pre_reasoning: p.pre_reasoning || undefined,
  };
}

/**
 * Reconstruct an ordered AssistantBlock[] from persisted history so reload
 * renders thinking interleaved with tool calls exactly as it did live.
 *
 * Order rule (mirrors backend `_build_reasoning_steps` + live stream order):
 *   pre_reasoning_i → tool_use_i → tool_result_i (for each tool call i)
 *   followed by unique orphan reasoning_steps not already captured as pre_reasoning.
 *
 * Deduplication: the backend `_build_reasoning_steps` uses index-based mapping
 * between raw thinking blocks and tool calls. When internal LLM calls (e.g. inside
 * planner_tool) produce extra thinking blocks, the index mapping drifts and some
 * reasoning_steps end up containing content identical to a tool's pre_reasoning.
 * We deduplicate by content so thinking blocks do not appear twice.
 *
 * Returns undefined if there are no blocks to render (no tools & no orphan steps).
 */
function buildBlocksFromHistory(
  tools: StreamToolCall[] | undefined,
  reasoningSteps: import("../lib/backend-types").PersistedReasoningStep[] | null | undefined,
): AssistantBlock[] | undefined {
  const blocks: AssistantBlock[] = [];
  let counter = 0;
  const nextId = (prefix: string): string => `hist-blk-${prefix}-${counter++}`;

  // Collect trimmed pre_reasoning content for deduplication of orphan steps.
  const preReasoningSet = new Set<string>();

  // Index infra-tool reasoning steps by tool_name for inline fallback rendering.
  // These steps have tool_name set because pre_reasoning was discarded for infra tools
  // but the backend still persists them in reasoning_steps so reload can show them.
  const infraStepsByToolName = new Map<string, import("../lib/backend-types").PersistedReasoningStep[]>();
  if (reasoningSteps) {
    for (const step of reasoningSteps) {
      if (step.tool_name) {
        const arr = infraStepsByToolName.get(step.tool_name) ?? [];
        arr.push(step);
        infraStepsByToolName.set(step.tool_name, arr);
      }
    }
  }

  if (tools && tools.length > 0) {
    for (const tool of tools) {
      const trimmedPre = tool.pre_reasoning?.trim();
      if (trimmedPre) {
        preReasoningSet.add(trimmedPre);
        blocks.push({
          type: "thinking",
          id: nextId("think"),
          content: tool.pre_reasoning!,
          kind: "tool_synthesis",
        });
      } else {
        // Infra tool fallback: pre_reasoning was discarded on backend; use the
        // reasoning_step that was saved with this tool_name instead.
        const stepsForTool = infraStepsByToolName.get(tool.tool_name) ?? [];
        for (const step of stepsForTool) {
          const content = step.content?.trim();
          if (content && !preReasoningSet.has(content)) {
            preReasoningSet.add(content);
            blocks.push({
              type: "thinking",
              id: nextId(`rs-${step.step_index}`),
              content: step.content,
              kind: step.kind ?? "tool_synthesis",
            });
          }
        }
      }
      const toolUseId = nextId("tool");
      blocks.push({
        type: "tool_use",
        id: toolUseId,
        tool_name: tool.tool_name,
        input_summary: tool.input_summary,
        input_preview: tool.input_preview,
        status: tool.status,
        started_at: tool.started_at,
        result_summary: undefined,
        output_preview: tool.output_preview,
        artifact_keys: tool.artifact_keys,
      });
      blocks.push({
        type: "tool_result",
        id: nextId("res"),
        tool_use_id: toolUseId,
        tool_name: tool.tool_name,
        status: tool.status === "error" ? "error" : "ok",
        result_summary: "",
        output_preview: tool.output_preview,
        artifact_keys: tool.artifact_keys,
      });
    }
  }

  // Orphan reasoning steps (final_synthesis etc.) appear AFTER tool calls.
  // Skip any step already placed inline (either via pre_reasoning or infra fallback).
  if (reasoningSteps && reasoningSteps.length > 0) {
    for (const step of reasoningSteps) {
      const content = step.content?.trim();
      if (!content) continue;
      if (preReasoningSet.has(content)) continue; // already shown inline
      if (step.tool_name) continue; // infra step — already placed inline above
      blocks.push({
        type: "thinking",
        id: nextId(`rs-${step.step_index}`),
        content: step.content,
        kind: step.kind ?? "unknown",
      });
    }
  }

  return blocks.length > 0 ? blocks : undefined;
}

function toChatMessages(
  sessionId: string,
  history: Array<{
    id?: string;
    role: string;
    content: string;
    timestamp: string;
    reasoning?: string | null;
    reasoning_steps?: import("../lib/backend-types").PersistedReasoningStep[] | null;
    artifacts?: ArtifactPayload[];
    tools?: PersistedToolCall[];
  }>,
): ChatMessage[] {
  const messages = history.map((item, index) => {
    // Filter META_TOOLS (e.g. get_tool_instructions) to match live streaming behavior.
    // Backend persists all tool calls; frontend hides meta tools from the UI.
    const filteredPersistedTools = item.tools?.filter(
      (t) => !META_TOOLS.has(t.tool_name),
    );
    const tools = filteredPersistedTools?.length
      ? filteredPersistedTools.map(persistedToolToStream)
      : undefined;
    const blocks =
      item.role === "assistant" || item.role === "ai"
        ? buildBlocksFromHistory(tools, item.reasoning_steps ?? null)
        : undefined;
    return {
      id: item.id ?? `${item.timestamp}-${index}`,
      backendId: item.id,
      timestamp: item.timestamp,
      role: item.role === "user" ? "user" : ("assistant" as const),
      content: item.content,
      reasoning: item.reasoning ?? null,
      reasoning_steps: item.reasoning_steps ?? null,
      artifacts: item.artifacts ?? [],
      tools,
      blocks,
    };
  });
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

type SessionSlot = {
  messages: ChatMessage[];
  artifacts: ArtifactPayload[];
};

type UseChatAgentArgs = {
  sessionId: string;
  includeReasoning: boolean;
  useHistory: boolean;
  analysisDepth?: string;
  selectedSkillIds?: string[];
};

type UseChatAgentResult = {
  messages: ChatMessage[];
  artifacts: ArtifactPayload[];
  isStreaming: boolean;
  isStreamingCurrentSession: boolean;
  backgroundStreamingSessionId: string | null;
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
  selectedSkillIds,
}: UseChatAgentArgs): UseChatAgentResult {
  // Per-session storage: messages and artifacts are keyed by session ID.
  const [sessionData, setSessionData] = useState<Map<string, SessionSlot>>(new Map());
  const [displayedSessionId, setDisplayedSessionId] = useState("");

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

  // Derived values for the currently displayed session.
  const messages = sessionData.get(displayedSessionId)?.messages ?? [];
  const artifacts = sessionData.get(displayedSessionId)?.artifacts ?? [];

  // True only when the user is looking at the session that is currently streaming.
  const isStreamingCurrentSession = isStreaming && streamingSessionId === displayedSessionId;

  // Non-null when a stream is running in a session the user is NOT currently viewing.
  const backgroundStreamingSessionId =
    isStreaming && streamingSessionId !== null && streamingSessionId !== displayedSessionId
      ? streamingSessionId
      : null;

  const messagesRef = useRef<ChatMessage[]>([]);
  messagesRef.current = messages;

  // Ref so sendQuery always captures the session the user is actually viewing,
  // even if args.sessionId hasn't been updated yet via the bindChatAgent effect.
  const displayedSessionIdRef = useRef(displayedSessionId);
  displayedSessionIdRef.current = displayedSessionId;

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const isStreamingRef = useRef(isStreaming);
  isStreamingRef.current = isStreaming;

  const streamingSessionIdRef = useRef<string | null>(null);
  streamingSessionIdRef.current = streamingSessionId;

  // Helper: update the messages array for a specific session slot.
  const patchSlotMessages = useCallback(
    (sid: string, updater: (prev: ChatMessage[]) => ChatMessage[]) => {
      setSessionData((prev) => {
        const newMap = new Map(prev);
        const slot = newMap.get(sid) ?? { messages: [], artifacts: [] };
        newMap.set(sid, { ...slot, messages: updater(slot.messages) });
        return newMap;
      });
    },
    [],
  );

  // Helper: replace both messages and artifacts for a specific session slot.
  const replaceSlot = useCallback(
    (sid: string, nextMessages: ChatMessage[], nextArtifacts: ArtifactPayload[]) => {
      setSessionData((prev) => {
        const newMap = new Map(prev);
        newMap.set(sid, { messages: nextMessages, artifacts: nextArtifacts });
        return newMap;
      });
    },
    [],
  );

  // Helper: append artifacts for a specific session slot.
  const appendSlotArtifacts = useCallback(
    (sid: string, newArtifacts: ArtifactPayload[]) => {
      setSessionData((prev) => {
        const newMap = new Map(prev);
        const slot = newMap.get(sid) ?? { messages: [], artifacts: [] };
        newMap.set(sid, { ...slot, artifacts: [...slot.artifacts, ...newArtifacts] });
        return newMap;
      });
    },
    [],
  );

  const hydrate = useCallback((
    session: SessionState,
    options?: { preserveStreamingForSessionId?: string | null },
  ) => {
    const sid = session.session_id;
    // Should we keep the live messages for this session (it's currently streaming)?
    const isCurrentlyStreamingThis =
      isStreamingRef.current && streamingSessionIdRef.current === sid;
    const shouldPreserveMessages =
      isCurrentlyStreamingThis && options?.preserveStreamingForSessionId === sid;

    const hydratedMessages = toChatMessages(sid, session.chat_history);
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

    setSessionData((prev) => {
      const newMap = new Map(prev);
      if (shouldPreserveMessages) {
        // Keep live messages and merge artifacts: server state is canonical, but
        // preserve any locally-generated artifacts that aren't yet persisted
        // (e.g. the current streaming response that triggered this reload).
        const existing = newMap.get(sid);
        const serverIds = new Set(session.artifacts.map((a) => a.id));
        const localOnly = (existing?.artifacts ?? []).filter((a) => !serverIds.has(a.id));
        newMap.set(sid, {
          messages: existing?.messages ?? hydratedMessages,
          artifacts: [...session.artifacts, ...localOnly],
        });
      } else {
        newMap.set(sid, { messages: hydratedMessages, artifacts: session.artifacts });
      }
      return newMap;
    });

    setDisplayedSessionId(sid);
    setError(null);

    // Clear streaming display state only when switching to a non-streaming session.
    // The stream itself continues uninterrupted in its own session slot.
    if (!isCurrentlyStreamingThis) {
      setStreamDraft("");
      setStreamReasoning("");
      setStreamPhases([]);
      setStreamTools([]);
      setStreamBlocks([]);
      if (!isStreamingRef.current) {
        setLastQuery(null);
      }
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
    setSessionData(new Map());
    setDisplayedSessionId("");
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

      // Capture the session ID for this request — it must not change even if the
      // user navigates to a different session while the stream is in flight.
      // Prefer displayedSessionIdRef (updated synchronously by hydrate) over
      // sessionId from args (updated via useEffect, one render later).
      const capturedSessionId = displayedSessionIdRef.current || sessionId;

      setError(null);
      setLastQuery(prompt);
      setStreamDraft("");
      setStreamReasoning("");
      setStreamPhases([]);
      setStreamTools([]);
      setStreamBlocks([]);
      setStreamGraph(null);
      setIsStreaming(true);
      setStreamingSessionId(capturedSessionId);

      // Add the user message to the captured session's slot.
      patchSlotMessages(capturedSessionId, (prev) => [
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
      let collectedReasoning = "";
      let pendingThinkingBlock = "";
      let pendingIntentText = "";
      // Tokens emitted before thinking_start (vLLM strips <think> opening tag, so
      // reasoning content arrives as token events until </think> is seen).
      let prethinkPrefix = "";
      let aborted = false;
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await streamQuery(
          capturedSessionId,
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
                setStreamReasoning((prev) => prev + reasoningChunk);
              } else if (mode === "chunk") {
                setStreamReasoning(reasoningChunk);
              }
            },
            onThinkingStart: () => {
              // When vLLM strips the <think> opening tag, the backend emits token
              // events for all reasoning content until it finds </think>. Save
              // any accumulated tokens as a prefix for the upcoming thinking block.
              if (pendingIntentText) {
                prethinkPrefix = pendingIntentText;
                pendingIntentText = "";
                setStreamDraft("");
                // Populate the live thinking block immediately so the user sees
                // the full reasoning text (not just the 7-char buffer tail that
                // arrives as the sole reasoning_token in vLLM mode).
                setStreamReasoning(prethinkPrefix);
              } else {
                setStreamReasoning("");
              }
            },
            onThinkingEnd: (text) => {
              // Prepend any tokens that arrived before thinking_start was detected
              // (happens when vLLM strips the <think> opening tag server-side).
              const combined = prethinkPrefix
                ? (prethinkPrefix + text).trim()
                : text.trim();
              prethinkPrefix = "";
              if (combined) {
                pendingThinkingBlock = combined;
                collectedReasoning = collectedReasoning
                  ? `${collectedReasoning}\n\n${combined}`
                  : combined;
                collectedBlocks.push({
                  type: "thinking",
                  id: nextBlockId(),
                  content: combined,
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
              pendingThinkingBlock = "";

              const intentText = pendingIntentText.trim();
              pendingIntentText = "";
              if (intentText) {
                collectedBlocks.push({
                  type: "text",
                  id: nextBlockId(),
                  content: intentText,
                });
              }

              const toolBlockId = nextBlockId();
              const entry: StreamToolCall = {
                id: callId,
                tool_name: event.tool_name,
                input_summary: inputSummary,
                input_preview: event.input_preview || undefined,
                status: "running",
                started_at: Date.now(),
              };
              collectedTools.push(entry);
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
              // Force immediate render so the "running" state is visible even if
              // tool_end arrives in the same reader.read() chunk (React 18 would
              // otherwise batch both updates and skip straight to "done").
              flushSync(() => {
                setStreamBlocks([...collectedBlocks]);
                setStreamTools((prev) => [...prev, entry]);
              });
            },
            onToolEnd: (event) => {
              if (META_TOOLS.has(event.tool_name)) return;
              const endPatch = {
                status: (event.status === "ok" ? "done" : "error") as "done" | "error",
                artifact_keys: event.artifact_keys ?? [],
                ...(event.output_preview ? { output_preview: event.output_preview } : {}),
              };
              for (let i = collectedTools.length - 1; i >= 0; i--) {
                if (collectedTools[i]!.tool_name === event.tool_name && collectedTools[i]!.status === "running") {
                  collectedTools[i] = { ...collectedTools[i]!, ...endPatch };
                  break;
                }
              }

              for (let i = collectedBlocks.length - 1; i >= 0; i--) {
                const blk = collectedBlocks[i]!;
                if (blk.type === "tool_use" && blk.tool_name === event.tool_name && blk.status === "running") {
                  collectedBlocks[i] = {
                    ...blk,
                    status: event.status === "ok" ? "done" : "error",
                    input_code: blk.input_code,
                    result_summary: event.result_summary || undefined,
                    output_preview: event.output_preview || undefined,
                    artifact_keys: event.artifact_keys,
                  };
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
          selectedSkillIds,
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

      // Assign kind to thinking blocks to match backend's _build_reasoning_steps logic.
      // During streaming, onThinkingEnd pushes thinking blocks without kind. After streaming
      // ends, filterBlocks() defaults kind=undefined to "tool_synthesis", which may differ
      // from the kind the backend assigns (e.g. first block → "planning"). This causes
      // thinking blocks to disappear until page reload restores them via reasoning_steps.
      // Fix: replicate backend's position-based kind assignment before saving the message.
      {
        const thinkingIndices: number[] = [];
        for (let i = 0; i < collectedBlocks.length; i++) {
          const b = collectedBlocks[i]!;
          if (b.type === "thinking" && !b.kind) thinkingIndices.push(i);
        }
        const n = thinkingIndices.length;
        if (n === 1) {
          (collectedBlocks[thinkingIndices[0]!] as import("../lib/backend-types").ThinkingBlock).kind = "final_synthesis";
        } else if (n > 1) {
          for (let pos = 0; pos < n; pos++) {
            const kind: import("../lib/backend-types").ThinkingBlock["kind"] =
              pos === 0 ? "planning" : pos === n - 1 ? "final_synthesis" : "tool_synthesis";
            (collectedBlocks[thinkingIndices[pos]!] as import("../lib/backend-types").ThinkingBlock).kind = kind;
          }
        }
      }

      if (aborted) {
        const partialReasoning = buildStreamingReasoning(collectedReasoning);
        if (streamState.streamedText.trim() || partialReasoning) {
          patchSlotMessages(capturedSessionId, (prev) => [
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
      const fallbackReasoning = buildStreamingReasoning(collectedReasoning);
      if (finalPayload) {
        const finalReasoning = mergeReasoning(finalPayload.reasoning, fallbackReasoning);
        const livePhases = collectedPhases.length > 0 ? [...collectedPhases] : undefined;
        saveLiveReasoningSnapshot(
          capturedSessionId,
          {
            role: "assistant",
            content: finalPayload.text,
            artifacts: finalPayload.artifacts,
          },
          fallbackReasoning,
          livePhases,
          collectedTools.length > 0 ? [...collectedTools] : undefined,
        );
        setSessionData((prev) => {
          const newMap = new Map(prev);
          const slot = newMap.get(capturedSessionId) ?? { messages: [], artifacts: [] };
          newMap.set(capturedSessionId, {
            messages: [
              ...slot.messages,
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
            ],
            artifacts: [...slot.artifacts, ...finalPayload.artifacts],
          });
          return newMap;
        });
        return;
      }

      try {
        const recoveredSession = await getSession(capturedSessionId);
        const recoveredHistory = toChatMessages(
          recoveredSession.session_id,
          recoveredSession.chat_history,
        );
        const lastRecoveredAssistant = [...recoveredHistory]
          .reverse()
          .find((message) => message.role === "assistant");
        if (lastRecoveredAssistant) {
          saveLiveReasoningSnapshot(
            capturedSessionId,
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
          replaceSlot(capturedSessionId, hydratedWithLiveTrace, recoveredSession.artifacts);
          return;
        }
      } catch {
        // Keep local fallback.
      }

      if (streamState.streamedText.trim()) {
        patchSlotMessages(capturedSessionId, (prev) => [
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

      patchSlotMessages(capturedSessionId, (prev) => [
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
    [analysisDepth, appendSlotArtifacts, includeReasoning, isStreaming, patchSlotMessages, replaceSlot, sessionId, useHistory],
  );

  const retryLast = useCallback(async () => {
    if (isStreaming || !sessionId) {
      return;
    }
    const msgs = messagesRef.current;
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
    if (lastUserMsg?.backendId) {
      try {
        await deleteLastMessages(sessionId, lastUserMsg.backendId);
      } catch {
        // Best-effort: continue even if the message wasn't persisted yet.
      }
    }
    // Remove the last user + assistant pair from this session's slot.
    setSessionData((prev) => {
      const newMap = new Map(prev);
      const slot = newMap.get(sessionId);
      if (slot) {
        newMap.set(sessionId, {
          ...slot,
          messages: slot.messages.length >= 2 ? slot.messages.slice(0, -2) : [],
        });
      }
      return newMap;
    });
    await sendQuery(query);
  }, [isStreaming, lastQuery, sendQuery, sessionId]);

  return {
    messages,
    artifacts,
    isStreaming,
    isStreamingCurrentSession,
    backgroundStreamingSessionId,
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
