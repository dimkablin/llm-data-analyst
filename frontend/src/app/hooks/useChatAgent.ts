import { useCallback, useEffect, useRef, useState } from "react";

import { getSession, streamQuery } from "../lib/backend-api";
import type {
  ArtifactPayload,
  ChatMessage,
  PhaseEvent,
  QueryResponse,
  SessionState,
} from "../lib/backend-types";

type LiveReasoningSnapshot = {
  fingerprint: string;
  liveReasoningTrace: string | null;
  livePhases: PhaseEvent[];
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
): void {
  if (typeof window === "undefined" || !sessionId) {
    return;
  }
  const fingerprint = buildAssistantMessageFingerprint(message);
  const normalizedTrace = String(liveReasoningTrace ?? "").trim();
  const normalizedPhases = Array.isArray(livePhases) ? livePhases.filter(Boolean) : [];
  if (!fingerprint || (!normalizedTrace && normalizedPhases.length === 0)) {
    return;
  }
  const snapshot: LiveReasoningSnapshot = {
    fingerprint,
    liveReasoningTrace: normalizedTrace || null,
    livePhases: normalizedPhases,
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
  };
  return copy;
}

function toChatMessages(
  sessionId: string,
  history: Array<{
    role: string;
    content: string;
    timestamp: string;
    reasoning?: string | null;
    artifacts?: ArtifactPayload[];
  }>,
): ChatMessage[] {
  const messages = history.map((item, index) => ({
    id: `${item.timestamp}-${index}`,
    timestamp: item.timestamp,
    role: item.role === "user" ? "user" : "assistant",
    content: item.content,
    reasoning: item.reasoning ?? null,
    artifacts: item.artifacts ?? [],
  }));
  return applyLiveReasoningSnapshot(sessionId, messages);
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
  const [error, setError] = useState<string | null>(null);
  const [lastQuery, setLastQuery] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const phaseTokenBufRef = useRef("");
  const phaseFlushRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (phaseFlushRef.current) {
        clearTimeout(phaseFlushRef.current);
      }
    };
  }, []);

  const hydrate = useCallback((
    session: SessionState,
    options?: { preserveStreamingForSessionId?: string | null },
  ) => {
    const preserveStreaming =
      isStreaming &&
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
      setIsStreaming(false);
      setLastQuery(null);
    }
  }, [isStreaming]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const setErrorMessage = useCallback((value: string | null) => {
    setError(value);
  }, []);

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
    if (phaseFlushRef.current) {
      clearTimeout(phaseFlushRef.current);
      phaseFlushRef.current = null;
    }
    phaseTokenBufRef.current = "";
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    if (phaseFlushRef.current) {
      clearTimeout(phaseFlushRef.current);
      phaseFlushRef.current = null;
    }
    phaseTokenBufRef.current = "";
      setMessages([]);
      setArtifacts([]);
      setIsStreaming(false);
      setStreamingSessionId(null);
      setStreamDraft("");
    setStreamReasoning("");
    setStreamPhases([]);
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
      let collectedReasoning = "";
      let liveReasoningStarted = false;
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
              setStreamDraft((prev) => prev + token);
            },
            onFinal: (payload) => {
              streamState.finalPayload = payload;
            },
            onReasoning: (reasoningChunk, mode) => {
              if (!includeReasoning || !reasoningChunk) {
                return;
              }
              if (mode === "token") {
                if (!liveReasoningStarted) {
                  liveReasoningStarted = true;
                  collectedReasoning = collectedReasoning.trim()
                    ? `${collectedReasoning}\n\n### Р”СѓРјР°СЋ\n`
                    : "### Р”СѓРјР°СЋ\n";
                }
                collectedReasoning += reasoningChunk;
                setStreamReasoning((prev) => {
                  let next = prev;
                  if (!liveReasoningStarted) {
                    liveReasoningStarted = true;
                    next = next.trim() ? `${next}\n\n### Думаю\n` : "### Думаю\n";
                  }
                  return next + reasoningChunk;
                });
                return;
              }
              const normalized = reasoningChunk.trim();
              if (!normalized) {
                return;
              }
              collectedReasoning = collectedReasoning
                ? `${collectedReasoning}\n\n${normalized}`
                : normalized;
              setStreamReasoning((prev) => (prev ? `${prev}\n\n${normalized}` : normalized));
            },
            onPhase: (phaseEvent) => {
              if (phaseTokenBufRef.current) {
                phaseTokenBufRef.current = "";
              }
              if (phaseFlushRef.current) {
                clearTimeout(phaseFlushRef.current);
                phaseFlushRef.current = null;
              }
              if (phaseEvent.id) {
                const idx = collectedPhases.findIndex((p) => p.id === phaseEvent.id);
                if (idx >= 0) {
                  collectedPhases[idx] = phaseEvent;
                } else {
                  collectedPhases.push(phaseEvent);
                }
              } else {
                collectedPhases.push(phaseEvent);
              }
              setStreamPhases((prev) => {
                if (phaseEvent.id) {
                  const idx = prev.findIndex((p) => p.id === phaseEvent.id);
                  if (idx >= 0) {
                    const copy = [...prev];
                    copy[idx] = phaseEvent;
                    return copy;
                  }
                }
                return [...prev, phaseEvent];
              });
            },
            onToolStart: (event) => {
              if (!includeReasoning) return;
              const text = `\n🔧 **${event.tool_name}** запущен`;
              collectedReasoning += text;
              setStreamReasoning((prev) => prev + text);
            },
            onToolEnd: (event) => {
              if (!includeReasoning) return;
              const status = event.status === "ok" ? "✅" : "❌";
              const artifacts = event.artifact_keys?.length
                ? ` → ${event.artifact_keys.join(", ")}`
                : "";
              const text = `\n${status} **${event.tool_name}** завершён${artifacts}`;
              collectedReasoning += text;
              setStreamReasoning((prev) => prev + text);
            },
            onPhaseToken: (token) => {
              phaseTokenBufRef.current += token;
              if (!phaseFlushRef.current) {
                phaseFlushRef.current = setTimeout(() => {
                  const buf = phaseTokenBufRef.current;
                  phaseTokenBufRef.current = "";
                  phaseFlushRef.current = null;
                  if (!buf) {
                    return;
                  }
                  setStreamPhases((prev) => {
                    if (prev.length === 0) {
                      return prev;
                    }
                    const copy = [...prev];
                    const last = copy[copy.length - 1];
                    if (last.status === "streaming") {
                      const updated = {
                        ...last,
                        content: last.content + buf,
                      };
                      copy[copy.length - 1] = updated;
                      const collectedIndex = collectedPhases.findIndex(
                        (phase) => phase.id === updated.id,
                      );
                      if (collectedIndex >= 0) {
                        collectedPhases[collectedIndex] = updated;
                      }
                      return copy;
                    }
                    return prev;
                  });
                }, 60);
              }
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
            liveReasoningTrace: fallbackReasoning,
            livePhases,
            metrics: finalPayload.metrics,
            artifacts: finalPayload.artifacts,
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
    if (!lastQuery || isStreaming) {
      return;
    }
    await sendQuery(lastQuery);
  }, [isStreaming, lastQuery, sendQuery]);

  return {
    messages,
    artifacts,
    isStreaming,
    streamingSessionId,
    streamDraft,
    streamReasoning,
    streamPhases,
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
