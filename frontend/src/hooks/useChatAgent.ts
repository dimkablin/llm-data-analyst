import { useCallback, useEffect, useRef, useState } from "react";

import { getSession, streamQuery } from "../api";
import type { ArtifactPayload, ChatMessage, PhaseEvent, QueryResponse, SessionState } from "../types";

function toChatMessages(
  history: Array<{
    role: string;
    content: string;
    timestamp: string;
    reasoning?: string | null;
    artifacts?: ArtifactPayload[];
  }>
): ChatMessage[] {
  return history.map((item, index) => ({
    id: `${item.timestamp}-${index}`,
    role: item.role === "user" ? "user" : "assistant",
    content: item.content,
    reasoning: item.reasoning ?? null,
    artifacts: item.artifacts ?? []
  }));
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
  streamDraft: string;
  streamReasoning: string;
  streamPhases: PhaseEvent[];
  error: string | null;
  lastQuery: string | null;
  hydrate: (session: SessionState) => void;
  sendQuery: (query: string) => Promise<void>;
  retryLast: () => Promise<void>;
  stopStreaming: () => void;
  reset: () => void;
  clearError: () => void;
  setErrorMessage: (value: string | null) => void;
};

export function useChatAgent({ sessionId, includeReasoning, useHistory, analysisDepth }: UseChatAgentArgs): UseChatAgentResult {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [artifacts, setArtifacts] = useState<ArtifactPayload[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
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
      if (phaseFlushRef.current) {
        clearTimeout(phaseFlushRef.current);
      }
    };
  }, []);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (phaseFlushRef.current) clearTimeout(phaseFlushRef.current);
    };
  }, []);

  const hydrate = useCallback((session: SessionState) => {
    const hydratedMessages = toChatMessages(session.chat_history);
    const hasArtifactMessages = hydratedMessages.some((item) => (item.artifacts?.length ?? 0) > 0);
    if (!hasArtifactMessages && session.artifacts.length > 0) {
      hydratedMessages.push({
        id: `restored-${Date.now()}`,
        role: "assistant",
        content: "Восстановлены артефакты предыдущей сессии.",
        artifacts: session.artifacts
      });
    }
    setMessages(hydratedMessages);
    setArtifacts(session.artifacts);
    setError(null);
    setStreamDraft("");
    setStreamReasoning("");
    setStreamPhases([]);
    setIsStreaming(false);
    setLastQuery(null);
  }, []);

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
      setMessages((prev) => [
        ...prev,
        {
          id: `u-${Date.now()}`,
          role: "user",
          content: prompt
        }
      ]);

      const streamState: { finalPayload?: QueryResponse; streamedText: string } = {
        streamedText: ""
      };
      const collectedPhases: PhaseEvent[] = [];
      let liveReasoningStarted = false;
      let aborted = false;
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await streamQuery(sessionId, prompt, includeReasoning, useHistory, {
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
                      copy[copy.length - 1] = {
                        ...last,
                        content: last.content + buf,
                      };
                      return copy;
                    }
                    return prev;
                  });
                }, 60);
              }
            },
            onError: (streamError) => {
              setError(streamError);
            }
          }, controller.signal, analysisDepth);
      } catch (err) {
        if ((err as Error)?.name === "AbortError") {
          aborted = true;
        } else {
          setError(String(err));
        }
      } finally {
        abortRef.current = null;
        setIsStreaming(false);
        setStreamDraft("");
        setStreamReasoning("");
        setStreamPhases([]);
      }

      if (aborted) return;

      const finalPayload = streamState.finalPayload;
      const savedPhases = collectedPhases.filter((p) => p.status === "done");
      if (finalPayload) {
        setMessages((prev) => [
          ...prev,
          {
            id: `a-${Date.now()}`,
            role: "assistant",
            content: finalPayload.text,
            reasoning: finalPayload.reasoning ?? null,
            phases: savedPhases.length > 0 ? savedPhases : undefined,
            metrics: finalPayload.metrics,
            artifacts: finalPayload.artifacts
          }
        ]);
        setArtifacts((prev) => [...prev, ...finalPayload.artifacts]);
        return;
      }

      try {
        const recoveredSession = await getSession(sessionId);
        const recoveredHistory = toChatMessages(recoveredSession.chat_history);
        if (recoveredHistory.length > 0) {
          setMessages(recoveredHistory);
          setArtifacts(recoveredSession.artifacts);
          return;
        }
      } catch {
        // Keep local fallback flow below if the reconciliation request fails.
      }

      if (streamState.streamedText.trim()) {
        setMessages((prev) => [
          ...prev,
          {
            id: `a-${Date.now()}`,
            role: "assistant",
            content: streamState.streamedText,
            phases: savedPhases.length > 0 ? savedPhases : undefined
          }
        ]);
        return;
      }

      setMessages((prev) => [
        ...prev,
        {
          id: `a-fallback-${Date.now()}`,
          role: "assistant",
          content:
            "Ответ сформирован на backend, но не доставлен в потоковом канале. Обновите чат или повторите запрос."
        }
      ]);
    },
    [includeReasoning, isStreaming, sessionId, useHistory, analysisDepth]
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
    setErrorMessage
  };
}
