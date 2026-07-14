import { useCallback, useEffect, useRef, useState } from "react";
import { Navigate } from "react-router";
import { AnimatePresence, motion } from "motion/react";
import { Navigation } from "../components/Navigation";
import { DashboardPanel } from "../components/workspace/DashboardPanel";
import { ChatPanel } from "../components/workspace/ChatPanel";
import { SettingsPanel } from "../components/workspace/SettingsPanel";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "../components/ui/resizable";
import { useAppSession } from "../context/AppSessionContext";
import { useChatAgentContext } from "../context/ChatAgentContext";
import {
  createSession,
  getRuntimeModelProfile,
  getSession,
  listSessions,
  uploadTabularFiles,
} from "../lib/backend-api";
import { exportChatHistory } from "../lib/chat-export";
import type {
  ArtifactPayload,
  ChatMessage,
  RuntimeModelProfile,
  SessionSource,
  SessionSourceState,
  SessionState,
  TabularPreprocessingOptions,
} from "../lib/backend-types";
import {
  buildMessageNoteArtifact,
  mergePinnedIdsForBoard,
  messageNoteArtifactId,
} from "../lib/board-artifacts";
import { summarizeError } from "../lib/format";

const ACTIVE_SESSION_KEY = "llm_new_frontend_active_session";
const PINNED_KEY = "llm_new_frontend_pinned_artifacts";
const USER_PINNED_KEY = "llm_new_frontend_user_pinned_artifacts";
const UNPINNED_KEY = "llm_new_frontend_unpinned_artifacts";
const BOARD_TEXT_ARTIFACTS_KEY = "llm_new_frontend_text_artifacts";
const RESTORED_ARTIFACTS_MESSAGE = "Артефакты восстановлены из сохраненной сессии";

/** Auto-pin only charts; analytical note is pinned when the stream ends. Tables — manually. */
const AUTO_BOARD_ARTIFACT_TYPES = new Set(["plot"]);

function shouldAutoPinToBoard(artifact: ArtifactPayload): boolean {
  return AUTO_BOARD_ARTIFACT_TYPES.has(artifact.type);
}

function isRestoredArtifactsNote(artifact: ArtifactPayload): boolean {
  const content =
    artifact.type === "note" &&
    artifact.data?.format === "markdown" &&
    typeof (artifact.data.data as { content?: unknown })?.content === "string"
      ? String((artifact.data.data as { content?: unknown }).content)
      : "";
  return content.includes(RESTORED_ARTIFACTS_MESSAGE);
}

function loadIdList(storageKey: string, sessionId: string): string[] {
  const raw = window.localStorage.getItem(`${storageKey}_${sessionId}`);
  if (!raw) {
    return [];
  }
  try {
    const parsed = JSON.parse(raw) as string[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function getActiveSessionStorageKey(userId: number | undefined): string {
  return userId ? `${ACTIVE_SESSION_KEY}_${userId}` : ACTIVE_SESSION_KEY;
}

export function Workspace() {
  const { user, settings, setLocalSettings, saveSettings } = useAppSession();
  const [showSettings, setShowSettings] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [sessionTitle, setSessionTitle] = useState("Новый чат");
  const [datasetName, setDatasetName] = useState("");
  const [hasDataset, setHasDataset] = useState(false);
  const [activeSource, setActiveSource] = useState<SessionSourceState>({});
  const [sessionSources, setSessionSources] = useState<SessionSource[]>([]);
  const [pinnedArtifactIds, setPinnedArtifactIds] = useState<string[]>([]);
  const [userPinnedArtifactIds, setUserPinnedArtifactIds] = useState<string[]>([]);
  const [userUnpinnedArtifactIds, setUserUnpinnedArtifactIds] = useState<string[]>([]);
  const [visualizationsFocusBump, setVisualizationsFocusBump] = useState(0);
  const [textArtifacts, setTextArtifacts] = useState<ArtifactPayload[]>([]);
  const autoPinnedArtifactIdsRef = useRef<Set<string>>(new Set());
  const wasStreamingCurrentSessionRef = useRef(false);
  const lastAutoNoteMessageIdRef = useRef<string>("");
  const [modelProfile, setModelProfile] = useState<RuntimeModelProfile | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [chatDraft, setChatDraft] = useState("");

  const {
    bindChatAgent,
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
    contextUsage: liveContextUsage,
    error,
    lastQuery,
    hydrate,
    reset,
    sendQuery,
    retryLast,
    stopStreaming,
    setErrorMessage,
  } = useChatAgentContext();
  const refreshSessions = useCallback(async () => {
    const rows = await listSessions();
    return rows;
  }, []);

  useEffect(() => {
    bindChatAgent({
      sessionId,
      includeReasoning: settings.default_include_reasoning,
      useHistory: true,
      analysisDepth: settings.analysis_depth,
    });
  }, [
    bindChatAgent,
    sessionId,
    settings.analysis_depth,
    settings.default_include_reasoning,
  ]);

  const isStreamingRef = useRef(isStreaming);
  isStreamingRef.current = isStreaming;
  const streamingSessionIdRef = useRef(streamingSessionId);
  streamingSessionIdRef.current = streamingSessionId;

  const applySessionState = useCallback(
    (session: SessionState, nextSessionId: string) => {
      hydrate(session, {
        preserveStreamingForSessionId:
          isStreamingRef.current && streamingSessionIdRef.current === nextSessionId ? nextSessionId : null,
        suppressRestoredArtifactsMessage:
          session.source_type === "openproject" || session.source_mode === "postgres_sync",
      });
      setSessionId(nextSessionId);
      setSessionTitle(session.title || "Новый чат");
      setHasDataset(Boolean(session.has_dataset));
      setActiveSource({
        source_type: session.source_type ?? null,
        source_ref_id: session.source_ref_id ?? null,
        source_label: session.source_label ?? null,
        source_mode: session.source_mode ?? null,
      });
      setDatasetName(String(session.dataset_name || ""));
      setSessionSources(session.sources ?? []);
      window.localStorage.setItem(
        getActiveSessionStorageKey(user?.id),
        nextSessionId,
      );

      autoPinnedArtifactIdsRef.current = new Set();
      wasStreamingCurrentSessionRef.current = false;
      lastAutoNoteMessageIdRef.current = "";
      setPinnedArtifactIds(loadIdList(PINNED_KEY, nextSessionId));
      setUserPinnedArtifactIds(loadIdList(USER_PINNED_KEY, nextSessionId));
      setUserUnpinnedArtifactIds(loadIdList(UNPINNED_KEY, nextSessionId));
    },
    [hydrate, user?.id],
  );

  const loadSession = useCallback(
    async (nextSessionId: string) => {
      // Не вызываем reset() здесь: текущий поток в другой сессии должен сохраниться.
      // Переключение вкладки не должно сбрасывать поток. Вызов hydrate() внутри
      // applySessionState корректно обновляет отображение без остановки фонового потока.
      const session = await getSession(nextSessionId);
      applySessionState(session, nextSessionId);
    },
    [applySessionState],
  );

  useEffect(() => {
    if (!user) {
      return;
    }
    let cancelled = false;
    // Пропускаем сброс при активном потоке: переход туда и обратно
    // (например /workspace -> /sessions -> /workspace) не должен останавливать текущий поток.
    if (!isStreamingRef.current) {
      reset();
    }

    void (async () => {
      try {
        const [rows, model] = await Promise.all([
          refreshSessions(),
          getRuntimeModelProfile(),
        ]);
        if (cancelled) {
          return;
        }
        setModelProfile(model);
        const activeSessionStorageKey = getActiveSessionStorageKey(user.id);
        const storedSessionId =
          window.localStorage.getItem(activeSessionStorageKey) || "";
        const hasStoredSession = rows.some(
          (row) => row.session_id === storedSessionId,
        );
        let initialSessionId = hasStoredSession
          ? storedSessionId
          : rows[0]?.session_id || "";
        if (storedSessionId && !hasStoredSession) {
          window.localStorage.removeItem(activeSessionStorageKey);
        }
        if (!initialSessionId) {
          initialSessionId = await createSession(false);
          if (cancelled) {
            return;
          }
          window.localStorage.setItem(activeSessionStorageKey, initialSessionId);
        }
        await loadSession(initialSessionId);
      } catch (loadError) {
        if (!cancelled) {
          setErrorMessage(summarizeError(loadError));
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [loadSession, refreshSessions, reset, setErrorMessage, user]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }
    window.localStorage.setItem(
      `${PINNED_KEY}_${sessionId}`,
      JSON.stringify(pinnedArtifactIds),
    );
  }, [pinnedArtifactIds, sessionId]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }
    window.localStorage.setItem(
      `${USER_PINNED_KEY}_${sessionId}`,
      JSON.stringify(userPinnedArtifactIds),
    );
  }, [sessionId, userPinnedArtifactIds]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }
    window.localStorage.setItem(
      `${UNPINNED_KEY}_${sessionId}`,
      JSON.stringify(userUnpinnedArtifactIds),
    );
  }, [sessionId, userUnpinnedArtifactIds]);

  useEffect(() => {
    if (!sessionId) {
      autoPinnedArtifactIdsRef.current = new Set();
      return;
    }
    const unpinned = new Set(userUnpinnedArtifactIds);
    const newcomers = artifacts.filter(
      (artifact) =>
        shouldAutoPinToBoard(artifact) &&
        !autoPinnedArtifactIdsRef.current.has(artifact.id) &&
        !unpinned.has(artifact.id),
    );
    if (!newcomers.length) {
      return;
    }
    newcomers.forEach((artifact) => autoPinnedArtifactIdsRef.current.add(artifact.id));
    const newcomerIds = newcomers.map((artifact) => artifact.id);
    const boardArtifacts = [...textArtifacts, ...artifacts];
    setPinnedArtifactIds((prev) => {
      const next = mergePinnedIdsForBoard(
        prev,
        newcomerIds,
        boardArtifacts,
        messages,
        sessionId,
      );
      return next.length === prev.length && next.every((id, index) => id === prev[index]) ? prev : next;
    });
    setVisualizationsFocusBump((value) => value + 1);
  }, [artifacts, messages, sessionId, userUnpinnedArtifactIds]);

  const pinMessageNoteToBoard = useCallback(
    (
      content: string,
      messageId: string,
      timestamp: string,
      auto = false,
      userQuestion = "",
    ) => {
      if (!sessionId) {
        return;
      }
      const textArtifact = buildMessageNoteArtifact(
        sessionId,
        messageId,
        content,
        timestamp,
        auto,
        userQuestion,
      );
      if (!textArtifact) {
        return;
      }
      const noteId = textArtifact.id;
      setTextArtifacts((prev) => {
        const nextText = prev.some((artifact) => artifact.id === noteId)
          ? prev
          : [...prev, textArtifact];
        const boardArtifacts = [
          ...nextText,
          ...artifacts.filter(
            (artifact) => !nextText.some((item) => item.id === artifact.id),
          ),
        ];
        setPinnedArtifactIds((pins) =>
          mergePinnedIdsForBoard(
            pins.filter((id) => id !== noteId),
            [noteId],
            boardArtifacts,
            messages,
            sessionId,
          ),
        );
        return nextText;
      });
      autoPinnedArtifactIdsRef.current.add(noteId);
      setUserUnpinnedArtifactIds((prev) => prev.filter((artifactId) => artifactId !== noteId));
      setVisualizationsFocusBump((value) => value + 1);
    },
    [artifacts, messages, sessionId],
  );

  useEffect(() => {
    const streamJustEnded = wasStreamingCurrentSessionRef.current && !isStreamingCurrentSession;
    wasStreamingCurrentSessionRef.current = isStreamingCurrentSession;

    if (!streamJustEnded || !sessionId) {
      return;
    }
    if (activeSource.source_type === "openproject") {
      return;
    }

    const lastAssistant = [...messages]
      .reverse()
      .find(
        (message): message is ChatMessage =>
          message.role === "assistant" && message.content.trim().length > 0,
      );
    if (!lastAssistant || lastAutoNoteMessageIdRef.current === lastAssistant.id) {
      return;
    }

    const lastUser = [...messages]
      .reverse()
      .find((message) => message.role === "user");
    const userQuestion = lastUser?.content.trim() ?? "";

    const noteId = messageNoteArtifactId(sessionId, lastAssistant.id);
    if (userUnpinnedArtifactIds.includes(noteId)) {
      lastAutoNoteMessageIdRef.current = lastAssistant.id;
      return;
    }

    const plotArtifacts = artifacts.filter((artifact) => shouldAutoPinToBoard(artifact));
    if (plotArtifacts.length > 0) {
      const plotIds = plotArtifacts.map((artifact) => artifact.id);
      plotIds.forEach((id) => autoPinnedArtifactIdsRef.current.add(id));
      const boardArtifacts = [...textArtifacts, ...artifacts];
      setPinnedArtifactIds((prev) =>
        mergePinnedIdsForBoard(prev, plotIds, boardArtifacts, messages, sessionId),
      );
      setVisualizationsFocusBump((value) => value + 1);
    }

    pinMessageNoteToBoard(
      lastAssistant.content,
      lastAssistant.id,
      lastAssistant.timestamp,
      true,
      userQuestion,
    );
    lastAutoNoteMessageIdRef.current = lastAssistant.id;
  }, [
    artifacts,
    activeSource.source_type,
    isStreamingCurrentSession,
    messages,
    pinMessageNoteToBoard,
    sessionId,
    textArtifacts,
    userUnpinnedArtifactIds,
  ]);

  useEffect(() => {
    if (!sessionId) {
      setTextArtifacts([]);
      return;
    }
    const raw = window.localStorage.getItem(`${BOARD_TEXT_ARTIFACTS_KEY}_${sessionId}`);
    if (!raw) {
      setTextArtifacts([]);
      return;
    }
    try {
      const parsed = JSON.parse(raw) as ArtifactPayload[];
      setTextArtifacts(Array.isArray(parsed) ? parsed : []);
    } catch {
      setTextArtifacts([]);
    }
  }, [sessionId]);

  useEffect(() => {
    if (activeSource.source_type !== "openproject" && activeSource.source_mode !== "postgres_sync") {
      return;
    }
    setTextArtifacts((prev) => {
      const next = prev.filter((artifact) => !isRestoredArtifactsNote(artifact));
      return next.length === prev.length ? prev : next;
    });
    setPinnedArtifactIds((prev) => prev.filter((id) => {
      const artifact = textArtifacts.find((item) => item.id === id);
      return !artifact || !isRestoredArtifactsNote(artifact);
    }));
  }, [activeSource.source_mode, activeSource.source_type, textArtifacts]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }
    window.localStorage.setItem(
      `${BOARD_TEXT_ARTIFACTS_KEY}_${sessionId}`,
      JSON.stringify(textArtifacts),
    );
  }, [sessionId, textArtifacts]);

  if (!user) {
    return <Navigate to="/auth" replace />;
  }

  const modelLabel =
    user.is_admin && modelProfile
      ? `${modelProfile.provider} / ${modelProfile.model}`
      : undefined;

  async function handleUpload(
    files: File[],
    preprocessingOptions?: TabularPreprocessingOptions,
  ): Promise<void> {
    if (!sessionId) {
      return;
    }
    if (files.length === 0) {
      return;
    }
    setIsUploading(true);
    try {
      await uploadTabularFiles(sessionId, files, preprocessingOptions);
      await loadSession(sessionId);
      await refreshSessions();
    } catch (uploadError) {
      setErrorMessage(summarizeError(uploadError));
    } finally {
      setIsUploading(false);
    }
  }

  async function handleSaveSettings(
    nextSettings: Partial<typeof settings>,
  ): Promise<void> {
    const updated = await saveSettings(nextSettings);
    setLocalSettings(updated);
  }

  function handlePinArtifact(artifact: ArtifactPayload): void {
    autoPinnedArtifactIdsRef.current.add(artifact.id);
    setUserPinnedArtifactIds((prev) =>
      prev.includes(artifact.id) ? prev : [...prev, artifact.id],
    );
    setUserUnpinnedArtifactIds((prev) => prev.filter((id) => id !== artifact.id));
    const boardArtifacts = [...textArtifacts, ...artifacts];
    setPinnedArtifactIds((prev) =>
      mergePinnedIdsForBoard(
        prev.filter((id) => id !== artifact.id),
        [artifact.id],
        boardArtifacts,
        messages,
        sessionId,
      ),
    );
    setVisualizationsFocusBump((value) => value + 1);
  }

  function handlePinArtifactIds(artifactIds: string[]): void {
    const nextIds = artifactIds.filter((id) => id.trim().length > 0);
    if (nextIds.length === 0) {
      return;
    }
    nextIds.forEach((id) => autoPinnedArtifactIdsRef.current.add(id));
    setUserUnpinnedArtifactIds((prev) => prev.filter((id) => !nextIds.includes(id)));
    setPinnedArtifactIds((prev) => {
      const merged = [...prev];
      nextIds.forEach((id) => {
        if (!merged.includes(id)) {
          merged.push(id);
        }
      });
      return merged;
    });
    setVisualizationsFocusBump((value) => value + 1);
  }

  function handlePinMessageAsArtifact(content: string, messageId: string, timestamp: string): void {
    if (!sessionId) {
      return;
    }
    pinMessageNoteToBoard(content, messageId, timestamp, false);
    const noteId = messageNoteArtifactId(sessionId, messageId);
    setUserPinnedArtifactIds((prev) =>
      prev.includes(noteId) ? prev : [...prev, noteId],
    );
    lastAutoNoteMessageIdRef.current = messageId;
  }

  function handleUnpinArtifact(artifactId: string): void {
    setUserUnpinnedArtifactIds((prev) =>
      prev.includes(artifactId) ? prev : [...prev, artifactId],
    );
    setUserPinnedArtifactIds((prev) => prev.filter((id) => id !== artifactId));
    setPinnedArtifactIds((prev) => prev.filter((id) => id !== artifactId));
    setTextArtifacts((prev) => prev.filter((artifact) => artifact.id !== artifactId));
  }

  function handleExportChat(): void {
    exportChatHistory(sessionId, sessionTitle, datasetName, messages);
  }

return (
    <div className="relative overflow-hidden bg-background font-sans text-foreground" style={{ height: "calc(100vh / var(--ui-zoom, 1))" }}>
      <Navigation />

      <div className="absolute inset-x-0 bottom-0 top-14 flex min-h-0 overflow-hidden pt-3 sm:top-16">
        <ResizablePanelGroup
          direction="horizontal"
          autoSaveId="workspace-layout-v2"
          className="h-full w-full"
        >
          <ResizablePanel defaultSize={62} minSize={40}>
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="h-full min-h-0 overflow-hidden p-3 pt-3 lg:p-5 lg:pt-4 xl:p-8 xl:pt-6"
            >
              <DashboardPanel
                sessionId={sessionId}
                messages={messages}
                artifacts={[...textArtifacts, ...artifacts]}
                pinnedArtifactIds={pinnedArtifactIds}
                userPinnedArtifactIds={userPinnedArtifactIds}
                hiddenArtifactIds={userUnpinnedArtifactIds}
                visualizationsFocusBump={visualizationsFocusBump}
                datasetName={datasetName}
                hasDataset={hasDataset}
                activeSource={activeSource}
                sources={sessionSources}
                showCode
                onUpload={handleUpload}
                onRefreshSession={() => loadSession(sessionId)}
                onPinArtifactIds={handlePinArtifactIds}
                onUnpinArtifact={handleUnpinArtifact}
              />
            </motion.div>
          </ResizablePanel>

          <ResizableHandle className="w-px bg-border/40 transition-colors hover:bg-primary/40 data-[resize-handle-state=drag]:bg-primary/60" />

          <ResizablePanel defaultSize={38} minSize={26} maxSize={60}>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="relative z-10 flex h-full min-h-0 flex-col overflow-hidden border-l border-border/30"
            >
              <ChatPanel
                title={user.username}
                modelLabel={modelLabel}
                datasetName={datasetName}
                messages={messages}
                streamDraft={streamDraft}
                streamReasoning={streamReasoning}
                streamPhases={streamPhases}
                streamTools={streamTools}
                streamBlocks={streamBlocks}
                streamGraph={streamGraph}
                contextUsage={liveContextUsage}
                isContextUsageLoading={false}
                isStreaming={isStreamingCurrentSession}
                isBackgroundStreaming={Boolean(backgroundStreamingSessionId)}
                error={error}
                canRetry={Boolean(lastQuery)}
                isReady={Boolean(sessionId)}
                isUploading={isUploading}
                hasDataset={hasDataset}
                activeSource={activeSource}
                onSubmit={sendQuery}
                onDraftChange={setChatDraft}
                onStop={stopStreaming}
                onRetry={retryLast}
                onSettingsClick={() => setShowSettings((prev) => !prev)}
                onUploadClick={() => {
                  const input = document.querySelector<HTMLInputElement>(
                    'input[type="file"]',
                  );
                  input?.click();
                }}
                onExportChat={handleExportChat}
                onPinArtifact={handlePinArtifact}
                onPinMessage={handlePinMessageAsArtifact}
                settings={settings}
              />

              <AnimatePresence>
                {showSettings ? (
                  <motion.div
                    initial={{ x: "100%" }}
                    animate={{ x: 0 }}
                    exit={{ x: "100%" }}
                    transition={{ type: "spring", damping: 25, stiffness: 200 }}
                    className="absolute inset-0 z-50 border-l border-border/50 bg-background/95 backdrop-blur-xl"
                  >
                    <SettingsPanel
                      onClose={() => setShowSettings(false)}
                      sessionId={sessionId}
                      sessionTitle={sessionTitle}
                      datasetName={datasetName}
                      settings={settings}
                      modelProfile={modelProfile}
                      isAdmin={user.is_admin}
                      onSave={handleSaveSettings}
                      isStreaming={isStreaming}
                    />
                  </motion.div>
                ) : null}
              </AnimatePresence>
            </motion.div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </div>
  );
}
