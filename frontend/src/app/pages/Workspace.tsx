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
  listSkills,
  uploadCsv,
} from "../lib/backend-api";
import { exportChatHistory } from "../lib/chat-export";
import type {
  ArtifactPayload,
  RuntimeModelProfile,
  SessionSourceState,
  SessionState,
  Skill,
} from "../lib/backend-types";
import { summarizeError } from "../lib/format";

const ACTIVE_SESSION_KEY = "llm_new_frontend_active_session";
const PINNED_KEY = "llm_new_frontend_pinned_artifacts";

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
  const [pinnedArtifactIds, setPinnedArtifactIds] = useState<string[]>([]);
  const [modelProfile, setModelProfile] = useState<RuntimeModelProfile | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [availableSkills, setAvailableSkills] = useState<Skill[]>([]);
  const [selectedSkillIds, setSelectedSkillIds] = useState<string[]>([]);

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
    listSkills()
      .then(setAvailableSkills)
      .catch(() => {/* skills unavailable — не блокируем UI */});
  }, []);

  const handleToggleSkill = useCallback((skillId: string) => {
    setSelectedSkillIds((prev) =>
      prev.includes(skillId) ? prev.filter((id) => id !== skillId) : [...prev, skillId],
    );
  }, []);

  useEffect(() => {
    bindChatAgent({
      sessionId,
      includeReasoning: settings.default_include_reasoning,
      useHistory: true,
      analysisDepth: settings.analysis_depth,
      selectedSkillIds,
    });
  }, [
    bindChatAgent,
    sessionId,
    selectedSkillIds,
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
      window.localStorage.setItem(
        getActiveSessionStorageKey(user?.id),
        nextSessionId,
      );

      const rawPinned = window.localStorage.getItem(`${PINNED_KEY}_${nextSessionId}`);
      if (!rawPinned) {
        setPinnedArtifactIds([]);
        return;
      }
      try {
        const parsed = JSON.parse(rawPinned) as string[];
        setPinnedArtifactIds(Array.isArray(parsed) ? parsed : []);
      } catch {
        setPinnedArtifactIds([]);
      }
    },
    [hydrate, user?.id],
  );

  const loadSession = useCallback(
    async (nextSessionId: string) => {
      // Do NOT call reset() here — an ongoing stream in another session must survive
      // the tab switch. The hydrate() call inside applySessionState handles display
      // state correctly without aborting the background stream.
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
    // Skip reset if a stream is currently active — navigating away and back
    // (e.g. /workspace → /sessions → /workspace) must not abort an ongoing stream.
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

  const modelLabel = modelProfile
    ? `${modelProfile.provider} / ${modelProfile.model}`
    : undefined;

  if (!user) {
    return <Navigate to="/auth" replace />;
  }

  async function handleUpload(file: File): Promise<void> {
    if (!sessionId) {
      return;
    }
    setIsUploading(true);
    try {
      await uploadCsv(sessionId, file);
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
    setPinnedArtifactIds((prev) =>
      prev.includes(artifact.id) ? prev : [artifact.id, ...prev],
    );
  }

  function handleExportChat(): void {
    exportChatHistory(sessionId, sessionTitle, datasetName, messages);
  }

return (
    <div className="relative overflow-hidden bg-background font-sans text-foreground" style={{ height: "calc(100vh / var(--ui-zoom, 1))" }}>
      <Navigation />

      <div className="absolute inset-x-0 bottom-0 top-14 flex min-h-0 overflow-hidden px-2 pb-2 pt-2 sm:top-16 lg:px-3 lg:pb-3 lg:pt-3 xl:px-6 xl:pb-6 xl:pt-5">
        <ResizablePanelGroup
          direction="horizontal"
          autoSaveId="workspace-layout-v2"
          className="h-full w-full"
        >
          <ResizablePanel defaultSize={62} minSize={40}>
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="h-full min-h-0 overflow-hidden rounded-2xl border border-border/40 bg-card/70 p-3 pt-4 shadow-none lg:rounded-[28px] lg:p-5 lg:pt-6 xl:p-8 xl:pt-10 dark:bg-card/10 dark:shadow-[0_16px_48px_rgba(0,0,0,0.14)]"
            >
              <DashboardPanel
                sessionId={sessionId}
                artifacts={artifacts}
                pinnedArtifactIds={pinnedArtifactIds}
                datasetName={datasetName}
                hasDataset={hasDataset}
                activeSource={activeSource}
                showCode
                onUpload={handleUpload}
                onRefreshSession={() => loadSession(sessionId)}
                onUnpinArtifact={(artifactId) =>
                  setPinnedArtifactIds((prev) =>
                    prev.filter((id) => id !== artifactId),
                  )
                }
              />
            </motion.div>
          </ResizablePanel>

          <ResizableHandle
            withHandle
            className="mx-1 my-2 w-2 bg-transparent after:w-2 after:rounded-full after:bg-border/50 transition-colors hover:after:bg-primary/40 data-[resize-handle-state=drag]:after:bg-primary/70 lg:mx-3"
          />

          <ResizablePanel defaultSize={38} minSize={26} maxSize={60}>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="relative z-10 flex h-full min-h-0 flex-col overflow-hidden rounded-2xl border border-border/40 bg-card/75 shadow-none backdrop-blur-3xl lg:rounded-[28px] dark:bg-card/20 dark:shadow-2xl"
            >
              <ChatPanel
                title={user.username}
                modelLabel={modelLabel}
                messages={messages}
                streamDraft={streamDraft}
                streamReasoning={streamReasoning}
                streamPhases={streamPhases}
                streamTools={streamTools}
                streamBlocks={streamBlocks}
                streamGraph={streamGraph}
                isStreaming={isStreamingCurrentSession}
                isBackgroundStreaming={Boolean(backgroundStreamingSessionId)}
                error={error}
                canRetry={Boolean(lastQuery)}
                isReady={Boolean(sessionId)}
                isUploading={isUploading}
                hasDataset={hasDataset}
                activeSource={activeSource}
                onSubmit={sendQuery}
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
                availableSkills={availableSkills}
                selectedSkillIds={selectedSkillIds}
                onToggleSkill={handleToggleSkill}
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
