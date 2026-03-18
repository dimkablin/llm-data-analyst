import { useCallback, useEffect, useState } from "react";
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
import { useChatAgent } from "../hooks/useChatAgent";
import {
  createSession,
  getRuntimeModelProfile,
  getSession,
  listSessions,
  uploadCsv,
} from "../lib/backend-api";
import { exportChatHistory } from "../lib/chat-export";
import type {
  ArtifactPayload,
  RuntimeModelProfile,
  SessionSourceState,
  SessionState,
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

  const {
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
    setErrorMessage,
  } = useChatAgent({
    sessionId,
    includeReasoning: settings.default_include_reasoning,
    useHistory: true,
    analysisDepth: settings.analysis_depth,
  });

  const refreshSessions = useCallback(async () => {
    const rows = await listSessions();
    return rows;
  }, []);

  const applySessionState = useCallback(
    (session: SessionState, nextSessionId: string) => {
      hydrate(session);
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
    [hydrate],
  );

  const loadSession = useCallback(
    async (nextSessionId: string) => {
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
  }, [loadSession, refreshSessions, setErrorMessage, user]);

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
    try {
      await uploadCsv(sessionId, file);
      await loadSession(sessionId);
      await refreshSessions();
    } catch (uploadError) {
      setErrorMessage(summarizeError(uploadError));
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
    <div className="relative h-screen overflow-hidden bg-background font-sans text-foreground">
      <Navigation />

      <div className="absolute inset-x-0 bottom-0 top-16 flex min-h-0 overflow-hidden px-6 pb-6 pt-5">
        <ResizablePanelGroup
          direction="horizontal"
          autoSaveId="workspace-layout-v2"
          className="h-full w-full"
        >
          <ResizablePanel defaultSize={62} minSize={40}>
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="h-full min-h-0 overflow-hidden rounded-[28px] border border-border/40 bg-card/70 p-8 pt-10 shadow-none dark:bg-card/10 dark:shadow-[0_16px_48px_rgba(0,0,0,0.14)]"
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
            className="mx-3 my-2 w-2 bg-transparent after:w-2 after:rounded-full after:bg-border/50 transition-colors hover:after:bg-primary/40 data-[resize-handle-state=drag]:after:bg-primary/70"
          />

          <ResizablePanel defaultSize={38} minSize={26} maxSize={60}>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="relative z-10 flex h-full min-h-0 flex-col overflow-hidden rounded-[28px] border border-border/40 bg-card/75 shadow-none backdrop-blur-3xl dark:bg-card/20 dark:shadow-2xl"
            >
              <ChatPanel
                title={user.username}
                modelLabel={modelLabel}
                messages={messages}
                streamDraft={streamDraft}
                streamReasoning={streamReasoning}
                streamPhases={streamPhases}
                isStreaming={isStreaming}
                error={error}
                canRetry={Boolean(lastQuery)}
                isReady={Boolean(sessionId)}
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
                      sessionTitle={sessionTitle}
                      datasetName={datasetName}
                      settings={settings}
                      modelProfile={modelProfile}
                      onSave={handleSaveSettings}
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
