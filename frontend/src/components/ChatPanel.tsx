import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

import type { ArtifactPayload, ChatMessage, PhaseEvent } from "../types";
import { ArtifactCard } from "./ArtifactCard";
import { MarkdownBlock } from "./MarkdownBlock";

type ChatPanelProps = {
  sessionId: string;
  includeReasoning: boolean;
  showCode: boolean;
  themeMode: "light" | "dark";
  messages: ChatMessage[];
  pinnedArtifactIds: Set<string>;
  onPinArtifact: (artifact: ArtifactPayload) => void;
  streamDraft: string;
  streamReasoning: string;
  streamPhases: PhaseEvent[];
  isStreaming: boolean;
  isReady: boolean;
  error: string | null;
  onClearError: () => void;
  onSubmit: (query: string) => Promise<void>;
  onStop: () => void;
  onRetry: () => Promise<void>;
  canRetry: boolean;
};

type ChatTurn = {
  id: string;
  question: ChatMessage | null;
  answers: ChatMessage[];
};

type ReasoningSummary = {
  route?: string;
  toolCalls?: number;
  durationMs?: number;
  toolEvents: number;
  liveToolEvents: number;
};

type ActivityBlock = {
  id: string;
  title: string;
  content: string;
};

type ActivityEntry = {
  id: string;
  title: string;
  meta: string;
  kind: "saved" | "live";
  blocks: ActivityBlock[];
};

function buildTurns(messages: ChatMessage[]): ChatTurn[] {
  const turns: ChatTurn[] = [];
  let active: ChatTurn | null = null;

  messages.forEach((message) => {
    if (message.role === "user") {
      active = {
        id: `turn-${message.id}`,
        question: message,
        answers: []
      };
      turns.push(active);
      return;
    }

    if (!active) {
      active = {
        id: `turn-${message.id}`,
        question: null,
        answers: [message]
      };
      turns.push(active);
      return;
    }

    active.answers.push(message);
  });

  return turns;
}

function splitReasoningBlocks(content: string): ActivityBlock[] {
  const normalized = String(content || "").replace(/\r\n/g, "\n").trim();
  if (!normalized) {
    return [];
  }

  const chunks = normalized.split(/\n(?=###\s+)/g);
  const blocks: ActivityBlock[] = [];

  chunks.forEach((chunk, index) => {
    const lines = chunk.split("\n");
    const headingMatch = lines[0]?.match(/^###\s+(.+)$/);
    const title = headingMatch ? headingMatch[1].trim() : index === 0 ? "Думаю" : `Шаг ${index + 1}`;
    const body = headingMatch ? lines.slice(1).join("\n").trim() : chunk.trim();
    blocks.push({
      id: `block-${index}`,
      title: title || `Шаг ${index + 1}`,
      content: body || "_Без дополнительного текста_"
    });
  });

  return blocks;
}

function parseReasoningSummary(text: string | null | undefined): ReasoningSummary {
  const source = String(text || "");
  const route = source.match(/- Route:\s*`([^`]+)`/i)?.[1]?.trim();
  const toolCallsRaw = source.match(/- Tool calls:\s*`(\d+)`/i)?.[1];
  const durationRaw = source.match(/- Duration:\s*`(\d+)\s*ms`/i)?.[1];
  const toolEvents = (source.match(/^\d+\.\s+`/gm) || []).length;
  const liveToolEvents = (source.match(/Live Tool #\d+/gi) || []).length;

  return {
    route: route || undefined,
    toolCalls: toolCallsRaw ? Number(toolCallsRaw) : undefined,
    durationMs: durationRaw ? Number(durationRaw) : undefined,
    toolEvents,
    liveToolEvents
  };
}

const PHASE_DISPLAY: Record<string, { icon: string; label: string }> = {
  think:    { icon: "🧠", label: "Рассуждение" },
  act:      { icon: "⚙️", label: "Действие" },
  evaluate: { icon: "✅", label: "Оценка" },
  finalize: { icon: "📄", label: "Финализация" },
};

function buildPhaseBlocks(phases: PhaseEvent[]): ActivityBlock[] {
  const seen = new Set<string>();
  const deduped: PhaseEvent[] = [];
  for (let i = phases.length - 1; i >= 0; i--) {
    const key = phases[i].id || `idx-${i}`;
    if (!seen.has(key)) {
      seen.add(key);
      deduped.unshift(phases[i]);
    }
  }
  return deduped.map((phase, index) => {
    const display = PHASE_DISPLAY[phase.phase] ?? { icon: "ℹ️", label: phase.phase };
    const isStreaming = phase.status === "streaming";
    const statusSuffix = isStreaming ? " ▍" : phase.status === "done" ? "" : phase.status ? ` [${phase.status}]` : "";
    const content = phase.content || (isStreaming ? "..." : "_Без дополнительного текста_");
    return {
      id: phase.id || `phase-${index}`,
      title: `${display.icon} ${phase.title || display.label}${statusSuffix}`,
      content,
    };
  });
}

function buildReasoningMeta(
  summary: ReasoningSummary,
  mode: "saved" | "live"
): string {
  const parts: string[] = [];
  if (summary.route) {
    parts.push(summary.route);
  }
  if (typeof summary.toolCalls === "number" && Number.isFinite(summary.toolCalls)) {
    parts.push(`${summary.toolCalls} инструментов`);
  }
  const totalEvents = summary.toolEvents + summary.liveToolEvents;
  if (totalEvents > 0) {
    parts.push(`${totalEvents} событий`);
  }
  if (typeof summary.durationMs === "number" && Number.isFinite(summary.durationMs)) {
    parts.push(`${Math.max(1, Math.round(summary.durationMs / 1000))}s`);
  }
  if (parts.length === 0) {
    return mode === "live" ? "пошаговый live trace" : "детали шага";
  }
  return parts.join(" · ");
}

export function ChatPanel({
  sessionId,
  includeReasoning,
  showCode,
  themeMode,
  messages,
  pinnedArtifactIds,
  onPinArtifact,
  streamDraft,
  streamReasoning,
  streamPhases,
  isStreaming,
  isReady,
  error,
  onClearError,
  onSubmit,
  onStop,
  onRetry,
  canRetry
}: ChatPanelProps): JSX.Element {
  const [query, setQuery] = useState("");
  const [isActivityOpen, setIsActivityOpen] = useState(false);
  const [activeMessageId, setActiveMessageId] = useState<string | null>(null);
  const messagesRef = useRef<HTMLDivElement | null>(null);
  const activityBodyRef = useRef<HTMLDivElement | null>(null);
  const hasContent = useMemo(() => messages.length > 0 || Boolean(streamDraft), [messages.length, streamDraft]);
  const turns = useMemo(() => buildTurns(messages), [messages]);
  const hasLivePhases = isStreaming && streamPhases.length > 0;
  const hasLiveActivity = isStreaming && (streamPhases.length > 0 || streamReasoning.trim().length > 0);

  const activityEntries = useMemo(() => {
    if (activeMessageId === "live") {
      if (streamPhases.length > 0) {
        return [{
          id: "live-phases",
          title: "ReAct цикл",
          meta: `${streamPhases.length} ${streamPhases.length === 1 ? "фаза" : streamPhases.length >= 2 && streamPhases.length <= 4 ? "фазы" : "фаз"}`,
          kind: "live" as const,
          blocks: buildPhaseBlocks(streamPhases)
        }];
      }
      if (streamReasoning.trim()) {
        const liveSummary = parseReasoningSummary(streamReasoning);
        return [{
          id: "live-reasoning",
          title: "Рассуждение",
          meta: buildReasoningMeta(liveSummary, "live"),
          kind: "live" as const,
          blocks: splitReasoningBlocks(streamReasoning)
        }];
      }
      return [];
    }

    const msg = messages.find((m) => m.id === activeMessageId);
    if (!msg || msg.role !== "assistant") return [];

    const entries: ActivityEntry[] = [];
    if (msg.phases && msg.phases.length > 0) {
      entries.push({
        id: `phases-${msg.id}`,
        title: "ReAct цикл",
        meta: `${msg.phases.length} ${msg.phases.length === 1 ? "фаза" : msg.phases.length >= 2 && msg.phases.length <= 4 ? "фазы" : "фаз"}`,
        kind: "saved",
        blocks: buildPhaseBlocks(msg.phases)
      });
    }
    if (msg.reasoning?.trim()) {
      const summary = parseReasoningSummary(msg.reasoning);
      entries.push({
        id: `reasoning-${msg.id}`,
        title: "Рассуждение",
        meta: buildReasoningMeta(summary, "saved"),
        kind: "saved",
        blocks: splitReasoningBlocks(msg.reasoning)
      });
    }
    return entries;
  }, [activeMessageId, messages, streamPhases, streamReasoning]);
  const activityCount = activityEntries.length;

  useEffect(() => {
    if (!hasContent) {
      return;
    }
    const element = messagesRef.current;
    if (!element) {
      return;
    }
    element.scrollTo({ top: element.scrollHeight, behavior: "smooth" });
  }, [hasContent, messages, streamDraft]);

  useEffect(() => {
    if (!isStreaming && activeMessageId === "live") {
      setIsActivityOpen(false);
      setActiveMessageId(null);
    }
  }, [isStreaming, activeMessageId]);

  const hasAutoOpenedRef = useRef(false);

  useEffect(() => {
    if (hasLiveActivity && !hasAutoOpenedRef.current) {
      hasAutoOpenedRef.current = true;
      setActiveMessageId("live");
      setIsActivityOpen(true);
    }
    if (!isStreaming) {
      hasAutoOpenedRef.current = false;
    }
  }, [hasLiveActivity, isStreaming]);

  useEffect(() => {
    if (isActivityOpen && activityBodyRef.current && activeMessageId === "live") {
      activityBodyRef.current.scrollTo({ top: activityBodyRef.current.scrollHeight, behavior: "smooth" });
    }
  }, [isActivityOpen, activeMessageId, streamPhases, streamReasoning]);

  useEffect(() => {
    if (!isActivityOpen) {
      return;
    }
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsActivityOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [isActivityOpen]);

  function openMessageActivity(messageId: string): void {
    setActiveMessageId(messageId);
    setIsActivityOpen(true);
  }

  async function handleSubmit(event: FormEvent): Promise<void> {
    event.preventDefault();
    await submitCurrentQuery();
  }

  async function submitCurrentQuery(): Promise<void> {
    const prompt = query.trim();
    if (!prompt || isStreaming || !isReady) {
      return;
    }
    setQuery("");
    await onSubmit(prompt);
  }

  async function handleTextareaKeyDown(event: KeyboardEvent<HTMLTextAreaElement>): Promise<void> {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) {
      return;
    }
    event.preventDefault();
    await submitCurrentQuery();
  }

  return (
    <section className="panel panel-chat panel-chat-compact">
      <div className="panel-head">
        <h2>Чат</h2>
        <div className="chat-head-tools">
          {includeReasoning ? (
            <button
              type="button"
              className={`btn-ghost btn-xs activity-open-btn${isActivityOpen ? " active" : ""}`}
              onClick={() => {
                if (isActivityOpen) {
                  setIsActivityOpen(false);
                } else {
                  const lastWithActivity = [...messages].reverse().find(
                    (m) => m.role === "assistant" && (m.phases?.length || m.reasoning?.trim())
                  );
                  if (lastWithActivity) {
                    openMessageActivity(lastWithActivity.id);
                  } else if (hasLiveActivity) {
                    openMessageActivity("live");
                  }
                }
              }}
            >
              Активность
            </button>
          ) : null}
          <span className="session-id">{sessionId ? `сессия: ${sessionId.slice(0, 10)}...` : "сессия: init"}</span>
        </div>
      </div>

      <div ref={messagesRef} className="messages">
        {turns.map((turn, turnIdx) => (
          <section key={turn.id} className="chat-turn">
            <div className="chat-turn-head">
              <span className="chat-turn-label">Запрос {turnIdx + 1}</span>
            </div>

            {turn.question ? (
              <article
                key={turn.question.id}
                className="msg msg-user msg-question"
                style={{ animationDelay: `${turnIdx * 30}ms` }}
              >
                <div className="msg-role msg-role-user">Вопрос</div>
                <MarkdownBlock content={turn.question.content} />
              </article>
            ) : null}

            {turn.answers.map((message, answerIdx) => {
              const reasoningSummary = parseReasoningSummary(message.reasoning);
              const reasoningMeta = buildReasoningMeta(reasoningSummary, "saved");

              return (
                <article
                  key={message.id}
                  className="msg msg-assistant msg-answer"
                  style={{ animationDelay: `${(turnIdx + answerIdx + 1) * 35}ms` }}
                >
                  <div className="msg-role msg-role-assistant">Ответ агента</div>
                  <MarkdownBlock content={message.content} />
                  {(message.phases?.length || (message.reasoning && includeReasoning)) ? (
                    <div className="msg-tools">
                      <button type="button" className="btn-ghost btn-xs" onClick={() => openMessageActivity(message.id)}>
                        Активность
                      </button>
                      {message.phases?.length ? (
                        <span className="msg-tools-meta">
                          {message.phases.length} {message.phases.length === 1 ? "фаза" : message.phases.length >= 2 && message.phases.length <= 4 ? "фазы" : "фаз"}
                        </span>
                      ) : (
                        <span className="msg-tools-meta">{reasoningMeta}</span>
                      )}
                    </div>
                  ) : null}
                  {message.metrics ? (
                    <div className="msg-metrics">
                      <span>{message.metrics.duration_ms} ms</span>
                      <span>{message.metrics.artifact_count} артефактов</span>
                      <span>{message.metrics.model}</span>
                    </div>
                  ) : null}
                  {message.artifacts && message.artifacts.length > 0 ? (
                    <div className="inline-artifacts">
                      {message.artifacts.map((artifact) => (
                        <ArtifactCard
                          key={artifact.id}
                          artifact={artifact}
                          showCode={showCode}
                          themeMode={themeMode}
                          actionLabel={pinnedArtifactIds.has(artifact.id) ? "В дашборде" : "Добавить в дашборд"}
                          actionDisabled={pinnedArtifactIds.has(artifact.id)}
                          onAction={onPinArtifact}
                        />
                      ))}
                    </div>
                  ) : null}
                </article>
              );
            })}
          </section>
        ))}
        {isStreaming ? (
          <section className="chat-turn chat-turn-live">
            <div className="chat-turn-head">
              <span className="chat-turn-label">Текущий ответ</span>
            </div>
            <article className="msg msg-assistant msg-answer msg-streaming">
              <div className="msg-role msg-role-assistant">Ответ агента</div>
              <MarkdownBlock content={streamDraft || "Генерирую ответ..."} />
              {hasLiveActivity ? (
                <div className="msg-tools">
                  <button type="button" className="btn-ghost btn-xs" onClick={() => openMessageActivity("live")}>
                    Смотреть активность
                  </button>
                  {hasLivePhases ? (
                    <span className="msg-tools-meta">
                      {streamPhases.length} {streamPhases.length === 1 ? "фаза" : streamPhases.length >= 2 && streamPhases.length <= 4 ? "фазы" : "фаз"}
                    </span>
                  ) : null}
                </div>
              ) : null}
            </article>
          </section>
        ) : null}
      </div>

      <form className="query-form" onSubmit={(event) => void handleSubmit(event)}>
        <textarea
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => void handleTextareaKeyDown(event)}
          placeholder="Например: какие 3 фактора сильнее всего влияют на churn?"
          rows={3}
        />
        <div className="query-actions">
          <div className="query-buttons">
            <button type="button" className="btn-ghost" disabled={!isStreaming} onClick={onStop}>
              Стоп
            </button>
            <button type="button" className="btn-ghost" disabled={isStreaming || !canRetry} onClick={() => void onRetry()}>
              Повторить
            </button>
            <button type="submit" disabled={!isReady || isStreaming || !query.trim()}>
              {isStreaming ? "Выполняется..." : "Отправить запрос"}
            </button>
          </div>
        </div>
      </form>
      {error ? (
        <div className="alert-error">
          {error}
          <button type="button" className="btn-inline" onClick={onClearError}>
            скрыть
          </button>
        </div>
      ) : null}

      {activeMessageId ? (
        <>
          <button
            type="button"
            className={`activity-overlay${isActivityOpen ? " open" : ""}`}
            aria-label="Закрыть панель активности"
            onClick={() => setIsActivityOpen(false)}
          />
          <aside className={`activity-drawer${isActivityOpen ? " open" : ""}`} aria-label="Активность сообщения">
            <div className="activity-drawer-head">
              <div className="activity-title-wrap">
                <h3>{activeMessageId === "live" ? "Активность (текущий запрос)" : "Активность"}</h3>
                {activityCount > 0 ? <span className="activity-count-chip">{activityCount}</span> : null}
              </div>
              <button
                type="button"
                className="btn-ghost btn-xs activity-close-btn"
                onClick={() => setIsActivityOpen(false)}
              >
                Закрыть
              </button>
            </div>
            <div className="activity-drawer-body" ref={activityBodyRef}>
              {activityEntries.length === 0 ? (
                <p className="activity-empty">
                  {activeMessageId === "live" ? "Ожидание фаз агента..." : "Нет данных об активности для этого сообщения."}
                </p>
              ) : (
                <div className="activity-list">
                  {activityEntries.map((entry) => (
                    <article key={entry.id} className="activity-entry is-focused">
                      <header>
                        <strong>{entry.title}</strong>
                        <span>{entry.meta}</span>
                      </header>
                      <div className="activity-entry-blocks">
                        {entry.blocks.map((block) => (
                          <section key={`${entry.id}-${block.id}`} className={`activity-block activity-block-${entry.kind}`}>
                            <h4>{block.title}</h4>
                            <MarkdownBlock content={block.content} />
                          </section>
                        ))}
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </div>
          </aside>
        </>
      ) : null}
    </section>
  );
}
