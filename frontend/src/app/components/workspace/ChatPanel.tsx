import React, { type ReactNode, useEffect, useRef, useState } from "react";
import {
  BookOpen,
  Bot,
  Check,
  Copy,
  Download,
  FileSpreadsheet,
  Globe,
  Pin,
  Plus,
  RefreshCw,
  Send,
  Settings2,
  Square,
  User,
  X,
} from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import type {
  ArtifactPayload,
  AssistantBlock,
  ChatMessage,
  ExecutionGraph,
  PhaseEvent,
  SessionSourceState,
  StreamToolCall,
  UserSettings,
} from "../../lib/backend-types";
import { filterBlocks, filterReasoningSteps } from "../../lib/think-filter";
import { AgentActivityFeed, ToolCallList } from "./AgentActivityFeed";
import { SpinnerDisplay } from "../SpinnerDisplay";
import { BlockTimeline, ThinkingBlock } from "./blocks";
import { formatDurationMs, formatTime } from "../../lib/format";
import { MarkdownBlock } from "../MarkdownBlock";
function Tip({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="group/tip relative inline-flex">
      {children}
      <div className="pointer-events-none absolute bottom-full left-1/2 mb-2 -translate-x-1/2 whitespace-nowrap rounded-md border border-border/50 bg-popover px-2.5 py-1 text-[11px] font-medium text-popover-foreground shadow-md opacity-0 transition-opacity duration-150 group-hover/tip:opacity-100 z-50">
        {label}
        <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-popover" />
      </div>
    </div>
  );
}

const QUICK_SUGGESTIONS = [
  "Покажи ключевые метрики датасета",
  "Построй график по основным колонкам",
  "Найди аномалии и выбросы",
];

type Props = {
  title: string;
  modelLabel?: string;
  messages: ChatMessage[];
  streamDraft: string;
  streamReasoning: string;
  streamPhases: PhaseEvent[];
  streamTools: StreamToolCall[];
  streamBlocks: AssistantBlock[];
  streamGraph?: ExecutionGraph | null;
  isStreaming: boolean;
  isBackgroundStreaming?: boolean;
  error: string | null;
  canRetry: boolean;
  isReady: boolean;
  isUploading: boolean;
  hasDataset: boolean;
  activeSource?: SessionSourceState;
  onSubmit: (value: string) => Promise<void>;
  onStop: () => void;
  onRetry: () => Promise<void>;
  onSettingsClick: () => void;
  onUploadClick: () => void;
  onExportChat: () => void;
  onPinArtifact: (artifact: ArtifactPayload) => void;
  settings: Pick<UserSettings, "show_thinking" | "show_think_planning" | "show_think_tool" | "show_think_final">;
};

export function ChatPanel({
  title,
  modelLabel,
  messages,
  streamDraft,
  streamReasoning,
  streamPhases,
  streamTools,
  streamBlocks,
  streamGraph,
  isStreaming,
  isBackgroundStreaming = false,
  error,
  canRetry,
  isReady,
  isUploading,
  hasDataset,
  activeSource,
  onSubmit,
  onStop,
  onRetry,
  onSettingsClick,
  onUploadClick,
  onExportChat,
  onPinArtifact,
  settings,
}: Props) {
  const [input, setInput] = useState("");
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const shouldStickToBottomRef = useRef(true);

  useEffect(() => {
    if (!shouldStickToBottomRef.current) {
      return;
    }
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamDraft, streamReasoning, streamPhases]);

  useEffect(() => {
    const node = scrollContainerRef.current;
    if (!node) {
      return;
    }
    const handleScroll = () => {
      const distanceFromBottom =
        node.scrollHeight - node.scrollTop - node.clientHeight;
      shouldStickToBottomRef.current = distanceFromBottom < 48;
    };
    handleScroll();
    node.addEventListener("scroll", handleScroll, { passive: true });
    return () => {
      node.removeEventListener("scroll", handleScroll);
    };
  }, []);

  useEffect(() => {
    if (!isMenuOpen) return;
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isMenuOpen]);

  function handleMenuAction(action: "upload" | "search" | "research"): void {
    setIsMenuOpen(false);
    if (action === "upload") {
      onUploadClick();
    } else if (action === "search") {
      setInput((prev) => prev || "Найди в интернете: ");
    } else {
      setInput((prev) => prev || "Глубоко исследуй в интернете: ");
    }
  }

  async function handleSend(): Promise<void> {
    if (!input.trim() || isStreaming || isBackgroundStreaming || isUploading) {
      return;
    }
    const value = input;
    setInput("");
    await onSubmit(value);
  }

  const sourceStatus =
    activeSource?.source_type === "db_connection"
      ? "db connected"
      : hasDataset
        ? "dataset attached"
        : "no dataset";

  return (
    <div className="flex h-full flex-col overflow-hidden bg-card-sunken/30 dark:bg-card/15">
      <div className="flex items-center justify-between border-b border-border/50 p-3 lg:p-5">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary shadow-lg shadow-primary/20">
            <Bot className="h-5 w-5 text-primary-foreground" />
          </div>
          <div>
            <div className="text-[14px] font-bold tracking-tight">{title}</div>
            <div className="flex items-center gap-2 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              <span className={`h-1.5 w-1.5 rounded-full ${isReady ? "bg-emerald-500" : "bg-amber-400"}`}></span>
              <span>{modelLabel || "backend model"}</span>
              <span>{sourceStatus}</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Tip label="Экспорт чата">
            <button
              type="button"
              onClick={onExportChat}
              disabled={messages.length === 0}
              className="rounded-lg p-2 text-muted-foreground transition-all hover:bg-secondary hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Download className="h-5 w-5" />
            </button>
          </Tip>
          <Tip label="Настройки">
            <button
              type="button"
              onClick={onSettingsClick}
              className="rounded-lg p-2 text-muted-foreground transition-all hover:bg-secondary hover:text-foreground"
            >
              <Settings2 className="h-5 w-5" />
            </button>
          </Tip>
        </div>
      </div>

      <div
        ref={scrollContainerRef}
        className="custom-scrollbar flex-1 overflow-x-hidden overflow-y-auto p-3 lg:p-4 xl:p-6"
      >
        <div className="space-y-4 xl:space-y-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center gap-3 py-12 text-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-border/40 bg-secondary/50 text-muted-foreground/60">
                <Bot className="h-6 w-6" />
              </div>
              <div>
                <p className="text-[14px] font-medium text-foreground/70">Аналитик готов к работе</p>
                <p className="mt-1 text-[12px] text-muted-foreground/60">Задайте вопрос о данных или загрузите CSV</p>
              </div>
            </div>
          ) : null}

          {messages.map((message, index) => (
            <MessageBubble
              key={message.id}
              message={message}
              isLast={index === messages.length - 1}
              isStreaming={isStreaming}
              onPinArtifact={onPinArtifact}
              onRegenerate={onRetry}
              settings={settings}
            />
          ))}

          {isStreaming ? (
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex gap-3 lg:gap-4"
            >
              {/* Spinner avatar */}
              <div className="animate-ring-pulse mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary select-none">
                <SpinnerDisplay />
              </div>
              <div className="flex min-w-0 flex-1 flex-col gap-2">
                {/* Block timeline (new) or legacy tool list */}
                {streamBlocks.length > 0 ? (
                  <BlockTimeline
                    blocks={streamBlocks}
                    liveThinking={settings.show_thinking ? streamReasoning : ""}
                    isLive
                  />
                ) : streamTools.length > 0 || (settings.show_thinking && streamReasoning) ? (
                  <ToolCallList tools={streamTools} reasoning={settings.show_thinking ? streamReasoning : undefined} isLive />
                ) : null}

                {/* Streaming answer text */}
                {streamDraft ? (
                  <div className="rounded-2xl rounded-tl-none border border-border/40 bg-card px-4 py-3 text-[13px] leading-relaxed lg:px-5 lg:py-4 lg:text-[14px]">
                    <MarkdownBlock content={streamDraft} />
                  </div>
                ) : !streamBlocks.length && !streamTools.length ? (
                  <div className="rounded-2xl rounded-tl-none border border-border/40 bg-card px-4 py-3 text-[13px] leading-relaxed lg:px-5 lg:py-4 lg:text-[14px]">
                    <span className="text-muted-foreground/50">…</span>
                  </div>
                ) : null}
              </div>
            </motion.div>
          ) : null}

          <div ref={bottomRef} />
        </div>
      </div>

      <div className="border-t border-border/50 bg-background/40 p-3 backdrop-blur-xl lg:p-4 xl:p-6">
        {isBackgroundStreaming ? (
          <div className="mb-4 flex items-center justify-between rounded-2xl border border-amber-500/30 bg-amber-500/10 px-4 py-2.5 text-sm text-amber-300">
            <span>Генерация идёт в другой сессии...</span>
            <button
              type="button"
              onClick={onStop}
              className="ml-4 rounded-lg border border-amber-500/40 px-3 py-1 text-[12px] font-medium text-amber-300 transition-colors hover:bg-amber-500/20"
            >
              Остановить
            </button>
          </div>
        ) : null}
        {error ? (
          <div className="mb-4 rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">{error}</div>
        ) : null}
        {messages.length === 0 && !isStreaming ? (
          <div className="mb-3 flex flex-wrap items-center gap-2">
            {QUICK_SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => setInput(suggestion)}
                className="group flex items-center gap-1.5 rounded-full border border-border/40 bg-card/60 px-3.5 py-1.5 text-[12px] font-medium text-muted-foreground backdrop-blur-sm transition-all duration-150 hover:border-primary/30 hover:bg-primary/5 hover:text-foreground"
              >
                <span className="h-1 w-1 rounded-full bg-primary/40 transition-colors group-hover:bg-primary/70" />
                {suggestion}
              </button>
            ))}
          </div>
        ) : null}


        <div className="group relative rounded-3xl border border-border/50 bg-card/80 backdrop-blur-sm shadow-lg shadow-black/5 transition-all duration-200 focus-within:border-primary/40 focus-within:shadow-xl focus-within:shadow-primary/5 focus-within:ring-4 focus-within:ring-primary/8">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void handleSend();
              }
            }}
            placeholder="Спросите что-нибудь о данных, отчете или метриках..."
            className="min-h-[80px] w-full resize-none bg-transparent p-4 pl-12 pr-14 text-[13.5px] leading-relaxed outline-none placeholder:text-muted-foreground/50 lg:p-5 lg:pl-14 lg:pr-16 lg:text-[15px] xl:min-h-[108px]"
          />

          {/* "+" action menu — bottom left */}
          <div ref={menuRef} className="absolute bottom-3 left-3">
            <button
              type="button"
              onClick={() => setIsMenuOpen((v) => !v)}
              className={`flex h-9 w-9 items-center justify-center rounded-xl border transition-all ${
                isMenuOpen
                  ? "border-primary/60 bg-primary/10 text-primary"
                  : "border-border/50 bg-secondary/60 text-muted-foreground hover:border-border hover:bg-secondary hover:text-foreground"
              }`}
              title="Действия"
            >
              {isMenuOpen ? <X className="h-4 w-4" /> : <Plus className="h-4 w-4" />}
            </button>

            <AnimatePresence>
              {isMenuOpen ? (
                <motion.div
                  initial={{ opacity: 0, y: 6, scale: 0.97 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 6, scale: 0.97 }}
                  transition={{ duration: 0.15 }}
                  className="absolute bottom-full left-0 z-50 mb-2 min-w-[220px] overflow-hidden rounded-2xl border border-border/60 bg-card shadow-2xl"
                >
                  <div className="p-1.5">
                    <ActionMenuItem
                      icon={<FileSpreadsheet className="h-5 w-5" />}
                      label="Загрузить CSV файл"
                      description="Добавить данные для анализа"
                      onClick={() => handleMenuAction("upload")}
                    />
                    <div className="my-1 h-px bg-border/40" />
                    <ActionMenuItem
                      icon={<Globe className="h-5 w-5" />}
                      label="Поиск в сети"
                      description="Быстрый поиск по запросу"
                      onClick={() => handleMenuAction("search")}
                    />
                    <ActionMenuItem
                      icon={<BookOpen className="h-5 w-5" />}
                      label="Глубокое исследование"
                      description="Многоитерационный анализ"
                      onClick={() => handleMenuAction("research")}
                    />
                  </div>
                </motion.div>
              ) : null}
            </AnimatePresence>
          </div>

          {/* Send / Stop — bottom right */}
          <div className="absolute bottom-3 right-3">
            {isStreaming ? (
              <button type="button" onClick={onStop} className="rounded-lg bg-rose-500 p-2 text-white">
                <Square className="h-5 w-5" />
              </button>
            ) : (
              <button
                type="button"
                onClick={() => void handleSend()}
                disabled={!input.trim() || isUploading || isBackgroundStreaming}
                title={isUploading ? "Дождитесь загрузки файла..." : isBackgroundStreaming ? "Дождитесь завершения генерации..." : undefined}
                className={`rounded-lg p-2 transition-all ${
                  input.trim() && !isUploading && !isBackgroundStreaming
                    ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20"
                    : "cursor-not-allowed bg-secondary text-muted-foreground opacity-60"
                }`}
              >
                <Send className="h-5 w-5" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function ActionMenuItem({
  icon,
  label,
  description,
  onClick,
}: {
  icon: ReactNode;
  label: string;
  description: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors hover:bg-secondary/80"
    >
      <span className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl bg-secondary/80 text-foreground">
        {icon}
      </span>
      <div className="min-w-0">
        <div className="text-[13px] font-semibold text-foreground">{label}</div>
        <div className="text-[11px] text-muted-foreground">{description}</div>
      </div>
    </button>
  );
}

function MessageBubble({
  message,
  isLast,
  isStreaming,
  onPinArtifact,
  onRegenerate,
  settings,
}: {
  message: ChatMessage;
  isLast: boolean;
  isStreaming: boolean;
  onPinArtifact: (artifact: ArtifactPayload) => void;
  onRegenerate: () => Promise<void>;
  settings: Pick<UserSettings, "show_thinking" | "show_think_planning" | "show_think_tool" | "show_think_final">;
}) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);

  function handleCopy(): void {
    void navigator.clipboard.writeText(message.content).then(() => {
      setCopied(true);
    });
  }

  useEffect(() => {
    if (!copied) return;
    const timeout = setTimeout(() => setCopied(false), 2000);
    return () => clearTimeout(timeout);
  }, [copied]);

  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className={`group flex gap-2 lg:gap-4 ${isUser ? "flex-row-reverse" : ""}`}>
      <div className={`mt-0.5 flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-lg font-mono text-sm ${isUser ? "bg-secondary text-foreground" : "bg-primary/10 text-primary"}`}>
        {isUser ? <User className="h-4 w-4" /> : "●"}
      </div>

      <div className={`flex min-w-0 max-w-[88%] flex-col gap-1.5 ${isUser ? "items-end" : ""}`}>
        {/* History: BlockTimeline (thinking interleaved with tool calls).
            Filter out thinking blocks when show_thinking is off to match live streaming behaviour. */}
        {!isUser && message.blocks && message.blocks.length > 0 ? (
          <BlockTimeline
            blocks={filterBlocks(message.blocks, settings)}
          />
        ) : (
          <>
            {/* Reload: orphan reasoning steps (final synthesis, no tool call follows).
                Tool-associated steps are already shown via pre_reasoning inside ToolCallList.
                Filter by !tool_name for backward compat with state.json written before this fix. */}
            {!isUser && (() => {
              const filtered = filterReasoningSteps(
                message.reasoning_steps?.filter((s) => !s.tool_name),
                settings,
              );
              return filtered.length > 0
                ? filtered.map((step) => (
                    <ThinkingBlock key={`rs-${step.step_index}`} content={step.content} defaultCollapsed />
                  ))
                : null;
            })()}
            {/* Tool call list — reasoning omitted when reasoning_steps present (avoids CoT duplication) */}
            {!isUser && message.tools?.length ? (
              <ToolCallList
                tools={message.tools}
                reasoning={
                  message.reasoning_steps?.length ? undefined : (message.reasoning ?? undefined)
                }
              />
            ) : null}
            {/* Backward compat: old messages without reasoning_steps and without tools */}
            {!isUser &&
              message.reasoning &&
              !message.reasoning_steps?.length &&
              !message.tools?.length ? (
              <ThinkingBlock content={message.reasoning} defaultCollapsed />
            ) : null}
          </>
        )}

        <div className={`min-w-0 overflow-x-auto rounded-2xl border px-3 py-2.5 text-[13px] leading-relaxed shadow-sm lg:px-4 lg:py-3 lg:text-[14px] xl:px-5 xl:py-4 xl:text-[15px] ${isUser ? "rounded-tr-none border-primary/50 bg-primary text-primary-foreground" : "rounded-tl-none border-border/50 bg-card"}`}>
          <MarkdownBlock content={message.content} className={isUser ? "markdown-invert" : undefined} />
          {message.metrics ? (
            <div className="mt-4 flex flex-wrap gap-2 border-t border-border/20 pt-4 text-[11px] font-bold uppercase tracking-wider text-muted-foreground">
              <span>{formatDurationMs(message.metrics.duration_ms)}</span>
              <span>{message.metrics.model}</span>
              <span>{message.metrics.artifact_count} artifacts</span>
            </div>
          ) : null}
          {message.artifacts?.length ? (
            <div className="mt-4 flex flex-wrap gap-2 border-t border-border/20 pt-4">
              {message.artifacts.map((artifact) => (
                <button
                  key={artifact.id}
                  type="button"
                  onClick={() => onPinArtifact(artifact)}
                  className="inline-flex items-center gap-2 rounded-lg border border-primary/20 bg-primary/10 px-3 py-1.5 text-[12px] font-semibold text-primary transition-all hover:bg-primary/20"
                >
                  <Pin className="h-3.5 w-3.5" />
                  {artifact.text || artifact.type}
                </button>
              ))}
            </div>
          ) : null}
        </div>

        <div className={`flex items-center gap-2 ${isUser ? "flex-row-reverse" : ""}`}>
          <div className="px-1 text-[10px] font-bold uppercase tracking-widest text-muted-foreground/70">
            {formatTime(message.timestamp)}
          </div>
          <div className={`flex items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100 ${isUser ? "flex-row-reverse" : ""}`}>
            <Tip label={copied ? "Скопировано" : "Копировать"}>
              <button
                type="button"
                onClick={handleCopy}
                className="rounded-md p-1 text-muted-foreground/60 transition-colors hover:bg-secondary hover:text-foreground"
              >
                {copied ? <Check className="h-3.5 w-3.5 text-emerald-500" /> : <Copy className="h-3.5 w-3.5" />}
              </button>
            </Tip>
            {isLast && !isUser && !isStreaming ? (
              <Tip label="Повторить генерацию">
                <button
                  type="button"
                  onClick={() => void onRegenerate()}
                  className="rounded-md p-1 text-muted-foreground/60 transition-colors hover:bg-secondary hover:text-foreground"
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                </button>
              </Tip>
            ) : null}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
