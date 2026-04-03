import { type ReactNode, useEffect, useRef, useState } from "react";
import {
  BookOpen,
  Bot,
  Brain,
  Check,
  Copy,
  Download,
  FileSpreadsheet,
  Globe,
  Loader2,
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
  ChatMessage,
  ExecutionGraph,
  PhaseEvent,
  SessionSourceState,
} from "../../lib/backend-types";
import { ExecutionGraphView } from "./ExecutionGraphView";
import { formatDurationMs, formatTime } from "../../lib/format";
import { MarkdownBlock } from "../MarkdownBlock";
import { Tooltip, TooltipContent, TooltipTrigger } from "../ui/tooltip";

const QUICK_SUGGESTIONS = [
  "Покажи ключевые метрики датасета",
  "Построй график по основным колонкам",
  "Найди аномалии и выбросы",
];

const STREAM_REASONING_KEY = "__stream_reasoning__";

type Props = {
  title: string;
  modelLabel?: string;
  messages: ChatMessage[];
  streamDraft: string;
  streamReasoning: string;
  streamPhases: PhaseEvent[];
  streamGraph?: ExecutionGraph | null;
  isStreaming: boolean;
  error: string | null;
  canRetry: boolean;
  isReady: boolean;
  hasDataset: boolean;
  activeSource?: SessionSourceState;
  onSubmit: (value: string) => Promise<void>;
  onStop: () => void;
  onRetry: () => Promise<void>;
  onSettingsClick: () => void;
  onUploadClick: () => void;
  onExportChat: () => void;
  onPinArtifact: (artifact: ArtifactPayload) => void;
};

export function ChatPanel({
  title,
  modelLabel,
  messages,
  streamDraft,
  streamReasoning,
  streamPhases,
  streamGraph,
  isStreaming,
  error,
  canRetry,
  isReady,
  hasDataset,
  activeSource,
  onSubmit,
  onStop,
  onRetry,
  onSettingsClick,
  onUploadClick,
  onExportChat,
  onPinArtifact,
}: Props) {
  const [input, setInput] = useState("");
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [expandedReasoning, setExpandedReasoning] = useState<string | null>(null);
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
    if (!input.trim() || isStreaming) {
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
    <div className="flex h-full flex-col overflow-hidden rounded-[28px] bg-card/40 backdrop-blur-md">
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
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={onExportChat}
                disabled={messages.length === 0}
                className="rounded-lg p-2 text-muted-foreground transition-all hover:bg-secondary hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
              >
                <Download className="h-5 w-5" />
              </button>
            </TooltipTrigger>
            <TooltipContent>Экспорт чата</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={onSettingsClick}
                className="rounded-lg p-2 text-muted-foreground transition-all hover:bg-secondary hover:text-foreground"
              >
                <Settings2 className="h-5 w-5" />
              </button>
            </TooltipTrigger>
            <TooltipContent>Настройки</TooltipContent>
          </Tooltip>
        </div>
      </div>

      <div
        ref={scrollContainerRef}
        className="custom-scrollbar flex-1 overflow-y-auto p-3 lg:p-4 xl:p-6"
      >
        <div className="space-y-4 xl:space-y-6">
          {messages.length === 0 ? (
            <div className="rounded-3xl border border-dashed border-border/40 bg-secondary/20 p-6 text-sm text-muted-foreground">
              Начните диалог с аналитиком. Новый frontend остается основным, а live backend-сценарий уже подключен.
            </div>
          ) : null}

          {messages.map((message, index) => (
            <MessageBubble
              key={message.id}
              message={message}
              isReasoningOpen={expandedReasoning === message.id}
              isLast={index === messages.length - 1}
              isStreaming={isStreaming}
              onToggleReasoning={() =>
                setExpandedReasoning((current) => (current === message.id ? null : message.id))
              }
              onPinArtifact={onPinArtifact}
              onRegenerate={onRetry}
            />
          ))}

          {isStreaming ? (
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-3">
              <div className="flex gap-4">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/20 text-primary">
                  <Loader2 className="h-4 w-4 animate-spin" />
                </div>
                <div className="flex min-w-0 flex-1 flex-col gap-2">
                  {(streamReasoning || streamPhases.length > 0) ? (
                    <button
                      type="button"
                      onClick={() =>
                        setExpandedReasoning((current) =>
                          current === STREAM_REASONING_KEY ? null : STREAM_REASONING_KEY,
                        )
                      }
                      className="flex items-center gap-2 rounded-lg border border-border/30 bg-secondary/50 px-3 py-1.5 text-[11px] font-bold uppercase tracking-wider text-muted-foreground"
                    >
                      <Brain className="h-3.5 w-3.5" />
                      Ход мысли
                    </button>
                  ) : null}
                  {expandedReasoning === STREAM_REASONING_KEY ? (
                    <ReasoningPanel reasoning={streamReasoning} phases={streamPhases} graph={streamGraph} isLive />
                  ) : null}
                  <div className="rounded-2xl rounded-tl-none border border-border/40 bg-secondary/50 px-5 py-4 text-sm leading-relaxed">
                    <MarkdownBlock content={streamDraft || "Формирую ответ..."} />
                  </div>
                </div>
              </div>
            </motion.div>
          ) : null}

          <div ref={bottomRef} />
        </div>
      </div>

      <div className="border-t border-border/50 bg-background/40 p-3 backdrop-blur-xl lg:p-4 xl:p-6">
        {error ? (
          <div className="mb-4 rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">{error}</div>
        ) : null}
        {messages.length === 0 && !isStreaming ? (
          <div className="mb-4 flex flex-wrap items-center gap-2">
            {QUICK_SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => setInput(suggestion)}
                className="rounded-full border border-border/50 bg-secondary px-4 py-1.5 text-[12px] font-medium text-muted-foreground hover:text-foreground"
              >
                {suggestion}
              </button>
            ))}
          </div>
        ) : null}

        <div className="group relative rounded-2xl border border-border/60 bg-card shadow-xl focus-within:border-primary/50 focus-within:ring-4 focus-within:ring-primary/10">
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
            className="min-h-[72px] w-full resize-none bg-transparent p-3 pl-12 pr-14 text-[13px] outline-none lg:p-4 lg:pl-14 lg:pr-16 lg:text-[15px] xl:min-h-[100px]"
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
                disabled={!input.trim()}
                className={`rounded-lg p-2 transition-all ${
                  input.trim()
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

function ReasoningPanel({
  reasoning,
  phases,
  graph,
  isLive = false,
}: {
  reasoning?: string | null;
  phases?: PhaseEvent[];
  graph?: ExecutionGraph | null;
  isLive?: boolean;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const hasReasoning = Boolean(reasoning?.trim());
  const hasPhases = Boolean(phases?.length);
  const hasGraph = Boolean(graph?.nodes?.length);
  useEffect(() => {
    if (!isLive) {
      return;
    }
    const node = scrollRef.current;
    if (!node) {
      return;
    }
    node.scrollTop = node.scrollHeight;
  }, [isLive, phases, reasoning]);

  if (!hasReasoning && !hasPhases && !hasGraph) {
    return null;
  }

  const visiblePhases = phases ?? [];

  return (
    <div className="min-w-0 overflow-hidden rounded-2xl border border-border/30 bg-background/40 p-4 text-xs text-muted-foreground">
      <div className="mb-3 flex items-center gap-2 font-bold uppercase tracking-widest text-primary">
        <Brain className="h-3.5 w-3.5" />
        {isLive ? "Live reasoning" : "Ход мысли"}
      </div>

      {hasGraph ? (
        <div className="mb-3 pb-3 border-b border-border/20">
          <ExecutionGraphView graph={graph!} isLive={isLive} />
        </div>
      ) : null}

      <div className="relative">
        <div
          ref={scrollRef}
          className="custom-scrollbar max-h-[27.5rem] overflow-y-auto pr-1"
        >
          {reasoning ? (
            <MarkdownBlock content={reasoning} className="text-xs" />
          ) : null}
          {visiblePhases.length ? (
            <div className={`${reasoning ? "mt-4 border-t border-border/20 pt-4" : ""} grid gap-2`}>
              {visiblePhases.map((phase, index) => (
                <div
                  key={`${phase.id || phase.timestamp}-${index}`}
                  className="rounded-xl border border-border/30 bg-background/30 px-3 py-3"
                >
                  <div className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                    {phase.title}
                  </div>
                  <div className="mt-2">
                    <MarkdownBlock
                      content={phase.content || (isLive ? "_Шаг выполняется..._" : "_Шаг выполнен._")}
                      className="text-xs"
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function MessageBubble({
  message,
  isReasoningOpen,
  isLast,
  isStreaming,
  onToggleReasoning,
  onPinArtifact,
  onRegenerate,
}: {
  message: ChatMessage;
  isReasoningOpen: boolean;
  isLast: boolean;
  isStreaming: boolean;
  onToggleReasoning: () => void;
  onPinArtifact: (artifact: ArtifactPayload) => void;
  onRegenerate: () => Promise<void>;
}) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);
  const [isFullReasoningOpen, setIsFullReasoningOpen] = useState(true);

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
  const hasFullTrace =
    Boolean(message.liveReasoningTrace?.trim()) ||
    Boolean(message.livePhases?.length);
  const displayedReasoning =
    !isFullReasoningOpen || !message.liveReasoningTrace?.trim()
      ? message.reasoning
      : message.liveReasoningTrace;
  const displayedPhases =
    !isFullReasoningOpen || !message.livePhases?.length
      ? message.phases
      : message.livePhases;

  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className={`group flex gap-2 lg:gap-4 ${isUser ? "flex-row-reverse" : ""}`}>
      <div className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg ${isUser ? "bg-secondary" : "bg-primary/20 text-primary"}`}>
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>

      <div className={`flex min-w-0 max-w-[85%] flex-col gap-2 ${isUser ? "items-end" : ""}`}>
        {!isUser && (displayedReasoning || displayedPhases?.length || hasFullTrace) ? (
          <button
            type="button"
            onClick={onToggleReasoning}
            className="flex items-center gap-2 rounded-lg border border-border/30 bg-secondary/50 px-3 py-1.5 text-[11px] font-bold uppercase tracking-wider text-muted-foreground"
          >
            <Brain className="h-3.5 w-3.5" />
            Ход мысли
          </button>
        ) : null}

        {isReasoningOpen && (displayedReasoning || displayedPhases?.length || hasFullTrace) ? (
          <div className="min-w-0">
            {hasFullTrace ? (
              <div className="mb-2 flex justify-end">
                <button
                  type="button"
                  onClick={() => setIsFullReasoningOpen((current) => !current)}
                  className="inline-flex h-6 w-6 items-center justify-center rounded-md border border-border/40 bg-background/40 text-sm font-bold text-foreground/80 transition-colors hover:bg-secondary"
                  title={isFullReasoningOpen ? "Показать краткую версию" : "Показать полную live-версию"}
                  aria-label={isFullReasoningOpen ? "Показать краткую версию" : "Показать полную live-версию"}
                >
                  {isFullReasoningOpen ? "−" : "+"}
                </button>
              </div>
            ) : null}
            <ReasoningPanel reasoning={displayedReasoning} phases={displayedPhases} graph={message.executionGraph} />
          </div>
        ) : null}

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
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  onClick={handleCopy}
                  className="rounded-md p-1 text-muted-foreground/60 transition-colors hover:bg-secondary hover:text-foreground"
                >
                  {copied ? <Check className="h-3.5 w-3.5 text-emerald-500" /> : <Copy className="h-3.5 w-3.5" />}
                </button>
              </TooltipTrigger>
              <TooltipContent>{copied ? "Скопировано" : "Копировать"}</TooltipContent>
            </Tooltip>
            {isLast && !isUser && !isStreaming ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={() => void onRegenerate()}
                    className="rounded-md p-1 text-muted-foreground/60 transition-colors hover:bg-secondary hover:text-foreground"
                  >
                    <RefreshCw className="h-3.5 w-3.5" />
                  </button>
                </TooltipTrigger>
                <TooltipContent>Повторить генерацию</TooltipContent>
              </Tooltip>
            ) : null}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
