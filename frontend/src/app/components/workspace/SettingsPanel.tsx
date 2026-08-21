import { useEffect, useRef, useState, type ReactNode } from "react";
import { BookOpen, Brain, Cpu, Database, Eye, HelpCircle, Info, ListTodo, Loader2, Radio, RefreshCw, Settings, Sliders, X } from "lucide-react";
import { Switch } from "../ui/switch";
import { getSession, getSessionNotebookCells, type NotebookCell } from "../../lib/backend-api";
import {
  ANALYSIS_DEPTH_STEP_CEILING,
  clampAgentMaxStepsForDepth,
  type AnalysisDepth,
  type RuntimeModelProfile,
  type SemanticCatalogStatusResponse,
  type UserSettings,
} from "../../lib/backend-types";
import { MarkdownBlock } from "../MarkdownBlock";
import { SemanticCatalogBlock } from "./SemanticCatalogBlock";

const BUILD_COMMIT =
  (((import.meta as unknown as { env?: Record<string, string | undefined> }).env?.VITE_BUILD_COMMIT ?? "").trim() ||
    "unknown").slice(0, 10);

type Props = {
  onClose: () => void;
  sessionId: string;
  sessionTitle: string;
  datasetName: string;
  settings: UserSettings;
  modelProfile: RuntimeModelProfile | null;
  isAdmin?: boolean;
  onSave: (payload: Partial<UserSettings>) => Promise<void>;
  isStreaming?: boolean;
  semanticConnectionId?: string | null;
  semanticState: SemanticCatalogStatusResponse;
  onSemanticStateRefresh: () => Promise<SemanticCatalogStatusResponse>;
};

export function SettingsPanel({
  onClose,
  sessionId,
  sessionTitle,
  datasetName,
  settings,
  modelProfile,
  isAdmin = false,
  onSave,
  isStreaming,
  semanticConnectionId,
  semanticState,
  onSemanticStateRefresh,
}: Props) {
  const [draft, setDraft] = useState<UserSettings>(settings);
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  useEffect(() => {
    setDraft(settings);
  }, [settings]);

  async function handleSave(): Promise<void> {
    setIsSaving(true);
    setSaveMessage(null);
    try {
      await onSave(draft);
      setSaveMessage("Настройки сохранены на бэкенде.");
    } catch (error) {
      setSaveMessage(String(error));
    } finally {
      setIsSaving(false);
    }
  }

return (
    <div className="flex h-full flex-col bg-card/85 backdrop-blur-xl">
      <div className="flex items-center justify-between border-b border-border/40 px-6 py-5">
        <div>
          <div className="flex items-center gap-2">
            <Settings className="h-4 w-4 text-primary" />
            <h3 className="text-[14px] font-bold uppercase tracking-[0.12em]">Настройки</h3>
          </div>
          <p className="mt-1 text-[12px] text-muted-foreground">
            Параметры среды с сохранением в бэкенде.
          </p>
        </div>
        <button onClick={onClose} className="rounded-lg border border-border/40 bg-secondary/80 p-2 transition-all hover:bg-muted">
          <X className="h-4 w-4 text-muted-foreground" />
        </button>
      </div>

      <div className="no-scrollbar flex-1 space-y-6 overflow-y-auto p-6">
        <section className="rounded-2xl border border-border/50 bg-secondary/25 p-4">
          <div className="mb-3 flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.12em] text-muted-foreground">
            <Info className="h-3.5 w-3.5 text-primary" />
            Контекст сессии
          </div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-[13px]">
            {isAdmin ? <MetaRow label="Session ID" value={sessionId || "n/a"} /> : null}
            <MetaRow label="Сессия" value={sessionTitle || "Новый чат"} />
            <MetaRow label="Датасет" value={datasetName || "Не загружен"} />
            {isAdmin ? (
              <>
                <MetaRow label="Провайдер" value={modelProfile?.provider || "бэкенд"} />
                <MetaRow label="Модель" value={modelProfile?.model || "n/a"} />
                <MetaRow label="Commit" value={BUILD_COMMIT} />
              </>
            ) : null}
          </div>
        </section>

        <SectionCard
          title="Метаданные / Семантический слой"
          icon={<Database className="h-3.5 w-3.5" />}
          help="Семантический слой связывает бизнес-термины, метрики и связи с реальными таблицами источника. Для одного и того же файла или БД он сохраняется и переиспользуется между сессиями."
        >
          <SemanticCatalogBlock
            sessionId={sessionId}
            connectionId={semanticConnectionId}
            semanticState={semanticState}
            onSemanticStateRefresh={onSemanticStateRefresh}
          />
        </SectionCard>

        <SectionCard title="Профиль ответа" icon={<Brain className="h-3.5 w-3.5" />}>
          <div className="grid gap-4">
            <DepthButtons
              value={draft.analysis_depth}
              onChange={(value) =>
                setDraft((prev) => ({
                  ...prev,
                  analysis_depth: value,
                  agent_max_steps: clampAgentMaxStepsForDepth(value, prev.agent_max_steps),
                }))
              }
            />
          </div>
        </SectionCard>

        <SectionCard title="Среда агента" icon={<Cpu className="h-3.5 w-3.5" />}>
          <div className="grid grid-cols-2 gap-3">
            <NumberField label="Темп. чата" value={draft.llm_temperature_chat} step={0.05} onChange={(value) => setDraft((prev) => ({ ...prev, llm_temperature_chat: value }))} />
            <NumberField label="Темп. инструментов" value={draft.llm_temperature_tool} step={0.05} onChange={(value) => setDraft((prev) => ({ ...prev, llm_temperature_tool: value }))} />
            <NumberField label="Макс. токенов" value={draft.llm_max_tokens_default} step={128} onChange={(value) => setDraft((prev) => ({ ...prev, llm_max_tokens_default: Math.round(value) }))} />
            <NumberField label="Токены рассуждения" value={draft.llm_max_tokens_reasoning} step={128} onChange={(value) => setDraft((prev) => ({ ...prev, llm_max_tokens_reasoning: Math.round(value) }))} />
            <NumberField label="Таймаут бэкенда, сек" value={draft.backend_query_timeout_sec} step={5} onChange={(value) => setDraft((prev) => ({ ...prev, backend_query_timeout_sec: Math.round(value) }))} />
            <NumberField
              label="Макс. шагов"
              value={draft.agent_max_steps}
              min={2}
              max={ANALYSIS_DEPTH_STEP_CEILING[draft.analysis_depth]}
              step={1}
              onChange={(value) =>
                setDraft((prev) => ({
                  ...prev,
                  agent_max_steps: clampAgentMaxStepsForDepth(prev.analysis_depth, value),
                }))
              }
            />
            <NumberField label="Таймаут шага, сек" value={draft.agent_step_timeout_sec} step={5} onChange={(value) => setDraft((prev) => ({ ...prev, agent_step_timeout_sec: Math.round(value) }))} />
            <NumberField label="Внутр. рекурсия" value={draft.agent_inner_recursion_limit} step={1} onChange={(value) => setDraft((prev) => ({ ...prev, agent_inner_recursion_limit: Math.round(value) }))} />
          </div>
        </SectionCard>

        <SectionCard title="Рассуждение и поток" icon={<Eye className="h-3.5 w-3.5" />}>
          <div className="grid gap-3">
            {/* Управляет тем, формирует ли LLM рассуждение. */}
            <div className="inline-flex items-center justify-between rounded-xl border border-border/40 bg-background/25 px-4 py-3">
              <div className="flex items-center gap-2">
                <Brain className="h-3.5 w-3.5 text-muted-foreground" />
                <span className="text-sm">Включить расширенное рассуждение</span>
              </div>
              <Switch
                checked={draft.default_include_reasoning}
                onCheckedChange={(checked) => setDraft((prev) => ({ ...prev, default_include_reasoning: checked }))}
                className="ml-3"
              />
            </div>

            {/* Транспорт: потоковая передача токенов или цельный ответ. */}
            <div className="inline-flex items-center justify-between rounded-xl border border-border/40 bg-background/25 px-4 py-3">
              <div className="flex items-center gap-2">
                <Radio className="h-3.5 w-3.5 text-muted-foreground" />
                <span className="text-sm">Потоковая передача ответа</span>
              </div>
              <Switch
                checked={draft.llm_streaming}
                onCheckedChange={(checked) => setDraft((prev) => ({ ...prev, llm_streaming: checked }))}
                className="ml-3"
              />
            </div>

            {/* UI: показать или скрыть блоки рассуждения. */}
            <div className="inline-flex items-center justify-between rounded-xl border border-border/40 bg-background/25 px-4 py-3">
              <span className="text-sm">Показывать блоки рассуждения в чате</span>
              <Switch
                checked={draft.show_thinking}
                onCheckedChange={(checked) => setDraft((prev) => ({ ...prev, show_thinking: checked }))}
                className="ml-3"
              />
            </div>

            {/* Переключатели видов выключены, когда основной режим отключен. */}
            <div className={`ml-4 grid gap-2 transition-opacity ${draft.show_thinking ? "opacity-100" : "opacity-40"}`}>
              {(
                [
                  { key: "show_think_planning" as const, label: "Планирование" },
                  { key: "show_think_tool" as const,     label: "Синтез инструментов" },
                  { key: "show_think_final" as const,    label: "Финальный вывод" },
                ]
              ).map(({ key, label }) => (
                <div
                  key={key}
                  className="inline-flex items-center justify-between rounded-xl border border-border/30 bg-background/15 px-4 py-2.5"
                >
                  <span className="text-[13px] text-muted-foreground">{label}</span>
                  <Switch
                    checked={draft[key]}
                    disabled={!draft.show_thinking}
                    onCheckedChange={(checked) => setDraft((prev) => ({ ...prev, [key]: checked }))}
                    className="ml-3"
                  />
                </div>
              ))}
            </div>

            <div className="inline-flex items-center justify-between rounded-xl border border-border/40 bg-background/25 px-4 py-3">
              <div className="flex items-center gap-2">
                <ListTodo className="h-3.5 w-3.5 text-muted-foreground" />
                <span className="text-sm">Вызывать план при каждом запросе</span>
              </div>
              <Switch
                checked={draft.always_use_analysis_plan}
                aria-label="Вызывать план при каждом запросе"
                onCheckedChange={(checked) => setDraft((prev) => ({ ...prev, always_use_analysis_plan: checked }))}
                className="ml-3"
              />
            </div>
          </div>
        </SectionCard>

        <SectionCard title="Память сессии" icon={<Brain className="h-3.5 w-3.5" />}>
          <SessionMemoryBlock sessionId={sessionId} />
        </SectionCard>

        <SectionCard title="Блокнот сессии" icon={<BookOpen className="h-3.5 w-3.5" />}>
          <SessionNotebookBlock sessionId={sessionId} isStreaming={isStreaming} />
        </SectionCard>
      </div>

      <div className="space-y-3 border-t border-border/40 px-6 py-5">
        {saveMessage ? <div className="text-sm text-muted-foreground">{saveMessage}</div> : null}
        <button
          type="button"
          onClick={() => void handleSave()}
          disabled={isSaving}
          className="w-full rounded-2xl bg-primary py-3.5 font-bold text-primary-foreground shadow-xl shadow-primary/20 disabled:opacity-60"
        >
          {isSaving ? "Сохранение..." : "Сохранить изменения"}
        </button>
      </div>
    </div>
  );
}

function SessionMemoryBlock({ sessionId }: { sessionId: string }) {
  const [notes, setNotes] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const loadedRef = useRef(false);

  useEffect(() => {
    if (loadedRef.current) return;
    loadedRef.current = true;
    void load();
  }, []);

  async function load() {
    setIsLoading(true);
    setError(null);
    try {
      const session = await getSession(sessionId);
      setNotes(session.session_memory ?? "");
    } catch {
      setError("Не удалось загрузить память сессии");
    } finally {
      setIsLoading(false);
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 py-3 text-[13px] text-muted-foreground">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        Загрузка…
      </div>
    );
  }

  const isEmpty = !notes?.trim();

  return (
    <div className="space-y-3">
      <p className="text-[12px] text-muted-foreground leading-relaxed">
        Контекст текущей сессии: описания данных, ключевые находки, промежуточные выводы. Агент заполняет автоматически в ходе анализа.
      </p>

      {error && (
        <p className="rounded-lg bg-destructive/10 px-3 py-2 text-[12px] text-destructive">{error}</p>
      )}

      <div className={`relative min-h-[80px] rounded-xl border border-border/40 px-4 py-3 ${isEmpty ? "bg-muted/20" : "bg-background/30"}`}>
        {isEmpty ? (
          <p className="text-[13px] italic text-muted-foreground">Заметок пока нет — агент заполнит их в ходе анализа.</p>
        ) : (
          <MarkdownBlock
            content={notes ?? ""}
            className="text-[13px] [&_p]:mb-1 [&_ul]:mb-1 [&_li]:my-0 [&_h1]:text-sm [&_h2]:text-sm [&_h3]:text-xs"
          />
        )}
      </div>

      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={() => void load()}
          className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[12px] text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
        >
          <RefreshCw className="h-3 w-3" />
          Обновить
        </button>
      </div>
    </div>
  );
}

function NotebookCellView({ cell }: { cell: NotebookCell }) {
  const isDataSource = cell.entry_type === "data_source_change";

  if (isDataSource) {
    return (
      <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 overflow-hidden">
        <div className="flex items-center justify-between px-3 py-1.5 bg-blue-500/10 border-b border-blue-500/20">
          <span className="text-[11px] font-medium text-blue-400/80">📂 Источник данных</span>
          <span className="text-[10px] text-muted-foreground/60">{cell.timestamp}</span>
        </div>
        <div className="px-3 py-2 text-[12px] text-muted-foreground">{cell.result_summary}</div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border/40 bg-background/20 overflow-hidden">
      {/* Заголовок ячейки. */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-muted/30 border-b border-border/30">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-muted-foreground/50">Вход [{cell.index}]</span>
          <span className="text-[11px] font-medium text-foreground/70">{cell.tool_name || "код"}</span>
          {cell.language === "sql" && (
            <span className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-400/80 font-medium">SQL</span>
          )}
        </div>
        <span className="text-[10px] text-muted-foreground/50">{cell.timestamp}</span>
      </div>

      {/* Вопрос показывается только для sql_tool. */}
      {cell.question && (
        <div className="px-3 pt-2 pb-1 text-[11px] italic text-muted-foreground/70 border-b border-border/20">
          Вопрос: {cell.question}
        </div>
      )}

      {/* Блок кода. */}
      {cell.code && (
        <pre className="px-3 py-2 text-[11px] font-mono text-foreground/80 overflow-x-auto whitespace-pre-wrap break-all leading-relaxed bg-transparent m-0">
          <code>{cell.code}</code>
        </pre>
      )}

      {/* Результат. */}
      {cell.result_summary && (
        <div className="flex items-start gap-2 px-3 py-1.5 border-t border-border/30 bg-emerald-500/5">
          <span className="text-[10px] font-mono text-muted-foreground/50 shrink-0">Выход[{cell.index}]</span>
          <span className="text-[11px] text-emerald-400/80">{cell.result_summary}</span>
        </div>
      )}
    </div>
  );
}

function SessionNotebookBlock({ sessionId, isStreaming }: { sessionId: string; isStreaming?: boolean }) {
  const [cells, setCells] = useState<NotebookCell[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const prevStreamingRef = useRef(isStreaming);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    void load();
  }, []);

  // Перезагрузка после завершения потокового ответа агента.
  useEffect(() => {
    if (prevStreamingRef.current === true && isStreaming === false) {
      void load();
    }
    prevStreamingRef.current = isStreaming;
  }, [isStreaming]);

  async function load() {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getSessionNotebookCells(sessionId);
      setCells(data);
      setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
    } catch {
      setError("Не удалось загрузить блокнот сессии");
    } finally {
      setIsLoading(false);
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 py-3 text-[13px] text-muted-foreground">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        Загрузка…
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <p className="text-[12px] text-muted-foreground leading-relaxed">
        Лог выполнений песочницы: код, результаты и смены источников данных за текущую сессию.
      </p>

      {error && (
        <p className="rounded-lg bg-destructive/10 px-3 py-2 text-[12px] text-destructive">{error}</p>
      )}

      <div className="relative min-h-[80px] max-h-[480px] overflow-y-auto space-y-2 pr-0.5">
        {cells.length === 0 ? (
          <p className="text-[13px] italic text-muted-foreground py-2">Блокнот пуст: записи появятся после первого запроса с данными.</p>
        ) : (
          cells.map((cell) => <NotebookCellView key={cell.index} cell={cell} />)
        )}
        <div ref={bottomRef} />
      </div>

      <div className="flex items-center">
        <button
          type="button"
          onClick={() => { void load(); }}
          className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[12px] text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
        >
          <RefreshCw className="h-3 w-3" />
          Обновить
        </button>
      </div>
    </div>
  );
}

function SectionCard({ title, icon, help, children }: { title: string; icon: ReactNode; help?: string; children: ReactNode }) {
  return (
    <section className="rounded-2xl border border-border/50 bg-secondary/20 p-4">
      <div className="mb-4 flex items-start gap-3">
        <div className="rounded-lg bg-primary/10 p-1.5 text-primary">{icon}</div>
        <h4 className="flex items-center gap-2 text-[13px] font-bold uppercase tracking-[0.12em]">
          {title}
          {help ? <InfoHint text={help} /> : null}
        </h4>
      </div>
      {children}
    </section>
  );
}

function InfoHint({ text }: { text: string }) {
  return (
    <span className="group relative inline-flex">
      <span
        tabIndex={0}
        className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-border/50 bg-secondary/70 text-muted-foreground outline-none transition-all hover:border-primary/50 hover:bg-primary/10 hover:text-primary focus:border-primary/50 focus:bg-primary/10 focus:text-primary"
        aria-label={text}
      >
        <HelpCircle className="h-3.5 w-3.5" />
      </span>
      <span className="pointer-events-none absolute left-1/2 top-7 z-50 w-72 -translate-x-1/2 rounded-xl border border-border/60 bg-popover/95 px-3 py-2 text-left text-[12px] font-medium normal-case leading-relaxed tracking-normal text-popover-foreground opacity-0 shadow-2xl shadow-black/20 backdrop-blur-xl transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
        {text}
      </span>
    </span>
  );
}

function NumberField({
  label,
  value,
  onChange,
  step,
  min,
  max,
  hint,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: number;
  min?: number;
  max?: number;
  hint?: string;
}) {
  return (
    <div className="space-y-2">
      <label className="text-[11px] font-bold uppercase tracking-[0.1em] text-muted-foreground">{label}</label>
      {hint ? <p className="text-[10px] leading-snug text-muted-foreground">{hint}</p> : null}
      <input
        type="number"
        value={Number.isFinite(value) ? value : 0}
        step={step}
        min={min}
        max={max}
        onChange={(event) => onChange(parseFloat(event.target.value || "0"))}
        className="h-11 w-full rounded-xl border border-border/60 bg-secondary/70 px-3.5 text-[14px] font-medium outline-none transition-all focus:border-primary/50 focus:ring-4 focus:ring-primary/10"
      />
    </div>
  );
}

function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <>
      <div className="text-muted-foreground">{label}</div>
      <div className="text-right font-semibold">{value}</div>
    </>
  );
}

function DepthButtons({
  value,
  onChange,
}: {
  value: AnalysisDepth;
  onChange: (value: AnalysisDepth) => void;
}) {
  const options: Array<{ id: AnalysisDepth; label: string; desc: string }> = [
    { id: "light", label: "Легкий", desc: `До ${ANALYSIS_DEPTH_STEP_CEILING.light} шагов макс.` },
    { id: "medium", label: "Средний", desc: `До ${ANALYSIS_DEPTH_STEP_CEILING.medium} шагов макс.` },
    { id: "deep", label: "Глубокий", desc: `До ${ANALYSIS_DEPTH_STEP_CEILING.deep} шагов макс.` },
  ];
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-widest text-muted-foreground">
        <Sliders className="h-3.5 w-3.5" />
        Глубина анализа
      </div>
      <div className="grid grid-cols-3 gap-2">
        {options.map((option) => (
          <button
            key={option.id}
            type="button"
            onClick={() => onChange(option.id)}
            className={`rounded-xl border px-3 py-2 text-left transition-all ${
              value === option.id ? "border-primary/50 bg-primary/12 ring-1 ring-primary/20" : "border-border/40 bg-secondary/35"
            }`}
          >
            <div className={`text-[13px] font-bold ${value === option.id ? "text-primary" : "text-foreground"}`}>{option.label}</div>
            <div className="mt-0.5 text-[11px] text-muted-foreground">{option.desc}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
