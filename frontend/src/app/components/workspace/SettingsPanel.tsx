import { useEffect, useState, type ReactNode } from "react";
import { Brain, Cpu, Info, Settings, Shield, Sliders, X } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import type { AnalysisDepth, RuntimeModelProfile, UserSettings } from "../../lib/backend-types";

type Props = {
  onClose: () => void;
  sessionTitle: string;
  datasetName: string;
  settings: UserSettings;
  modelProfile: RuntimeModelProfile | null;
  onSave: (payload: Partial<UserSettings>) => Promise<void>;
};

export function SettingsPanel({ onClose, sessionTitle, datasetName, settings, modelProfile, onSave }: Props) {
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
      setSaveMessage("Настройки сохранены на backend.");
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
            Runtime-параметры с сохранением в backend.
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
            <MetaRow label="Сессия" value={sessionTitle || "Новый чат"} />
            <MetaRow label="Датасет" value={datasetName || "Не загружен"} />
            <MetaRow label="Провайдер" value={modelProfile?.provider || "backend"} />
            <MetaRow label="Модель" value={modelProfile?.model || "n/a"} />
          </div>
        </section>

        <SectionCard title="Профиль ответа" icon={<Brain className="h-3.5 w-3.5" />}>
          <div className="grid gap-4">
            <label className="inline-flex items-center justify-between rounded-xl border border-border/40 bg-background/25 px-4 py-3">
              <span className="text-sm">Показывать reasoning по умолчанию</span>
              <input
                type="checkbox"
                checked={draft.default_include_reasoning}
                onChange={(event) => setDraft((prev) => ({ ...prev, default_include_reasoning: event.target.checked }))}
                className="h-4 w-4 accent-primary"
              />
            </label>
            <label className="grid gap-2">
              <span className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground">Стиль ответа</span>
              <Select
                value={draft.default_answer_style}
                onValueChange={(value) =>
                  setDraft((prev) => ({
                    ...prev,
                    default_answer_style: value === "concise" ? "concise" : "detailed",
                  }))
                }
              >
                <SelectTrigger className="h-11 rounded-xl border border-border/60 bg-secondary/70 px-4 text-sm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="detailed">Развернутый</SelectItem>
                  <SelectItem value="concise">Краткий</SelectItem>
                </SelectContent>
              </Select>
            </label>
            <DepthButtons
              value={draft.analysis_depth}
              onChange={(value) => setDraft((prev) => ({ ...prev, analysis_depth: value }))}
            />
          </div>
        </SectionCard>

        <SectionCard title="Runtime агента" icon={<Cpu className="h-3.5 w-3.5" />}>
          <div className="grid grid-cols-2 gap-3">
            <NumberField label="Темп. чата" value={draft.llm_temperature_chat} step={0.05} onChange={(value) => setDraft((prev) => ({ ...prev, llm_temperature_chat: value }))} />
            <NumberField label="Темп. инструментов" value={draft.llm_temperature_tool} step={0.05} onChange={(value) => setDraft((prev) => ({ ...prev, llm_temperature_tool: value }))} />
            <NumberField label="Макс. токенов" value={draft.llm_max_tokens_default} step={128} onChange={(value) => setDraft((prev) => ({ ...prev, llm_max_tokens_default: Math.round(value) }))} />
            <NumberField label="Токены reasoning" value={draft.llm_max_tokens_reasoning} step={128} onChange={(value) => setDraft((prev) => ({ ...prev, llm_max_tokens_reasoning: Math.round(value) }))} />
            <NumberField label="Таймаут backend, сек" value={draft.backend_query_timeout_sec} step={5} onChange={(value) => setDraft((prev) => ({ ...prev, backend_query_timeout_sec: Math.round(value) }))} />
            <NumberField label="Макс. шагов" value={draft.agent_max_steps} step={1} onChange={(value) => setDraft((prev) => ({ ...prev, agent_max_steps: Math.round(value) }))} />
            <NumberField label="Таймаут шага, сек" value={draft.agent_step_timeout_sec} step={5} onChange={(value) => setDraft((prev) => ({ ...prev, agent_step_timeout_sec: Math.round(value) }))} />
            <NumberField label="Внутр. рекурсия" value={draft.agent_inner_recursion_limit} step={1} onChange={(value) => setDraft((prev) => ({ ...prev, agent_inner_recursion_limit: Math.round(value) }))} />
          </div>
        </SectionCard>

        <SectionCard title="Future-ready блок" icon={<Shield className="h-3.5 w-3.5" />}>
          <div className="rounded-xl border border-border/40 bg-background/25 p-4 text-sm text-muted-foreground">
            Блок оставлен как расширяемая зона под будущие runtime-переключатели и feature flags.
          </div>
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

function SectionCard({ title, icon, children }: { title: string; icon: ReactNode; children: ReactNode }) {
  return (
    <section className="rounded-2xl border border-border/50 bg-secondary/20 p-4">
      <div className="mb-4 flex items-start gap-3">
        <div className="rounded-lg bg-primary/10 p-1.5 text-primary">{icon}</div>
        <h4 className="text-[13px] font-bold uppercase tracking-[0.12em]">{title}</h4>
      </div>
      {children}
    </section>
  );
}

function NumberField({
  label,
  value,
  onChange,
  step,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: number;
}) {
  return (
    <div className="space-y-2">
      <label className="text-[11px] font-bold uppercase tracking-[0.1em] text-muted-foreground">{label}</label>
      <input
        type="number"
        value={Number.isFinite(value) ? value : 0}
        step={step}
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
    { id: "light", label: "Легкий", desc: "Быстрее" },
    { id: "medium", label: "Средний", desc: "Баланс" },
    { id: "deep", label: "Глубокий", desc: "Детальнее" },
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
