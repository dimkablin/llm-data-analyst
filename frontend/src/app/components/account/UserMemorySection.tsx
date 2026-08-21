import { useEffect, useRef, useState } from "react";
import { Brain, Pencil, RefreshCw, Save, X } from "lucide-react";
import { getUserMemory, updateUserMemory } from "../../lib/backend-api";
import type { UserMemory } from "../../lib/backend-types";
import { summarizeError } from "../../lib/format";
import { Button } from "../ui/button";

const EMPTY_MEMORY: UserMemory = { profile: "", notes: "" };

type EditTarget = "profile" | "notes" | null;

export function UserMemorySection() {
  const [memory, setMemory] = useState<UserMemory>(EMPTY_MEMORY);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [editing, setEditing] = useState<EditTarget>(null);
  const [draft, setDraft] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    void load();
  }, []);

  useEffect(() => {
    if (editing && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [editing]);

  async function load() {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getUserMemory();
      setMemory(data);
    } catch (e) {
      setError(summarizeError(e));
    } finally {
      setIsLoading(false);
    }
  }

  function startEdit(target: EditTarget) {
    if (!target) return;
    setDraft(memory[target]);
    setEditing(target);
    setSaveMessage(null);
  }

  function cancelEdit() {
    setEditing(null);
    setDraft("");
  }

  async function save() {
    if (!editing) return;
    setIsSaving(true);
    setError(null);
    try {
      const updated = await updateUserMemory({ [editing]: draft });
      setMemory(updated);
      setSaveMessage("Сохранено");
      setEditing(null);
      setDraft("");
      setTimeout(() => setSaveMessage(null), 2500);
    } catch (e) {
      setError(summarizeError(e));
    } finally {
      setIsSaving(false);
    }
  }

  async function clearField(field: "profile" | "notes") {
    setIsSaving(true);
    setError(null);
    try {
      const updated = await updateUserMemory({ [field]: "" });
      setMemory(updated);
    } catch (e) {
      setError(summarizeError(e));
    } finally {
      setIsSaving(false);
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-muted-foreground text-sm py-4">
        <RefreshCw className="h-4 w-4 animate-spin" />
        Загрузка памяти…
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Brain className="h-5 w-5 text-primary" />
        <h3 className="font-semibold text-base">Память агента</h3>
      </div>

      <p className="text-sm text-muted-foreground leading-relaxed">
        Агент использует эти данные для персонализации ответов. <strong>Профиль</strong> вы
        редактируете вручную. <strong>Заметки</strong> агент пишет сам — только факты о вас
        (предпочтения, роль, экспертиза). Контекст анализа хранится отдельно в памяти сессии.
      </p>

      {error && (
        <p className="text-sm text-destructive bg-destructive/10 rounded-md px-3 py-2">{error}</p>
      )}
      {saveMessage && (
        <p className="text-sm text-green-600 dark:text-green-400">{saveMessage}</p>
      )}

      {/* Profile */}
      <MemoryBlock
        title="Профиль пользователя"
        subtitle="Кто вы, в какой области работаете, какой стиль ответов предпочитаете?"
        value={memory.profile}
        isEditing={editing === "profile"}
        draft={draft}
        isSaving={isSaving}
        textareaRef={editing === "profile" ? textareaRef : undefined}
        onEdit={() => startEdit("profile")}
        onCancel={cancelEdit}
        onSave={save}
        onDraftChange={setDraft}
        onClear={() => void clearField("profile")}
      />

      {/* Notes */}
      <MemoryBlock
        title="Заметки агента"
        subtitle="Заметки агента о вас: предпочтения, экспертиза, роль."
        value={memory.notes}
        isEditing={editing === "notes"}
        draft={draft}
        isSaving={isSaving}
        textareaRef={editing === "notes" ? textareaRef : undefined}
        onEdit={() => startEdit("notes")}
        onCancel={cancelEdit}
        onSave={save}
        onDraftChange={setDraft}
        onClear={() => void clearField("notes")}
      />
    </div>
  );
}

// ── Sub-component ──────────────────────────────────────────────────────────

interface MemoryBlockProps {
  title: string;
  subtitle: string;
  value: string;
  isEditing: boolean;
  draft: string;
  isSaving: boolean;
  textareaRef?: React.RefObject<HTMLTextAreaElement>;
  onEdit: () => void;
  onCancel: () => void;
  onSave: () => void;
  onDraftChange: (v: string) => void;
  onClear: () => void;
}

function MemoryBlock({
  title,
  subtitle,
  value,
  isEditing,
  draft,
  isSaving,
  textareaRef,
  onEdit,
  onCancel,
  onSave,
  onDraftChange,
  onClear,
}: MemoryBlockProps) {
  const isEmpty = !value.trim();

  return (
    <div className="rounded-lg border bg-card p-4 space-y-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="font-medium text-sm">{title}</p>
          <p className="text-xs text-muted-foreground mt-0.5">{subtitle}</p>
        </div>
        {!isEditing && (
          <div className="flex gap-1 shrink-0">
            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onEdit} title="Редактировать">
              <Pencil className="h-3.5 w-3.5" />
            </Button>
            {!isEmpty && (
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7 text-muted-foreground hover:text-destructive"
                onClick={onClear}
                disabled={isSaving}
                title="Очистить"
              >
                <X className="h-3.5 w-3.5" />
              </Button>
            )}
          </div>
        )}
      </div>

      {isEditing ? (
        <div className="space-y-2">
          <textarea
            ref={textareaRef}
            value={draft}
            onChange={(e) => onDraftChange(e.target.value)}
            rows={6}
            className="w-full rounded-md border bg-background px-3 py-2 text-sm
                       placeholder:text-muted-foreground focus:outline-none focus:ring-1
                       focus:ring-ring resize-y font-mono"
            placeholder="Введите текст в формате разметки…"
          />
          <div className="flex gap-2">
            <Button size="sm" onClick={onSave} disabled={isSaving}>
              <Save className="h-3.5 w-3.5 mr-1.5" />
              {isSaving ? "Сохранение…" : "Сохранить"}
            </Button>
            <Button variant="outline" size="sm" onClick={onCancel} disabled={isSaving}>
              Отмена
            </Button>
          </div>
        </div>
      ) : (
        <div
          className={`text-sm rounded-md px-3 py-2 min-h-[3rem] whitespace-pre-wrap font-mono
                     ${isEmpty
                       ? "text-muted-foreground italic bg-muted/30"
                       : "bg-muted/40 text-foreground"
                     }`}
        >
          {isEmpty ? "Пусто" : value}
        </div>
      )}
    </div>
  );
}
