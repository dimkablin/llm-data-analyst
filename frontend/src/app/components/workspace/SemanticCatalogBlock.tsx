import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { HelpCircle, Link2, Loader2, Plus, RefreshCw, Sparkles, Trash2 } from "lucide-react";
import {
  createSemanticMetric,
  createSemanticRelationship,
  createSemanticTerm,
  deleteSemanticMetric,
  deleteSemanticRelationship,
  deleteSemanticTerm,
  getSemanticCatalog,
  getSemanticCatalogStatus,
  refreshSemanticCatalog,
  startSemanticCatalogGeneration,
  updateSemanticMetric,
  updateSemanticRelationship,
  updateSemanticTerm,
} from "../../lib/backend-api";
import {
  type SemanticCatalog,
  type SemanticCatalogGenerationSummary,
  type SemanticMetric,
  type SemanticMetricPayload,
  type SemanticRelationship,
  type SemanticRelationshipPayload,
  type SemanticTerm,
  type SemanticTermPayload,
} from "../../lib/backend-types";

type Tab = "overview" | "metrics" | "terms" | "relationships";

type MetricDraft = {
  key: string;
  name: string;
  base_table: string;
  expr: string;
  agg: NonNullable<SemanticMetricPayload["agg"]>;
  default_time_dimension: string;
  allowed_dimensions: string;
  synonyms: string;
  formula: string;
  advanced: boolean;
};

type TermDraft = {
  name: string;
  description: string;
  synonyms: string;
  entity_refs: string;
};

type RelationshipDraft = SemanticRelationshipPayload;

const EMPTY_METRIC_DRAFT: MetricDraft = {
  key: "",
  name: "",
  base_table: "",
  expr: "",
  agg: "sum",
  default_time_dimension: "",
  allowed_dimensions: "",
  synonyms: "",
  formula: "",
  advanced: false,
};

const EMPTY_TERM_DRAFT: TermDraft = {
  name: "",
  description: "",
  synonyms: "",
  entity_refs: "",
};

const EMPTY_RELATIONSHIP_DRAFT: RelationshipDraft = {
  from_table: "",
  from_column: "",
  to_table: "",
  to_column: "",
  cardinality: "many_to_one",
  description: "",
  is_active: true,
};

const TAB_LABELS: Record<Tab, string> = {
  overview: "Обзор",
  metrics: "Метрики",
  terms: "Термины",
  relationships: "Связи",
};

const TAB_HELP: Partial<Record<Tab, string>> = {
  metrics: "Метрики описывают проверенные бизнес-расчеты: формулу, таблицу, агрегацию, время и допустимые измерения.",
  terms: "Термины хранят бизнес-словарь и синонимы пользователя. Они не зависят от конкретной сессии.",
  relationships: "Связи задают безопасные join-пути между таблицами и защищают аналитику от double counting.",
};

const VALUE_LABELS: Record<string, string> = {
  metric_candidate: "кандидат в метрику",
  dimension: "измерение",
  time: "время",
  identifier: "идентификатор",
  foreign_key_candidate: "кандидат во внешний ключ",
  text: "текст",
  flag: "флаг",
  fact: "факт",
  bridge: "мост",
  snapshot: "снимок",
  unknown: "неизвестно",
  many_to_one: "многие к одному",
  one_to_one: "один к одному",
  sum: "сумма",
  avg: "среднее",
  count: "количество",
  count_distinct: "уникальное количество",
  min: "минимум",
  max: "максимум",
};

export function SemanticCatalogBlock({ sessionId }: { sessionId: string }) {
  const [catalog, setCatalog] = useState<SemanticCatalog | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("overview");
  const [status, setStatus] = useState<string>("empty");
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isEditorOpen, setIsEditorOpen] = useState(false);
  const [generationNotice, setGenerationNotice] = useState<string | null>(null);
  const [generationSummary] = useState<SemanticCatalogGenerationSummary | null>(null);
  const [editingMetricId, setEditingMetricId] = useState<string | null>(null);
  const [editingTermId, setEditingTermId] = useState<string | null>(null);
  const [editingRelationshipId, setEditingRelationshipId] = useState<string | null>(null);
  const [draft, setDraft] = useState<MetricDraft>(EMPTY_METRIC_DRAFT);
  const [termDraft, setTermDraft] = useState<TermDraft>(EMPTY_TERM_DRAFT);
  const [relationshipDraft, setRelationshipDraft] = useState<RelationshipDraft>(EMPTY_RELATIONSHIP_DRAFT);
  const generationPollRef = useRef<number | null>(null);

  useEffect(() => {
    void load();
    return () => clearGenerationPoll();
  }, [sessionId]);

  function clearGenerationPoll(): void {
    if (generationPollRef.current !== null) {
      window.clearTimeout(generationPollRef.current);
      generationPollRef.current = null;
    }
  }

  function applyCatalog(next: SemanticCatalog): void {
    setCatalog(next);
    setStatus(next.status);
    const firstTable = next.tables[0]?.qualified_name || "";
    setDraft((prev) => ({ ...prev, base_table: prev.base_table || firstTable }));
    setRelationshipDraft((prev) => ({
      ...prev,
      from_table: prev.from_table || firstTable,
      to_table: prev.to_table || next.tables[1]?.qualified_name || firstTable,
    }));
  }

  async function load(): Promise<void> {
    setIsLoading(true);
    setError(null);
    try {
      const nextStatus = await getSemanticCatalogStatus(sessionId);
      setStatus(nextStatus.status);
      if (nextStatus.status === "empty") {
        setCatalog(null);
        return;
      }
      applyCatalog(await getSemanticCatalog(sessionId));
    } catch (loadError) {
      setError(String(loadError));
    } finally {
      setIsLoading(false);
    }
  }

  async function refresh(): Promise<void> {
    setIsRefreshing(true);
    setError(null);
    setGenerationNotice(null);
    try {
      applyCatalog(await refreshSemanticCatalog(sessionId));
    } catch (refreshError) {
      setError(String(refreshError));
    } finally {
      setIsRefreshing(false);
    }
  }

  async function generateWithAi(): Promise<void> {
    clearGenerationPoll();
    setIsGenerating(true);
    setError(null);
    setGenerationNotice("AI-генерация запущена в фоне. Можно продолжать работу с агентом.");
    try {
      await startSemanticCatalogGeneration(sessionId);
      setStatus("indexing");
      queueGenerationPoll();
    } catch (generateError) {
      setGenerationNotice(null);
      setError(String(generateError));
      setIsGenerating(false);
    }
  }

  function queueGenerationPoll(attempt = 0): void {
    generationPollRef.current = window.setTimeout(() => void pollGeneration(attempt), 3000);
  }

  async function pollGeneration(attempt = 0): Promise<void> {
    let keepWaiting = false;
    generationPollRef.current = null;
    try {
      const nextStatus = await getSemanticCatalogStatus(sessionId);
      setStatus(nextStatus.status);
      if (nextStatus.status === "indexing" || nextStatus.status === "pending") {
        if (attempt < 120) {
          keepWaiting = true;
          queueGenerationPoll(attempt + 1);
        } else {
          setGenerationNotice("AI-генерация еще выполняется. Можно продолжать работу и обновить статус позже.");
        }
        return;
      }
      if (nextStatus.status === "failed") {
        setGenerationNotice(null);
        setError(nextStatus.error || "AI-генерация семантического слоя завершилась с ошибкой.");
      } else if (nextStatus.status !== "empty") {
        applyCatalog(await getSemanticCatalog(sessionId));
        setGenerationNotice("AI-генерация завершена. Семантический слой обновлен.");
      }
    } catch (pollError) {
      setError(String(pollError));
    } finally {
      if (!keepWaiting) {
        setIsGenerating(false);
      }
    }
  }

  function editMetric(metric: SemanticMetric): void {
    setEditingMetricId(metric.metric_id);
    setIsEditorOpen(true);
    setActiveTab("metrics");
    setDraft({
      key: metric.key,
      name: metric.name,
      base_table: metric.base_table,
      expr: metric.expr ?? "",
      agg: (metric.agg ?? "sum") as MetricDraft["agg"],
      default_time_dimension: metric.default_time_dimension ?? "",
      allowed_dimensions: metric.allowed_dimensions.join(", "),
      synonyms: metric.synonyms.join(", "),
      formula: metric.formula ?? "",
      advanced: metric.type !== "simple",
    });
  }

  function resetMetricDraft(): void {
    setEditingMetricId(null);
    setDraft({
      ...EMPTY_METRIC_DRAFT,
      base_table: catalog?.tables[0]?.qualified_name || "",
    });
  }

  function metricPayload(): SemanticMetricPayload {
    const common = {
      name: draft.name.trim(),
      default_time_dimension: draft.default_time_dimension.trim() || null,
      allowed_dimensions: splitCsv(draft.allowed_dimensions),
      synonyms: splitCsv(draft.synonyms),
    };
    if (draft.advanced) {
      return {
        key: draft.key.trim(),
        type: "derived",
        base_table: draft.base_table,
        ...common,
        formula: draft.formula.trim(),
      };
    }
    return {
      key: draft.key.trim(),
      type: "simple",
      base_table: draft.base_table,
      ...common,
      expr: draft.expr.trim(),
      agg: draft.agg,
    };
  }

  async function saveMetric(): Promise<void> {
    setIsSaving(true);
    setError(null);
    try {
      const payload = metricPayload();
      if (editingMetricId) {
        const { key: _key, ...updatePayload } = payload;
        await updateSemanticMetric(sessionId, editingMetricId, updatePayload);
      } else {
        await createSemanticMetric(sessionId, payload);
      }
      applyCatalog(await getSemanticCatalog(sessionId));
      resetMetricDraft();
      setIsEditorOpen(false);
    } catch (saveError) {
      setError(String(saveError));
    } finally {
      setIsSaving(false);
    }
  }

  async function removeMetric(metricId: string): Promise<void> {
    setError(null);
    try {
      await deleteSemanticMetric(sessionId, metricId);
      applyCatalog(await getSemanticCatalog(sessionId));
      if (editingMetricId === metricId) resetMetricDraft();
    } catch (deleteError) {
      setError(String(deleteError));
    }
  }

  function editTerm(term: SemanticTerm): void {
    setEditingTermId(term.term_id);
    setIsEditorOpen(true);
    setActiveTab("terms");
    setTermDraft({
      name: term.name,
      description: term.description,
      synonyms: term.synonyms.join(", "),
      entity_refs: term.entity_refs.join(", "),
    });
  }

  function resetTermDraft(): void {
    setEditingTermId(null);
    setTermDraft(EMPTY_TERM_DRAFT);
  }

  function termPayload(): SemanticTermPayload {
    return {
      name: termDraft.name.trim(),
      description: termDraft.description.trim(),
      synonyms: splitCsv(termDraft.synonyms),
      entity_refs: splitCsv(termDraft.entity_refs),
    };
  }

  async function saveTerm(): Promise<void> {
    setIsSaving(true);
    setError(null);
    try {
      const payload = termPayload();
      if (editingTermId) {
        await updateSemanticTerm(sessionId, editingTermId, payload);
      } else {
        await createSemanticTerm(sessionId, payload);
      }
      applyCatalog(await getSemanticCatalog(sessionId));
      resetTermDraft();
      setIsEditorOpen(false);
    } catch (saveError) {
      setError(String(saveError));
    } finally {
      setIsSaving(false);
    }
  }

  async function removeTerm(termId: string): Promise<void> {
    setError(null);
    try {
      await deleteSemanticTerm(sessionId, termId);
      applyCatalog(await getSemanticCatalog(sessionId));
      if (editingTermId === termId) resetTermDraft();
    } catch (deleteError) {
      setError(String(deleteError));
    }
  }

  function editRelationship(relationship: SemanticRelationship): void {
    setEditingRelationshipId(relationship.relationship_id);
    setIsEditorOpen(true);
    setActiveTab("relationships");
    setRelationshipDraft({
      from_table: relationship.from_table,
      from_column: relationship.from_column,
      to_table: relationship.to_table,
      to_column: relationship.to_column,
      cardinality: relationship.cardinality,
      description: relationship.description,
      is_active: relationship.is_active,
    });
  }

  function resetRelationshipDraft(): void {
    setEditingRelationshipId(null);
    setRelationshipDraft({
      ...EMPTY_RELATIONSHIP_DRAFT,
      from_table: catalog?.tables[0]?.qualified_name || "",
      to_table: catalog?.tables[1]?.qualified_name || catalog?.tables[0]?.qualified_name || "",
    });
  }

  async function saveRelationship(): Promise<void> {
    setIsSaving(true);
    setError(null);
    try {
      if (editingRelationshipId) {
        await updateSemanticRelationship(sessionId, editingRelationshipId, relationshipDraft);
      } else {
        await createSemanticRelationship(sessionId, relationshipDraft);
      }
      applyCatalog(await getSemanticCatalog(sessionId));
      resetRelationshipDraft();
      setIsEditorOpen(false);
    } catch (saveError) {
      setError(String(saveError));
    } finally {
      setIsSaving(false);
    }
  }

  async function removeRelationship(relationshipId: string): Promise<void> {
    setError(null);
    try {
      await deleteSemanticRelationship(sessionId, relationshipId);
      applyCatalog(await getSemanticCatalog(sessionId));
      if (editingRelationshipId === relationshipId) resetRelationshipDraft();
    } catch (deleteError) {
      setError(String(deleteError));
    }
  }

  const selectedColumns = columnsFor(catalog, draft.base_table);
  const relationshipFromColumns = columnsFor(catalog, relationshipDraft.from_table);
  const relationshipToColumns = columnsFor(catalog, relationshipDraft.to_table);
  const tableOptions = (catalog?.tables ?? []).map((table) => table.qualified_name);
  const metricReady = draft.name.trim() && draft.key.trim() && draft.base_table && (
    draft.advanced ? draft.formula.trim() : draft.expr.trim()
  );
  const termReady = termDraft.name.trim();
  const relationshipReady = relationshipDraft.from_table
    && relationshipDraft.from_column
    && relationshipDraft.to_table
    && relationshipDraft.to_column;
  const canGenerateWithAi = catalog?.source_type === "db_connection" || catalog?.source_type === "csv";

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 py-3 text-[13px] text-muted-foreground">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        Загрузка...
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-end gap-2">
        {canGenerateWithAi ? (
          <button
            type="button"
            onClick={() => void generateWithAi()}
            disabled={isGenerating}
            className="flex items-center gap-1.5 rounded-lg border border-primary/35 bg-primary/10 px-2.5 py-1.5 text-[12px] font-semibold text-primary transition-colors hover:bg-primary/15 disabled:opacity-60"
          >
            {isGenerating ? <Loader2 className="h-3 w-3 animate-spin" /> : <Sparkles className="h-3 w-3" />}
            Сгенерировать семантический слой
          </button>
        ) : null}
        <button
          type="button"
          onClick={() => void refresh()}
          disabled={isRefreshing}
          className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[12px] text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground disabled:opacity-60"
        >
          {isRefreshing ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
          Обновить
        </button>
      </div>

      {error ? (
        <p className="rounded-lg bg-destructive/10 px-3 py-2 text-[12px] text-destructive">{error}</p>
      ) : null}

      {generationNotice ? (
        <p className="rounded-lg border border-primary/25 bg-primary/10 px-3 py-2 text-[12px] text-primary">
          {generationNotice}
        </p>
      ) : null}

      {generationSummary ? (
        <p className="rounded-lg border border-primary/25 bg-primary/10 px-3 py-2 text-[12px] text-primary">
          AI обновил слой: таблиц {generationSummary.tables_scanned}, метрик +{generationSummary.metrics_added},
          терминов +{generationSummary.terms_added}, связей +{generationSummary.relationships_added}
          {generationSummary.rejected_items.length ? `, отклонено ${generationSummary.rejected_items.length}` : ""}.
        </p>
      ) : null}

      {status === "unbound" ? (
        <p className="rounded-xl border border-border/40 bg-secondary/35 px-3 py-2 text-[12px] text-muted-foreground">
          Бизнес-словарь доступен. Подключите источник данных, чтобы привязать таблицы, метрики и связи.
        </p>
      ) : null}

      {!catalog ? (
        <p className="text-[13px] text-muted-foreground">Семантический каталог пока не создан.</p>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
            {(Object.keys(TAB_LABELS) as Tab[]).map((tab) => (
              <button
                key={tab}
                type="button"
                onClick={() => {
                  setActiveTab(tab);
                  setIsEditorOpen(false);
                }}
                className={`rounded-xl border px-3 py-2 text-left text-[12px] font-bold transition-all ${
                  activeTab === tab ? "border-primary/50 bg-primary/12 text-primary ring-1 ring-primary/20" : "border-border/40 bg-secondary/35 text-muted-foreground hover:border-border/70 hover:text-foreground"
                }`}
              >
                <span className="flex items-center gap-1.5">
                  {TAB_LABELS[tab]}
                  {TAB_HELP[tab] ? <InfoHint text={TAB_HELP[tab]} /> : null}
                </span>
              </button>
            ))}
          </div>

          {activeTab === "overview" ? (
            <Overview catalog={catalog} />
          ) : null}

          {activeTab === "metrics" ? (
            <>
              <ListHeader title="Метрики" isOpen={isEditorOpen} onToggle={() => { resetMetricDraft(); setIsEditorOpen((open) => !open); }} />
              <div className="space-y-1">
                {catalog.metrics.length === 0 ? (
                  <p className="text-[12px] text-muted-foreground">Метрик пока нет.</p>
                ) : (
                  catalog.metrics.map((metric) => (
                    <RowButton key={metric.metric_id} onEdit={() => editMetric(metric)} onDelete={() => void removeMetric(metric.metric_id)}>
                      <span className="block truncate font-medium">{metric.key} / {metric.name}</span>
                      <span className="block truncate text-[11px] text-muted-foreground">{metric.formula}</span>
                    </RowButton>
                  ))
                )}
              </div>

              {isEditorOpen ? <div className="grid gap-3 rounded-xl border border-border/40 bg-background/25 p-3">
                <div className="grid grid-cols-2 gap-3">
                  <TextField label="Ключ" value={draft.key} disabled={Boolean(editingMetricId)} onChange={(value) => setDraft((prev) => ({ ...prev, key: value }))} />
                  <TextField label="Название" value={draft.name} onChange={(value) => setDraft((prev) => ({ ...prev, name: value }))} />
                  <SelectField
                    label="Таблица"
                    value={draft.base_table}
                    disabled={Boolean(editingMetricId)}
                    options={tableOptions}
                    onChange={(value) => setDraft((prev) => ({ ...prev, base_table: value, expr: "", default_time_dimension: "" }))}
                  />
                  <SelectField
                    label="Колонка"
                    value={draft.expr}
                    disabled={draft.advanced}
                    options={selectedColumns.map((column) => column.name)}
                    onChange={(value) => setDraft((prev) => ({ ...prev, expr: value }))}
                  />
                  <SelectField
                    label="Агрегация"
                    value={draft.agg}
                    disabled={draft.advanced}
                    options={["sum", "avg", "count", "count_distinct", "min", "max"]}
                    onChange={(value) => setDraft((prev) => ({ ...prev, agg: value as MetricDraft["agg"] }))}
                  />
                  <SelectField
                    label="Время"
                    value={draft.default_time_dimension}
                    options={["", ...selectedColumns.map((column) => column.name)]}
                    onChange={(value) => setDraft((prev) => ({ ...prev, default_time_dimension: value }))}
                  />
                </div>
                <TextField label="Измерения" value={draft.allowed_dimensions} onChange={(value) => setDraft((prev) => ({ ...prev, allowed_dimensions: value }))} />
                <TextField label="Синонимы" value={draft.synonyms} onChange={(value) => setDraft((prev) => ({ ...prev, synonyms: value }))} />
                <details className="rounded-xl border border-border/40 bg-background/25 px-3 py-2">
                  <summary className="cursor-pointer text-[12px] font-medium text-muted-foreground">Расширенная SQL-формула</summary>
                  <div className="mt-3 space-y-2">
                    <label className="flex items-center gap-2 text-[12px] text-muted-foreground">
                      <input
                        type="checkbox"
                        checked={draft.advanced}
                        onChange={(event) => setDraft((prev) => ({ ...prev, advanced: event.target.checked }))}
                      />
                      Использовать SQL-формулу
                    </label>
                    <textarea
                      value={draft.formula}
                      onChange={(event) => setDraft((prev) => ({ ...prev, formula: event.target.value }))}
                      disabled={!draft.advanced}
                      className="min-h-[72px] w-full rounded-xl border border-border/60 bg-secondary/70 px-3.5 py-2 text-[13px] font-mono outline-none transition-all focus:border-primary/50 focus:ring-4 focus:ring-primary/10"
                      placeholder="SUM(amount)"
                    />
                  </div>
                </details>
                <SaveButton disabled={isSaving || !metricReady} isSaving={isSaving} label={editingMetricId ? "Сохранить метрику" : "Добавить метрику"} onClick={() => void saveMetric()} />
              </div> : null}
            </>
          ) : null}

          {activeTab === "terms" ? (
            <>
              <ListHeader title="Термины" isOpen={isEditorOpen} onToggle={() => { resetTermDraft(); setIsEditorOpen((open) => !open); }} />
              <div className="space-y-1">
                {catalog.terms.length === 0 ? (
                  <p className="text-[12px] text-muted-foreground">Терминов пока нет.</p>
                ) : (
                  catalog.terms.map((term) => (
                    <RowButton key={term.term_id} onEdit={() => editTerm(term)} onDelete={() => void removeTerm(term.term_id)}>
                      <span className="block truncate font-medium">{term.name}</span>
                      <span className="block truncate text-[11px] text-muted-foreground">
                        {term.synonyms.join(", ") || term.description}
                      </span>
                    </RowButton>
                  ))
                )}
              </div>
              {isEditorOpen ? <div className="grid gap-3 rounded-xl border border-border/40 bg-background/25 p-3">
                <TextField label="Термин" value={termDraft.name} onChange={(value) => setTermDraft((prev) => ({ ...prev, name: value }))} />
                <TextField label="Определение" value={termDraft.description} onChange={(value) => setTermDraft((prev) => ({ ...prev, description: value }))} />
                <TextField label="Синонимы" value={termDraft.synonyms} onChange={(value) => setTermDraft((prev) => ({ ...prev, synonyms: value }))} />
                <TextField label="Связанные сущности" value={termDraft.entity_refs} onChange={(value) => setTermDraft((prev) => ({ ...prev, entity_refs: value }))} />
                <SaveButton disabled={isSaving || !termReady} isSaving={isSaving} label={editingTermId ? "Сохранить термин" : "Добавить термин"} onClick={() => void saveTerm()} />
              </div> : null}
            </>
          ) : null}

          {activeTab === "relationships" ? (
            <>
              <ListHeader title="Связи" isOpen={isEditorOpen} onToggle={() => { resetRelationshipDraft(); setIsEditorOpen((open) => !open); }} />
              <div className="space-y-1">
                {catalog.relationships.length === 0 ? (
                  <p className="text-[12px] text-muted-foreground">Связей пока нет.</p>
                ) : (
                  catalog.relationships.map((relationship) => (
                    <RowButton
                      key={relationship.relationship_id}
                      onEdit={() => editRelationship(relationship)}
                      onDelete={() => void removeRelationship(relationship.relationship_id)}
                    >
                      <span className="flex items-center gap-1 truncate font-medium">
                        <Link2 className="h-3 w-3 shrink-0" />
                        {relationship.from_table}.{relationship.from_column} → {relationship.to_table}.{relationship.to_column}
                      </span>
                      <span className="block truncate text-[11px] text-muted-foreground">{valueLabel(relationship.cardinality)}</span>
                    </RowButton>
                  ))
                )}
              </div>
              {isEditorOpen ? <div className="grid gap-3 rounded-xl border border-border/40 bg-background/25 p-3">
                <div className="grid grid-cols-2 gap-3">
                  <SelectField
                    label="Таблица-источник"
                    value={relationshipDraft.from_table}
                    options={tableOptions}
                    onChange={(value) => setRelationshipDraft((prev) => ({ ...prev, from_table: value, from_column: "" }))}
                  />
                  <SelectField
                    label="Колонка-источник"
                    value={relationshipDraft.from_column}
                    options={relationshipFromColumns.map((column) => column.name)}
                    onChange={(value) => setRelationshipDraft((prev) => ({ ...prev, from_column: value }))}
                  />
                  <SelectField
                    label="Таблица назначения"
                    value={relationshipDraft.to_table}
                    options={tableOptions}
                    onChange={(value) => setRelationshipDraft((prev) => ({ ...prev, to_table: value, to_column: "" }))}
                  />
                  <SelectField
                    label="Колонка назначения"
                    value={relationshipDraft.to_column}
                    options={relationshipToColumns.map((column) => column.name)}
                    onChange={(value) => setRelationshipDraft((prev) => ({ ...prev, to_column: value }))}
                  />
                  <SelectField
                    label="Кардинальность"
                    value={relationshipDraft.cardinality}
                    options={["many_to_one", "one_to_one"]}
                    onChange={(value) => setRelationshipDraft((prev) => ({ ...prev, cardinality: value as RelationshipDraft["cardinality"] }))}
                  />
                </div>
                <TextField label="Описание" value={relationshipDraft.description} onChange={(value) => setRelationshipDraft((prev) => ({ ...prev, description: value }))} />
                <SaveButton disabled={isSaving || !relationshipReady} isSaving={isSaving} label={editingRelationshipId ? "Сохранить связь" : "Добавить связь"} onClick={() => void saveRelationship()} />
              </div> : null}
            </>
          ) : null}
        </>
      )}
    </div>
  );
}

function Overview({ catalog }: { catalog: SemanticCatalog }) {
  const issues = [...catalog.validation.errors, ...catalog.validation.warnings];
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3 text-[12px] md:grid-cols-4">
        <MetaPill label="Сущности" value={catalog.entities.length} />
        <MetaPill label="Измерения" value={catalog.dimensions.length} />
        <MetaPill label="Факты" value={catalog.facts.length} />
        <MetaPill label="Качество" value={Math.round((catalog.validation.quality_score ?? 0) * 100)} />
      </div>
      {issues.length ? (
        <div className="space-y-1 rounded-xl border border-amber-500/30 bg-amber-500/10 p-2">
          {issues.slice(0, 5).map((issue, index) => (
            <p key={`${issue.code}-${index}`} className="text-[12px] text-amber-300">
              {issue.code}: {issue.message}
            </p>
          ))}
        </div>
      ) : null}
      <div className="max-h-36 space-y-2 overflow-y-auto">
        {catalog.tables.slice(0, 10).map((table) => (
          <div key={table.table_id} className="flex items-center justify-between gap-2 rounded-xl border border-border/40 bg-background/25 px-3 py-2 text-[12px]">
            <span className="truncate font-medium">{table.qualified_name}</span>
            <span className="shrink-0 text-muted-foreground">{valueLabel(table.semantic_role)} · {table.columns_count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ListHeader({ title, isOpen, onToggle }: { title: string; isOpen: boolean; onToggle: () => void }) {
  return (
    <div className="flex items-center justify-between text-[11px] font-bold uppercase tracking-[0.1em] text-muted-foreground">
      <span>{title}</span>
      <button
        type="button"
        onClick={onToggle}
        className="flex items-center gap-1 rounded-lg px-2 py-1 text-primary transition-colors hover:bg-primary/10"
      >
        {!isOpen ? <Plus className="h-3 w-3" /> : null}
        {isOpen ? "Скрыть" : "Добавить"}
      </button>
    </div>
  );
}

function RowButton({
  children,
  onEdit,
  onDelete,
}: {
  children: ReactNode;
  onEdit: () => void;
  onDelete: () => void;
}) {
  return (
    <div className="flex items-center gap-2 rounded-xl border border-border/40 bg-background/25 px-3 py-2 text-[13px]">
      <button type="button" onClick={onEdit} className="min-w-0 flex-1 text-left">
        {children}
      </button>
      <button
        type="button"
        onClick={onDelete}
        className="rounded-lg p-1.5 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
      >
        <Trash2 className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}

function SaveButton({
  disabled,
  isSaving,
  label,
  onClick,
}: {
  disabled: boolean | string;
  isSaving: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={Boolean(disabled)}
      className="flex h-11 items-center justify-center gap-2 rounded-xl bg-primary px-3 text-[13px] font-bold text-primary-foreground shadow-lg shadow-primary/10 disabled:opacity-50"
    >
      {isSaving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
      {label}
    </button>
  );
}

function columnsFor(catalog: SemanticCatalog | null, table: string) {
  return (catalog?.columns ?? []).filter((column) => column.table === table);
}

function splitCsv(value: string): string[] {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function valueLabel(value: string): string {
  return VALUE_LABELS[value] ?? value;
}

function InfoHint({ text }: { text: string }) {
  return (
    <span className="group relative inline-flex" onClick={(event) => event.stopPropagation()}>
      <span
        className="inline-flex h-[18px] w-[18px] items-center justify-center rounded-full border border-border/50 bg-background/40 text-muted-foreground transition-all hover:border-primary/50 hover:bg-primary/10 hover:text-primary"
        aria-label={text}
      >
        <HelpCircle className="h-3 w-3" />
      </span>
      <span className="pointer-events-none absolute left-1/2 top-6 z-50 w-64 -translate-x-1/2 rounded-xl border border-border/60 bg-popover/95 px-3 py-2 text-left text-[12px] font-medium normal-case leading-relaxed tracking-normal text-popover-foreground opacity-0 shadow-2xl shadow-black/20 backdrop-blur-xl transition-opacity group-hover:opacity-100">
        {text}
      </span>
    </span>
  );
}

function MetaPill({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-xl border border-border/40 bg-secondary/35 px-3 py-2">
      <div className="text-[10px] uppercase tracking-widest text-muted-foreground">{label}</div>
      <div className="mt-1 text-[16px] font-bold">{value}</div>
    </div>
  );
}

function TextField({
  label,
  value,
  onChange,
  disabled,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}) {
  return (
    <label className="space-y-2 text-[11px] font-bold uppercase tracking-[0.1em] text-muted-foreground">
      <span>{label}</span>
      <input
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        className="h-11 w-full rounded-xl border border-border/60 bg-secondary/70 px-3.5 text-[14px] font-medium normal-case outline-none transition-all focus:border-primary/50 focus:ring-4 focus:ring-primary/10 disabled:opacity-60"
      />
    </label>
  );
}

function SelectField({
  label,
  value,
  options,
  onChange,
  disabled,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
  disabled?: boolean;
}) {
  const safeOptions = options.includes(value) ? options : [value, ...options].filter(Boolean);
  return (
    <label className="space-y-2 text-[11px] font-bold uppercase tracking-[0.1em] text-muted-foreground">
      <span>{label}</span>
      <select
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        className="h-11 w-full rounded-xl border border-border/60 bg-secondary/70 px-3.5 text-[14px] font-medium normal-case outline-none transition-all focus:border-primary/50 focus:ring-4 focus:ring-primary/10 disabled:opacity-60"
      >
        {safeOptions.map((option) => (
          <option key={option || "__empty"} value={option}>
            {option ? valueLabel(option) : "—"}
          </option>
        ))}
      </select>
    </label>
  );
}
