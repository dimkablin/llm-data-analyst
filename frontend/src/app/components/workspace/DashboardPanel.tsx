import { useEffect, useMemo, useRef, useState } from "react";
import {
  CheckCircle2,
  Database,
  FileText,
  Grid2X2,
  LoaderCircle,
  MoreHorizontal,
  PenSquare,
  Pin,
  PlugZap,
  Plus,
  RefreshCcw,
  ShieldCheck,
  Trash2,
  Upload,
} from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { ArtifactSurface } from "./ArtifactSurface";
import {
  bindCsvSource,
  bindDbConnectionSource,
  clearSessionSource,
  createDbConnection,
  deleteDbConnection,
  listDbConnectionSchemas,
  listDbConnections,
  testDbConnection,
  updateDbConnection,
} from "../../lib/backend-api";
import type {
  ArtifactPayload,
  DBConnection,
  DBConnectionFormPayload,
  DBConnectionSchema,
  DBConnectionType,
  SessionSourceState,
} from "../../lib/backend-types";
import { summarizeError } from "../../lib/format";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";

type DashboardTab = "visualizations" | "sources";
type SourceSection = "db" | "csv";
type SecretMode = "keep" | "replace" | "clear";

type Props = {
  sessionId: string;
  artifacts: ArtifactPayload[];
  pinnedArtifactIds: string[];
  datasetName: string;
  hasDataset: boolean;
  activeSource: SessionSourceState;
  showCode: boolean;
  onUpload: (file: File) => Promise<void>;
  onRefreshSession: () => Promise<void>;
  onUnpinArtifact: (artifactId: string) => void;
};

type UploadItem = {
  name: string;
  date: string;
  status: "analyzed";
};

type ConnectionFormState = {
  name: string;
  dbType: DBConnectionType;
  host: string;
  port: string;
  database: string;
  username: string;
  password: string;
  secretMode: SecretMode;
  sslmode: string;
  secure: boolean;
};

const DEFAULT_FORM: ConnectionFormState = {
  name: "",
  dbType: "postgresql",
  host: "",
  port: "5432",
  database: "",
  username: "",
  password: "",
  secretMode: "replace",
  sslmode: "prefer",
  secure: false,
};

function defaultPortFor(dbType: DBConnectionType): string {
  return dbType === "clickhouse" ? "8123" : "5432";
}

function toFormState(connection?: DBConnection | null): ConnectionFormState {
  if (!connection) {
    return DEFAULT_FORM;
  }
  return {
    name: connection.name,
    dbType: connection.db_type,
    host: connection.host,
    port: connection.port ? String(connection.port) : defaultPortFor(connection.db_type),
    database: connection.database ?? "",
    username: connection.username ?? "",
    password: "",
    secretMode: "keep",
    sslmode:
      connection.db_type === "postgresql"
        ? String(connection.options_json?.sslmode ?? "prefer")
        : "prefer",
    secure: Boolean(connection.options_json?.secure),
  };
}

function buildPayload(form: ConnectionFormState, editing: boolean, existingConnection?: DBConnection | null): DBConnectionFormPayload {
  const existingSchema = typeof existingConnection?.options_json?.schema === "string"
    ? existingConnection.options_json.schema
    : null;
  const payload: DBConnectionFormPayload = {
    name: form.name.trim(),
    db_type: form.dbType,
    host: form.host.trim(),
    port: form.port.trim() ? Number(form.port.trim()) : null,
    database: form.database.trim() || null,
    username: form.username.trim() || null,
    options_json:
      form.dbType === "postgresql"
        ? { sslmode: form.sslmode, schema: existingSchema }
        : { secure: form.secure, schema: existingSchema },
  };

  if (!editing || form.secretMode === "replace") {
    payload.password = form.password || null;
  }
  if (editing && form.secretMode === "clear") {
    payload.clear_password = true;
  }
  return payload;
}

export function DashboardPanel(props: Props) {
  const {
    sessionId,
    artifacts,
    pinnedArtifactIds,
    datasetName,
    hasDataset,
    activeSource,
    showCode,
    onUpload,
    onRefreshSession,
    onUnpinArtifact,
  } = props;

  const inputRef = useRef<HTMLInputElement | null>(null);
  const dbListRef = useRef<HTMLDivElement | null>(null);
  const [activeTab, setActiveTab] = useState<DashboardTab>("visualizations");
  const [sourceSection, setSourceSection] = useState<SourceSection>("db");
  const [recentUploads, setRecentUploads] = useState<UploadItem[]>([]);
  const [connections, setConnections] = useState<DBConnection[]>([]);
  const [connectionsLoading, setConnectionsLoading] = useState(false);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingConnection, setEditingConnection] = useState<DBConnection | null>(null);
  const [form, setForm] = useState<ConnectionFormState>(DEFAULT_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [testingId, setTestingId] = useState<string | null>(null);
  const [connectionSchemas, setConnectionSchemas] = useState<Record<string, DBConnectionSchema[]>>({});
  const [schemasLoadingId, setSchemasLoadingId] = useState<string | null>(null);
  const [bindingId, setBindingId] = useState<string | null>(null);
  const [bindingCsv, setBindingCsv] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [clearingSource, setClearingSource] = useState(false);

  const pinnedSet = useMemo(() => new Set(pinnedArtifactIds), [pinnedArtifactIds]);
  const visibleArtifacts = useMemo(() => {
    if (pinnedArtifactIds.length === 0) {
      return artifacts;
    }
    return artifacts.filter((artifact) => pinnedSet.has(artifact.id));
  }, [artifacts, pinnedArtifactIds, pinnedSet]);

  const activeSourceText = useMemo(() => {
    if (!activeSource.source_type) {
      return null;
    }
    if (activeSource.source_type === "db_connection") {
      return {
        label: activeSource.source_label || "Подключение к БД",
        meta: `База данных${activeSource.source_mode ? ` · ${activeSource.source_mode}` : ""}`,
        status: "Подключен",
      };
    }
    return {
      label: activeSource.source_label || datasetName || "CSV dataset",
      meta: "CSV · Файл загружен",
      status: hasDataset ? "Готов" : "Не загружен",
    };
  }, [activeSource, datasetName, hasDataset]);

  async function loadConnections(): Promise<void> {
    setConnectionsLoading(true);
    setSourceError(null);
    try {
      setConnections(await listDbConnections());
    } catch (error) {
      setSourceError(summarizeError(error));
    } finally {
      setConnectionsLoading(false);
    }
  }

  useEffect(() => {
    if (activeTab === "sources") {
      void loadConnections();
    }
  }, [activeTab]);

  useEffect(() => {
    if (activeSource.source_type === "csv") {
      setSourceSection("csv");
      return;
    }
    if (activeSource.source_type === "db_connection") {
      setSourceSection("db");
    }
  }, [activeSource.source_type]);

  useEffect(() => {
    if (!datasetName || activeSource.source_type !== "csv") {
      return;
    }
    setRecentUploads((prev) => {
      const next = [{ name: datasetName, date: new Date().toLocaleDateString("ru-RU"), status: "analyzed" as const }, ...prev];
      const deduped = next.filter((item, index, array) => array.findIndex((row) => row.name === item.name) === index);
      return deduped.slice(0, 6);
    });
  }, [activeSource.source_type, datasetName]);

  function resetForm(next?: DBConnection | null): void {
    setEditingConnection(next ?? null);
    setForm(toFormState(next ?? null));
    setSourceError(null);
  }

  async function handleFilePick(file: File): Promise<void> {
    await onUpload(file);
    setActiveTab("sources");
  }

  async function handleOpenSavedConnections(): Promise<void> {
    setSourceSection("db");
    await loadConnections();
    requestAnimationFrame(() => {
      dbListRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  function handleSwitchSource(): void {
    const nextSection = activeSource.source_type === "csv" ? "csv" : "db";
    setSourceSection(nextSection);
    requestAnimationFrame(() => {
      if (nextSection === "db") {
        dbListRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  }

  async function handleSubmitConnection(): Promise<void> {
    setSubmitting(true);
    setSourceError(null);
    try {
      const payload = buildPayload(form, Boolean(editingConnection), editingConnection);
      if (editingConnection) {
        await updateDbConnection(editingConnection.id, payload);
      } else {
        await createDbConnection(payload);
      }
      setDialogOpen(false);
      resetForm(null);
      await loadConnections();
    } catch (error) {
      setSourceError(summarizeError(error));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDeleteConnection(connection: DBConnection): Promise<void> {
    if (!window.confirm(`Удалить подключение "${connection.name}"?`)) {
      return;
    }
    setDeletingId(connection.id);
    setSourceError(null);
    try {
      await deleteDbConnection(connection.id);
      await loadConnections();
    } catch (error) {
      setSourceError(summarizeError(error));
    } finally {
      setDeletingId(null);
    }
  }

  async function handleTestConnection(connection: DBConnection): Promise<void> {
    setTestingId(connection.id);
    try {
      const result = await testDbConnection(connection.id);
      await loadConnections();
      if (result.last_test_ok) {
        void loadSchemasForConnection(connection.id);
      }
    } catch (error) {
      await loadConnections();
    } finally {
      setTestingId(null);
    }
  }

  async function loadSchemasForConnection(connectionId: string): Promise<void> {
    setSchemasLoadingId(connectionId);
    try {
      const schemas = await listDbConnectionSchemas(connectionId);
      setConnectionSchemas((prev) => ({ ...prev, [connectionId]: schemas }));
    } catch {
      // ignore — schemas will remain unavailable
    } finally {
      setSchemasLoadingId((prev) => (prev === connectionId ? null : prev));
    }
  }

  async function handleSelectSchema(connection: DBConnection, schema: string): Promise<void> {
    try {
      const updatedOptionsJson = { ...(connection.options_json ?? {}), schema: schema || null };
      await updateDbConnection(connection.id, { options_json: updatedOptionsJson });
      await loadConnections();
    } catch (error) {
      setSourceError(summarizeError(error));
    }
  }

  async function handleBindConnection(connection: DBConnection): Promise<void> {
    if (!sessionId) {
      return;
    }
    setBindingId(connection.id);
    setSourceError(null);
    try {
      await bindDbConnectionSource(sessionId, connection.id);
      await onRefreshSession();
      await loadConnections();
    } catch (error) {
      setSourceError(summarizeError(error));
    } finally {
      setBindingId(null);
    }
  }

  async function handleClearSource(): Promise<void> {
    if (!sessionId) {
      return;
    }
    setClearingSource(true);
    setSourceError(null);
    try {
      await clearSessionSource(sessionId);
      await onRefreshSession();
    } catch (error) {
      setSourceError(summarizeError(error));
    } finally {
      setClearingSource(false);
    }
  }

  async function handleBindCsvSource(): Promise<void> {
    if (!sessionId || !hasDataset || !datasetName) {
      return;
    }
    setBindingCsv(true);
    setSourceError(null);
    try {
      await bindCsvSource(sessionId);
      await onRefreshSession();
    } catch (error) {
      setSourceError(summarizeError(error));
    } finally {
      setBindingCsv(false);
    }
  }

  const isEditing = Boolean(editingConnection);

  return (
    <div className="flex h-full min-h-0 flex-col space-y-6">
      <div className="flex items-center justify-between">
        <div className="inline-flex items-center rounded-[18px] border border-border/50 bg-secondary/40 p-1">
          <button
            type="button"
            onClick={() => setActiveTab("visualizations")}
            className={`inline-flex items-center gap-2 rounded-[14px] px-4 py-2.5 text-sm font-bold transition-all ${
              activeTab === "visualizations"
                ? "bg-card text-foreground shadow-sm ring-1 ring-border/20"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Grid2X2 className="h-4 w-4" />
            Визуализации
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("sources")}
            className={`inline-flex items-center gap-2 rounded-[14px] px-4 py-2.5 text-sm font-bold transition-all ${
              activeTab === "sources"
                ? "bg-card text-foreground shadow-sm ring-1 ring-white"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Upload className="h-4 w-4" />
            Источники
          </button>
        </div>

        <div className="flex items-center gap-2">
          {activeTab === "sources" ? (
            <button
              type="button"
              onClick={() => {
                resetForm(null);
                setDialogOpen(true);
              }}
              className="inline-flex h-12 items-center gap-2 rounded-2xl border border-border/50 bg-secondary/60 px-4 text-sm font-bold text-foreground transition-all hover:bg-muted"
            >
              <Database className="h-4 w-4" />
              Подключение
            </button>
          ) : null}
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="flex h-12 w-12 items-center justify-center rounded-2xl border border-border/50 bg-secondary/60 text-muted-foreground transition-all hover:bg-muted hover:text-foreground"
          >
            <Plus className="h-5 w-5" />
          </button>
        </div>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept=".csv,text/csv"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) {
            void handleFilePick(file);
          }
          event.currentTarget.value = "";
        }}
      />

      <div className="min-h-0 flex-1">
        <AnimatePresence mode="wait">
          {activeTab === "visualizations" ? (
            <motion.div
              key="visualizations"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="flex h-full min-h-0 flex-col"
            >
              {visibleArtifacts.length === 0 ? (
                <div className="flex min-h-[420px] items-center justify-center rounded-[28px] border border-dashed border-border/30 bg-card/35 p-10 text-center shadow-sm">
                  <div className="max-w-[520px]">
                    <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-3xl bg-primary/10 text-primary">
                      <Grid2X2 className="h-8 w-8" />
                    </div>
                    <h3 className="mb-3 text-2xl font-bold tracking-tight">Пока пусто</h3>
                    <p className="text-[15px] leading-relaxed text-muted-foreground">
                      Добавляйте артефакты в дашборд кнопкой в чате рядом с каждым артефактом.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="custom-scrollbar min-h-0 flex-1 overflow-y-auto pr-2">
                  <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
                    {visibleArtifacts.map((artifact) => (
                      <motion.div key={artifact.id} layout className="min-w-0" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
                        <div className="mb-2 flex justify-end">
                          {pinnedSet.has(artifact.id) ? (
                            <button
                              type="button"
                              onClick={() => onUnpinArtifact(artifact.id)}
                              className="inline-flex items-center gap-2 rounded-full border border-border/50 bg-secondary px-3 py-1 text-xs font-bold uppercase tracking-wider text-muted-foreground"
                            >
                              <Pin className="h-3.5 w-3.5" />
                              Открепить
                            </button>
                          ) : null}
                        </div>
                        <ArtifactSurface artifact={artifact} showCode={showCode} />
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          ) : (
            <motion.div
              key="sources"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="custom-scrollbar flex h-full min-h-0 flex-col space-y-6 overflow-y-auto pr-2"
            >
              <div className="grid gap-4 xl:grid-cols-2">
                <div className="flex min-h-[240px] flex-col items-center justify-center rounded-[28px] border border-dashed border-border/30 bg-card/35 p-7 text-center shadow-sm">
                  <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-3xl bg-primary/10 text-primary">
                    <Upload className="h-7 w-7" />
                  </div>
                  <h3 className="text-xl font-bold tracking-tight">CSV</h3>
                  <p className="mt-2 max-w-sm text-sm leading-relaxed text-muted-foreground">
                    Загрузите CSV-файл и используйте его как источник данных.
                  </p>
                  <button
                    type="button"
                    onClick={() => inputRef.current?.click()}
                    className="mt-6 inline-flex items-center justify-center gap-2 rounded-2xl bg-primary px-5 py-3 font-bold text-primary-foreground shadow-lg shadow-primary/20"
                  >
                    <Upload className="h-4 w-4" />
                    Загрузить CSV
                  </button>
                </div>

                <div className="flex min-h-[240px] flex-col items-center justify-center rounded-[28px] border border-dashed border-border/30 bg-card/35 p-7 text-center shadow-sm">
                  <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-3xl bg-sky-500/10 text-sky-500">
                    <Database className="h-7 w-7" />
                  </div>
                  <h3 className="text-xl font-bold tracking-tight">База данных</h3>
                  <p className="mt-2 max-w-sm text-sm leading-relaxed text-muted-foreground">
                    Подключите PostgreSQL или ClickHouse и выберите сохраненное соединение.
                  </p>
                  <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
                    <button
                      type="button"
                      onClick={() => void handleOpenSavedConnections()}
                      className="inline-flex items-center justify-center gap-2 rounded-2xl bg-sky-500 px-5 py-3 font-bold text-sky-950 shadow-lg shadow-sky-500/20"
                    >
                      <Database className="h-4 w-4" />
                      Выбрать подключение
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        resetForm(null);
                        setDialogOpen(true);
                      }}
                      className="inline-flex items-center justify-center gap-2 rounded-2xl border border-border/50 bg-card/70 px-5 py-3 font-bold text-foreground transition-all hover:bg-muted"
                    >
                      <Plus className="h-4 w-4" />
                      Новое
                    </button>
                  </div>
                </div>
              </div>

              <div className="rounded-[22px] border border-border/50 bg-secondary/20 px-5 py-4">
                <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                  <div className="min-w-0">
                    <div className="text-[12px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                      Текущий источник
                    </div>
                    <div className="mt-1 truncate text-base font-bold text-foreground">
                      {activeSourceText?.label ?? "Источник не выбран"}
                    </div>
                    <div className="mt-1 text-sm text-muted-foreground">
                      {activeSourceText ? `${activeSourceText.meta} · ${activeSourceText.status}` : "Источник данных пока не выбран"}
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      onClick={handleSwitchSource}
                      className="inline-flex items-center gap-2 rounded-xl border border-border/50 bg-card/70 px-3 py-2 text-xs font-bold text-foreground transition-all hover:bg-muted"
                    >
                      Сменить
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleClearSource()}
                      disabled={clearingSource || !activeSource.source_type}
                      className="inline-flex items-center gap-2 rounded-xl border border-border/50 bg-card/70 px-3 py-2 text-xs font-bold text-muted-foreground transition-all hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {clearingSource ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : null}
                      Очистить
                    </button>
                  </div>
                </div>
              </div>

              <div className="inline-flex w-fit items-center rounded-[18px] border border-border/50 bg-secondary/40 p-1">
                <button
                  type="button"
                  onClick={() => setSourceSection("db")}
                  className={`inline-flex items-center gap-2 rounded-[14px] px-4 py-2.5 text-sm font-bold transition-all ${
                    sourceSection === "db"
                      ? "bg-card text-foreground shadow-sm ring-1 ring-border/20 dark:ring-white"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <Database className="h-4 w-4" />
                  Подключения к БД
                </button>
                <button
                  type="button"
                  onClick={() => setSourceSection("csv")}
                  className={`inline-flex items-center gap-2 rounded-[14px] px-4 py-2.5 text-sm font-bold transition-all ${
                    sourceSection === "csv"
                      ? "bg-card text-foreground shadow-sm ring-1 ring-border/20 dark:ring-white"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <FileText className="h-4 w-4" />
                  Загрузки CSV
                </button>
              </div>

              {sourceSection === "db" ? (
              <div ref={dbListRef} className="rounded-[28px] border border-border/50 bg-card/45 p-6 shadow-sm">
                  <div className="mb-5 flex items-center justify-between">
                    <div>
                      <div className="text-[12px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                        DB Connections
                      </div>
                      <h3 className="mt-2 text-xl font-bold tracking-tight">Подключения к БД</h3>
                    </div>
                    <button
                      type="button"
                      onClick={() => void loadConnections()}
                      className="inline-flex h-10 w-10 items-center justify-center rounded-2xl border border-border/50 bg-secondary/70 text-muted-foreground transition-all hover:text-foreground"
                    >
                      <RefreshCcw className={`h-4 w-4 ${connectionsLoading ? "animate-spin" : ""}`} />
                    </button>
                  </div>

                  <div className="space-y-3">
                    {connectionsLoading ? (
                      <div className="flex items-center gap-3 rounded-[22px] border border-border/50 bg-secondary/30 px-4 py-4 text-sm text-muted-foreground">
                        <LoaderCircle className="h-4 w-4 animate-spin" />
                        Загружаю подключения...
                      </div>
                    ) : (
                      connections.map((connection) => {
                        const isActive =
                          activeSource.source_type === "db_connection" &&
                          activeSource.source_ref_id === connection.id;
                        const testedOk = connection.last_test_ok === true;
                        const hasSchema = typeof connection.options_json?.schema === "string" && connection.options_json.schema.trim().length > 0;
                        const canUse = testedOk && hasSchema;
                        return (
                          <div key={connection.id} className="rounded-[22px] border border-border/50 bg-secondary/20 px-4 py-4">
                            {/* Header: info + three-dots menu */}
                            <div className="flex items-start justify-between gap-3">
                              <div className="min-w-0">
                                <div className="flex flex-wrap items-center gap-2">
                                  <span className="text-sm font-bold text-foreground">{connection.name}</span>
                                  <span className="rounded-full border border-border/50 bg-card/70 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                                    {connection.db_type}
                                  </span>
                                  {isActive ? <span className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-emerald-400">active</span> : null}
                                </div>
                                <div className="mt-1 truncate text-xs text-muted-foreground">
                                  {connection.host}{connection.port ? `:${connection.port}` : ""}{connection.database ? ` / ${connection.database}` : ""}
                                </div>
                                <div className="mt-1.5 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                                  <span className="inline-flex items-center gap-1">
                                    <ShieldCheck className="h-3 w-3" />
                                    {connection.password_present ? "секрет сохранен" : "секрет не задан"}
                                  </span>
                                  {testedOk ? (
                                    <span className="inline-flex items-center gap-1 text-emerald-500">
                                      <CheckCircle2 className="h-3 w-3" />
                                      connection ok
                                    </span>
                                  ) : null}
                                </div>
                              </div>

                              <DropdownMenu>
                                <DropdownMenuTrigger asChild>
                                  <button type="button" className="inline-flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-xl border border-border/50 bg-card/70 text-muted-foreground transition-all hover:text-foreground">
                                    <MoreHorizontal className="h-4 w-4" />
                                  </button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent align="end" className="min-w-[150px]">
                                  <DropdownMenuItem
                                    onClick={() => void handleTestConnection(connection)}
                                    disabled={testingId === connection.id}
                                    className="gap-2 text-xs"
                                  >
                                    {testingId === connection.id ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
                                    Тест
                                  </DropdownMenuItem>
                                  <DropdownMenuItem
                                    onClick={() => { resetForm(connection); setDialogOpen(true); }}
                                    className="gap-2 text-xs"
                                  >
                                    <PenSquare className="h-3.5 w-3.5" />
                                    Изменить
                                  </DropdownMenuItem>
                                  <DropdownMenuSeparator />
                                  <DropdownMenuItem
                                    onClick={() => void handleDeleteConnection(connection)}
                                    disabled={deletingId === connection.id}
                                    className="gap-2 text-xs text-rose-400 focus:text-rose-400"
                                  >
                                    {deletingId === connection.id ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                                    Удалить
                                  </DropdownMenuItem>
                                </DropdownMenuContent>
                              </DropdownMenu>
                            </div>

                            {/* Action row: Use button + schema select */}
                            <div className="mt-3 flex items-center gap-2">
                              <button
                                type="button"
                                onClick={() => void handleBindConnection(connection)}
                                disabled={!sessionId || bindingId === connection.id || !canUse}
                                className="inline-flex items-center gap-1.5 rounded-xl border border-border/50 bg-card/70 px-3 py-2 text-xs font-bold text-foreground transition-all hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
                              >
                                {bindingId === connection.id ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <PlugZap className="h-3.5 w-3.5" />}
                                Использовать
                              </button>

                              <div className="ml-auto flex items-center gap-1.5">
                                {testedOk ? (
                                  <>
                                    <button
                                      type="button"
                                      onClick={() => void loadSchemasForConnection(connection.id)}
                                      disabled={schemasLoadingId === connection.id}
                                      className="inline-flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-xl border border-border/50 bg-card/60 text-muted-foreground transition-all hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
                                    >
                                      <RefreshCcw className={`h-3.5 w-3.5 ${schemasLoadingId === connection.id ? "animate-spin" : ""}`} />
                                    </button>
                                    <Select
                                      value={typeof connection.options_json?.schema === "string" ? connection.options_json.schema : ""}
                                      onValueChange={(value) => void handleSelectSchema(connection, value)}
                                      disabled={schemasLoadingId === connection.id}
                                    >
                                      <SelectTrigger className="h-8 w-44 rounded-xl border border-border/50 bg-card/60 px-3 text-xs text-muted-foreground">
                                        <SelectValue placeholder="Выберите схему" />
                                      </SelectTrigger>
                                      <SelectContent>
                                        {!connectionSchemas[connection.id] && typeof connection.options_json?.schema === "string" && connection.options_json.schema ? (
                                          <SelectItem value={connection.options_json.schema}>{connection.options_json.schema}</SelectItem>
                                        ) : null}
                                        {(connectionSchemas[connection.id] ?? []).map((s) => (
                                          <SelectItem key={s.name} value={s.name}>{s.display_name}</SelectItem>
                                        ))}
                                      </SelectContent>
                                    </Select>
                                  </>
                                ) : (
                                  <span className="text-[11px] text-muted-foreground/60">Сначала запустите тест</span>
                                )}
                              </div>
                            </div>

                            {connection.last_error ? (
                              <div className="mt-3 rounded-2xl border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-200">
                                {connection.last_error}
                              </div>
                            ) : null}
                          </div>
                        );
                      })
                    )}
                    {!connectionsLoading && connections.length === 0 ? (
                      <div className="rounded-[22px] border border-border/50 bg-secondary/20 px-4 py-5 text-sm text-muted-foreground">
                        Пока нет сохраненных подключений к БД.
                      </div>
                    ) : null}
                  </div>
              </div>
              ) : (
              <div className="rounded-[28px] border border-border/50 bg-card/45 p-6 shadow-sm">
                <div className="mb-4 text-[12px] font-bold uppercase tracking-[0.18em] text-muted-foreground">
                  Последние загрузки
                </div>
                {hasDataset ? (
                  <div className="mb-4 rounded-[24px] border border-border/50 bg-secondary/20 px-5 py-4">
                    <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                      <div>
                        <div className="text-[15px] font-bold text-foreground">
                          {datasetName || "CSV dataset"}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {activeSource.source_type === "csv"
                            ? "CSV уже выбран как активный источник"
                            : "Файл уже загружен в сессию и доступен для повторного выбора"}
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => void handleBindCsvSource()}
                        disabled={
                          bindingCsv ||
                          !sessionId ||
                          !datasetName ||
                          activeSource.source_type === "csv"
                        }
                        className="inline-flex items-center gap-2 rounded-xl border border-border/50 bg-card/70 px-3 py-2 text-xs font-bold text-foreground transition-all hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        {bindingCsv ? (
                          <LoaderCircle className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <FileText className="h-3.5 w-3.5" />
                        )}
                        Использовать CSV
                      </button>
                    </div>
                  </div>
                ) : null}
                <div className="space-y-4">
                  {recentUploads.length > 0 ? (
                    recentUploads.map((item) => (
                      <div key={`${item.name}-${item.date}`} className="flex items-center justify-between rounded-[24px] border border-border/50 bg-card/50 px-5 py-5 shadow-sm">
                        <div className="flex items-center gap-4">
                          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-secondary/70 text-muted-foreground">
                            <FileText className="h-5 w-5" />
                          </div>
                          <div>
                            <div className="text-[15px] font-bold">{item.name}</div>
                            <div className="text-sm text-muted-foreground">{item.date}</div>
                          </div>
                        </div>
                        <div className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1 text-[11px] font-bold uppercase tracking-wider text-emerald-400">
                          analyzed
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="rounded-[24px] border border-border/50 bg-card/40 px-5 py-6 text-sm text-muted-foreground">
                      Пока нет загруженных CSV файлов.
                    </div>
                  )}
                </div>
              </div>
              )}

              {sourceError ? (
                <div className="rounded-[24px] border border-rose-500/20 bg-rose-500/10 px-5 py-4 text-sm text-rose-700 dark:text-rose-200">
                  {sourceError}
                </div>
              ) : null}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <Dialog open={dialogOpen} onOpenChange={(nextOpen) => {
        setDialogOpen(nextOpen);
        if (!nextOpen) {
          resetForm(null);
        }
      }}>
        <DialogContent className="max-w-2xl rounded-[28px] border-border/60 bg-background/98 p-0 backdrop-blur-xl">
          <div className="p-7">
            <DialogHeader className="mb-6 text-left">
              <DialogTitle className="text-2xl font-bold tracking-tight">
                {isEditing ? "Изменить подключение" : "Новое подключение"}
              </DialogTitle>
              <DialogDescription className="text-sm leading-relaxed">
                Секрет хранится только на backend. После сохранения пароль больше не возвращается в интерфейс.
              </DialogDescription>
            </DialogHeader>

            <div className="grid gap-4 md:grid-cols-2">
              <label className="space-y-2">
                <span className="text-[12px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Название</span>
                <input value={form.name} onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))} className="h-12 w-full rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm outline-none transition focus:border-primary/40" placeholder="Sales Warehouse" />
              </label>

              <label className="space-y-2">
                <span className="text-[12px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Тип БД</span>
                <Select value={form.dbType} onValueChange={(value) => setForm((prev) => ({ ...prev, dbType: value as DBConnectionType, port: defaultPortFor(value as DBConnectionType) }))}>
                  <SelectTrigger className="h-12 rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="postgresql">PostgreSQL</SelectItem>
                    <SelectItem value="clickhouse">ClickHouse</SelectItem>
                  </SelectContent>
                </Select>
              </label>

              <label className="space-y-2">
                <span className="text-[12px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Host</span>
                <input value={form.host} onChange={(event) => setForm((prev) => ({ ...prev, host: event.target.value }))} className="h-12 w-full rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm outline-none transition focus:border-primary/40" placeholder="db.example.com" />
              </label>

              <label className="space-y-2">
                <span className="text-[12px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Port</span>
                <input value={form.port} onChange={(event) => setForm((prev) => ({ ...prev, port: event.target.value }))} className="h-12 w-full rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm outline-none transition focus:border-primary/40" placeholder={defaultPortFor(form.dbType)} />
              </label>

              <label className="space-y-2">
                <span className="text-[12px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Database</span>
                <input value={form.database} onChange={(event) => setForm((prev) => ({ ...prev, database: event.target.value }))} className="h-12 w-full rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm outline-none transition focus:border-primary/40" placeholder={form.dbType === "clickhouse" ? "default" : "analytics"} />
              </label>

              <label className="space-y-2">
                <span className="text-[12px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Username</span>
                <input value={form.username} onChange={(event) => setForm((prev) => ({ ...prev, username: event.target.value }))} className="h-12 w-full rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm outline-none transition focus:border-primary/40" placeholder="readonly_user" />
              </label>

              {form.dbType === "postgresql" ? (
                <label className="space-y-2">
                  <span className="text-[12px] font-bold uppercase tracking-[0.16em] text-muted-foreground">SSL mode</span>
                  <Select value={form.sslmode} onValueChange={(value) => setForm((prev) => ({ ...prev, sslmode: value }))}>
                    <SelectTrigger className="h-12 rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="disable">disable</SelectItem>
                      <SelectItem value="prefer">prefer</SelectItem>
                      <SelectItem value="require">require</SelectItem>
                    </SelectContent>
                  </Select>
                </label>
              ) : (
                <label className="space-y-2">
                  <span className="text-[12px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Transport</span>
                  <Select value={form.secure ? "https" : "http"} onValueChange={(value) => setForm((prev) => ({ ...prev, secure: value === "https" }))}>
                    <SelectTrigger className="h-12 rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="http">HTTP</SelectItem>
                      <SelectItem value="https">HTTPS</SelectItem>
                    </SelectContent>
                  </Select>
                </label>
              )}

              <div className="space-y-2 md:col-span-2">
                <span className="text-[12px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Секрет</span>
                {isEditing ? (
                  <div className="grid gap-3 md:grid-cols-[220px_1fr]">
                    <Select value={form.secretMode} onValueChange={(value) => setForm((prev) => ({ ...prev, secretMode: value as SecretMode, password: value === "replace" ? prev.password : "" }))}>
                      <SelectTrigger className="h-12 rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="keep">Оставить как есть</SelectItem>
                        <SelectItem value="replace">Заменить секрет</SelectItem>
                        <SelectItem value="clear">Очистить секрет</SelectItem>
                      </SelectContent>
                    </Select>
                    <input value={form.password} onChange={(event) => setForm((prev) => ({ ...prev, password: event.target.value, secretMode: prev.secretMode === "keep" ? "replace" : prev.secretMode }))} disabled={form.secretMode !== "replace"} type="password" className="h-12 w-full rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm outline-none transition focus:border-primary/40 disabled:cursor-not-allowed disabled:opacity-60" placeholder={editingConnection?.password_present ? "Секрет сохранен на backend" : "Введите пароль"} />
                  </div>
                ) : (
                  <input value={form.password} onChange={(event) => setForm((prev) => ({ ...prev, password: event.target.value }))} type="password" className="h-12 w-full rounded-2xl border border-border/60 bg-secondary/40 px-4 text-sm outline-none transition focus:border-primary/40" placeholder="Введите пароль или токен" />
                )}
              </div>
            </div>

            {sourceError ? (
              <div className="mt-5 rounded-2xl border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-200">
                {sourceError}
              </div>
            ) : null}

            <DialogFooter className="mt-6">
              <button type="button" onClick={() => { setDialogOpen(false); resetForm(null); }} className="inline-flex h-11 items-center justify-center rounded-2xl border border-border/50 px-5 text-sm font-bold text-muted-foreground transition-all hover:text-foreground">
                Отмена
              </button>
              <button type="button" disabled={submitting || !form.name.trim() || !form.host.trim()} onClick={() => void handleSubmitConnection()} className="inline-flex h-11 items-center justify-center gap-2 rounded-2xl bg-primary px-5 text-sm font-bold text-primary-foreground shadow-lg shadow-primary/20 disabled:cursor-not-allowed disabled:opacity-60">
                {submitting ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Database className="h-4 w-4" />}
                {isEditing ? "Сохранить изменения" : "Создать подключение"}
              </button>
            </DialogFooter>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
