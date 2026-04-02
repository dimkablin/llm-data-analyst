import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router";
import {
  BarChart3,
  Bot,
  CalendarDays,
  Cpu,
  Database,
  Download,
  LayoutGrid,
  MessageSquare,
  Plus,
  Search,
  Star,
  Table2,
  Trash2,
} from "lucide-react";
import { motion } from "motion/react";
import { Navigation } from "../components/Navigation";
import {
  createSession,
  deleteSession,
  getSession,
  getRuntimeModelProfile,
  listSessions,
  updateSessionTitle,
} from "../lib/backend-api";
import { exportChatHistory } from "../lib/chat-export";
import type { RuntimeModelProfile, SessionState, SessionSummary } from "../lib/backend-types";
import { formatDateTime, summarizeError } from "../lib/format";
import { useAppSession } from "../context/AppSessionContext";

type SessionFilter = "all" | "active" | "favorites" | "archive" | "errors";

function getActiveSessionStorageKey(userId: number | undefined): string {
  return userId
    ? `llm_new_frontend_active_session_${userId}`
    : "llm_new_frontend_active_session";
}

const FILTERS: Array<{ id: SessionFilter; label: string }> = [
  { id: "all", label: "Все" },
  { id: "active", label: "Активные" },
  { id: "favorites", label: "Избранные" },
  { id: "archive", label: "Архив" },
  { id: "errors", label: "Ошибки" },
];

export function Sessions() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAppSession();
  const [search, setSearch] = useState("");
  const [activeFilter, setActiveFilter] = useState<SessionFilter>("all");
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedState, setSelectedState] = useState<SessionState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [modelProfile, setModelProfile] = useState<RuntimeModelProfile | null>(null);
  const [favoriteIds, setFavoriteIds] = useState<string[]>(() => {
    try {
      const raw = window.localStorage.getItem("llm_new_frontend_favorite_sessions");
      return raw ? (JSON.parse(raw) as string[]) : [];
    } catch {
      return [];
    }
  });
  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!user) {
      return;
    }
    void Promise.all([refreshSessions(), getRuntimeModelProfile().catch(() => null)]).then(([, model]) => {
      if (model) {
        setModelProfile(model);
      }
    });
  }, [user]);

  useEffect(() => {
    if (!selectedId) {
      return;
    }
    void getSession(selectedId)
      .then((state) => {
        setSelectedState(state);
        setRenameDraft(state.title || "");
      })
      .catch((loadError) => setError(summarizeError(loadError)));
  }, [selectedId]);

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    if (params.get("focus") !== "search") {
      return;
    }

    const timer = window.setTimeout(() => {
      searchInputRef.current?.focus();
      searchInputRef.current?.select();
    }, 50);

    return () => window.clearTimeout(timer);
  }, [location.search]);

  useEffect(() => {
    window.localStorage.setItem("llm_new_frontend_favorite_sessions", JSON.stringify(favoriteIds));
  }, [favoriteIds]);

  if (!user) {
    return <Navigate to="/auth" replace />;
  }

  async function refreshSessions(): Promise<void> {
    try {
      const rows = await listSessions();
      setSessions(rows);
      if (!selectedId && rows[0]) {
        setSelectedId(rows[0].session_id);
      }
    } catch (loadError) {
      setError(summarizeError(loadError));
    }
  }

  async function handleCreateSession(): Promise<void> {
    try {
      const sessionId = await createSession(false);
      await refreshSessions();
      navigate("/workspace", { replace: false });
      window.localStorage.setItem(getActiveSessionStorageKey(user?.id), sessionId);
    } catch (createError) {
      setError(summarizeError(createError));
    }
  }

  async function handleDeleteSession(sessionId: string): Promise<void> {
    try {
      await deleteSession(sessionId);
      if (selectedId === sessionId) {
        setSelectedId(null);
        setSelectedState(null);
      }
      await refreshSessions();
    } catch (deleteError) {
      setError(summarizeError(deleteError));
    }
  }

  async function handleRename(): Promise<void> {
    if (!selectedId || !renameDraft.trim()) {
      return;
    }
    try {
      const updated = await updateSessionTitle(selectedId, renameDraft.trim());
      setSessions((prev) => prev.map((item) => (item.session_id === updated.session_id ? updated : item)));
      setSelectedState((prev) => (prev ? { ...prev, title: updated.title } : prev));
    } catch (renameError) {
      setError(summarizeError(renameError));
    }
  }

  const filtered = useMemo(() => {
    const now = Date.now();

    return sessions.filter((session) => {
      const haystack = `${session.title} ${session.last_message_preview || ""}`.toLowerCase();
      const matchesSearch = haystack.includes(search.toLowerCase());
      if (!matchesSearch) return false;

      const isFavorite = favoriteIds.includes(session.session_id);
      const lastAccessTime = new Date(session.last_access).getTime();
      const isRecent = Number.isFinite(lastAccessTime) && now - lastAccessTime < 1000 * 60 * 60 * 24 * 3;
      const isArchived = Number.isFinite(lastAccessTime) && now - lastAccessTime > 1000 * 60 * 60 * 24 * 14;
      const hasErrorSignal = /error|ошиб|failed|traceback|exception/i.test(
        `${session.title} ${session.last_message_preview || ""}`,
      );

      switch (activeFilter) {
        case "active":
          return isRecent || session.has_dataset;
        case "favorites":
          return isFavorite;
        case "archive":
          return isArchived;
        case "errors":
          return hasErrorSignal;
        case "all":
        default:
          return true;
      }
    });
  }, [activeFilter, favoriteIds, search, sessions]);

  function toggleFavorite(sessionId: string): void {
    setFavoriteIds((current) =>
      current.includes(sessionId)
        ? current.filter((id) => id !== sessionId)
        : [...current, sessionId],
    );
  }

  function getPinnedCount(sessionId: string): number {
    try {
      const raw = window.localStorage.getItem(`llm_new_frontend_pinned_artifacts_${sessionId}`);
      if (!raw) return 0;
      const parsed = JSON.parse(raw) as string[];
      return Array.isArray(parsed) ? parsed.length : 0;
    } catch {
      return 0;
    }
  }

  const modelLabel = modelProfile?.model || modelProfile?.provider || "runtime model";

  function handleExportSelectedSession(): void {
    if (!selectedState) {
      return;
    }
    exportChatHistory(
      selectedState.session_id,
      selectedState.title || "Сессия",
      String(selectedState.source_label || ""),
      selectedState.chat_history,
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground font-sans">
      <Navigation />

      <main className="mx-auto max-w-[1400px] px-4 py-6 sm:px-6 lg:px-8 lg:py-8 xl:py-12">
        <div className="mb-6 flex items-end justify-between lg:mb-8 xl:mb-12">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
            <div className="inline-flex items-center gap-2.5 rounded-full border border-primary/20 bg-primary/10 px-3.5 py-1.5 text-primary">
              <MessageSquare className="h-4 w-4" />
              <span className="text-[13px] font-semibold uppercase tracking-wide">Архив аналитики</span>
            </div>
            <h1 className="text-2xl font-bold tracking-tight lg:text-3xl xl:text-4xl">Сессии аналитика</h1>
            <p className="max-w-xl text-[17px] leading-relaxed text-muted-foreground">
              Централизованное хранилище ваших диалогов с AI, сгенерированных графиков и аналитических отчетов.
            </p>
          </motion.div>

          <button
            type="button"
            onClick={() => void handleCreateSession()}
            className="flex items-center gap-2.5 rounded-2xl bg-primary px-6 py-3.5 font-semibold text-primary-foreground shadow-xl shadow-primary/20"
          >
            <Plus className="h-5 w-5" />
            Новая сессия
          </button>
        </div>

        {error ? <div className="mb-6 rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">{error}</div> : null}

        <div className="mb-6 grid grid-cols-1 gap-4 md:grid-cols-4 lg:gap-5 xl:mb-8">
          <StatCard label="Всего сессий" value={String(sessions.length)} icon={<MessageSquare className="h-5 w-5" />} />
          <StatCard label="С датасетом" value={String(sessions.filter((item) => item.has_dataset).length)} icon={<Table2 className="h-5 w-5" />} />
          <StatCard label="Последний доступ" value={sessions[0] ? formatDateTime(sessions[0].last_access) : "—"} icon={<CalendarDays className="h-5 w-5" />} />
          <StatCard label="Режим" value="Live backend" icon={<Cpu className="h-5 w-5" />} />
        </div>

        <div className="mb-6 flex items-center gap-4 xl:mb-8">
          <div className="group relative flex-1">
            <Search className="absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-muted-foreground" />
            <input
              ref={searchInputRef}
              type="text"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Поиск по заголовку или последнему сообщению..."
              className="w-full rounded-2xl border border-border/50 bg-card py-3.5 pl-12 pr-4 text-[15px]"
            />
          </div>
          <div className="flex shrink-0 items-center rounded-2xl border border-border/50 bg-card p-1">
            {FILTERS.map((filter) => (
              <button
                key={filter.id}
                type="button"
                onClick={() => setActiveFilter(filter.id)}
                className={`rounded-xl px-4 py-2 text-sm font-semibold transition-all ${
                  activeFilter === filter.id
                    ? "bg-secondary text-foreground shadow-sm ring-1 ring-border/20"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {filter.label}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-12 lg:gap-6 xl:gap-8">
          <div className="space-y-4 lg:col-span-7">
            {filtered.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-border/40 bg-card/70 px-6 py-10 text-center text-sm text-muted-foreground">
                По текущему фильтру сессии не найдены.
              </div>
            ) : null}
            {filtered.map((session) => (
              <motion.div
                key={session.session_id}
                layout
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                onClick={() => setSelectedId(session.session_id)}
                className={`group cursor-pointer rounded-2xl border p-4 transition-all lg:p-5 xl:p-6 ${
                  selectedId === session.session_id
                    ? "border-primary/50 bg-primary/[0.03] ring-1 ring-primary/20"
                    : "border-border/50 bg-card hover:shadow-md"
                }`}
              >
                <div className="mb-3 flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`h-2.5 w-2.5 rounded-full ${session.has_dataset ? "bg-emerald-500" : "bg-zinc-500"}`} />
                    <h3 className="text-lg font-semibold group-hover:text-primary">{session.title}</h3>
                  </div>
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      toggleFavorite(session.session_id);
                    }}
                    className="rounded-lg p-1.5 text-amber-400/80 transition-all hover:bg-amber-400/10 hover:text-amber-400"
                    aria-label="Переключить избранное"
                  >
                    <Star
                      className={`h-5 w-5 ${
                        favoriteIds.includes(session.session_id) ? "fill-amber-400 text-amber-400" : ""
                      }`}
                    />
                  </button>
                </div>
                <p className="mb-6 line-clamp-2 text-[15px] leading-relaxed text-muted-foreground">{session.last_message_preview || "Пока без сообщений."}</p>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4 text-muted-foreground">
                    <div className="flex items-center gap-1.5 rounded-lg bg-secondary/80 px-2.5 py-1 text-[12px] font-medium">
                      <Bot className="h-3.5 w-3.5" />
                      {modelLabel}
                    </div>
                    <div className="flex items-center gap-1.5 text-[13px]">
                      <BarChart3 className="h-3.5 w-3.5" />
                      <span>{getPinnedCount(session.session_id)}</span>
                    </div>
                    <div className="flex items-center gap-1.5 text-[13px]">
                      <LayoutGrid className="h-3.5 w-3.5" />
                      <span>{session.has_dataset ? 1 : 0}</span>
                    </div>
                    <div className="flex items-center gap-1.5 text-[13px]">
                      <Database className="h-3.5 w-3.5" />
                      <span>{session.has_dataset ? 1 : 0}</span>
                    </div>
                  </div>
                  <div className="text-[13px] text-muted-foreground">{formatDateTime(session.last_access)}</div>
                </div>
              </motion.div>
            ))}
          </div>

          <div className="sticky top-28 lg:col-span-5">
            {selectedState ? (
              <div className="overflow-hidden rounded-3xl border border-border/60 bg-card shadow-2xl">
                <div className="border-b border-border/40 p-5 lg:p-6 xl:p-8">
                  <div className="mb-6 flex items-center justify-between">
                    <div className="rounded-full bg-secondary px-3 py-1 text-[12px] font-bold uppercase tracking-wider text-muted-foreground">
                      ID: {selectedState.session_id}
                    </div>
                    <Link to="/workspace" onClick={() => window.localStorage.setItem(getActiveSessionStorageKey(user?.id), selectedState.session_id)} className="rounded-xl bg-primary px-4 py-2 text-sm font-bold text-primary-foreground">
                      Открыть
                    </Link>
                  </div>
                  <input
                    value={renameDraft}
                    onChange={(event) => setRenameDraft(event.target.value)}
                    className="mb-4 w-full rounded-2xl border border-border/60 bg-secondary/50 px-4 py-3 text-2xl font-bold tracking-tight"
                  />
                  <div className="flex gap-3">
                    <button type="button" onClick={() => void handleRename()} className="rounded-2xl bg-foreground px-4 py-3 text-sm font-bold text-background">
                      Сохранить title
                    </button>
                    <button type="button" onClick={() => void handleDeleteSession(selectedState.session_id)} className="rounded-2xl border border-rose-500/30 px-4 py-3 text-sm font-bold text-rose-400">
                      <Trash2 className="mr-1 inline h-4 w-4" />
                      Удалить
                    </button>
                  </div>
                </div>

                <div className="space-y-6 p-5 lg:p-6 xl:space-y-8 xl:p-8">
                  <div>
                    <h4 className="mb-4 text-[13px] font-bold uppercase tracking-widest text-muted-foreground">Обзор</h4>
                    <div className="rounded-2xl border border-border/30 bg-secondary/50 p-5 text-[15px] italic">
                      "{sessions.find((item) => item.session_id === selectedState.session_id)?.last_message_preview || "Детали появятся после первого сообщения."}"
                    </div>
                  </div>

                  <div>
                    <h4 className="mb-4 text-[13px] font-bold uppercase tracking-widest text-muted-foreground">История</h4>
                    <div className="custom-scrollbar max-h-[360px] space-y-4 overflow-y-auto pr-2">
                      {selectedState.chat_history.map((message, index) => (
                        <div key={`${message.timestamp}-${index}`} className={`flex gap-3 ${message.role === "user" ? "flex-row-reverse" : ""}`}>
                          <div className={`flex h-8 w-8 items-center justify-center rounded-full ${message.role === "user" ? "bg-secondary text-muted-foreground" : "bg-primary/20 text-primary"}`}>
                            {message.role === "user" ? <MessageSquare className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                          </div>
                          <div className={`max-w-[85%] rounded-2xl p-3.5 text-[14px] leading-relaxed ${message.role === "user" ? "rounded-tr-none bg-primary text-primary-foreground" : "rounded-tl-none border border-border/20 bg-secondary/80"}`}>
                            {message.content}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="grid gap-3 border-t border-border/40 pt-4">
                    <button
                      type="button"
                      onClick={handleExportSelectedSession}
                      className="flex items-center justify-center gap-2 rounded-2xl border border-border/50 bg-secondary py-3 text-[14px] font-semibold text-muted-foreground transition-all hover:text-foreground"
                    >
                      <Download className="h-4 w-4" />
                      Скачать чат
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex h-[400px] flex-col items-center justify-center rounded-3xl border-2 border-dashed border-border/20 p-8 text-center xl:h-[600px]">
                <div className="mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-secondary/30">
                  <MessageSquare className="h-10 w-10 text-muted-foreground/50" />
                </div>
                <h3 className="mb-3 text-2xl font-bold">Выберите сессию</h3>
                <p className="max-w-xs text-muted-foreground">Правая панель остается частью нового frontend и показывает живые backend-данные по выбранной сессии.</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

function StatCard({ label, value, icon }: { label: string; value: string; icon: ReactNode }) {
  return (
    <div className="rounded-2xl border border-border/50 bg-card p-4 shadow-sm lg:p-5 xl:p-6">
      <div className="mb-3 flex items-center justify-between xl:mb-4">
        <div className="rounded-xl bg-white/5 p-2.5 text-primary">{icon}</div>
      </div>
      <div className="mb-1 text-2xl font-bold tracking-tight">{value}</div>
      <div className="text-[14px] text-muted-foreground">{label}</div>
    </div>
  );
}
