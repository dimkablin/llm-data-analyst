import { FormEvent, useEffect, useRef, useState } from "react";

import type { AnalysisDepth, AuthUser, SessionSummary } from "../types";

type SettingsSidebarProps = {
  open: boolean;
  pinned: boolean;
  onTogglePinned: () => void;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
  onRequestClose: () => void;
  currentUser: AuthUser;
  onLogout: () => Promise<void>;
  sessions: SessionSummary[];
  activeSessionId: string;
  onSelectSession: (sessionId: string) => Promise<void>;
  onDeleteSession: (sessionId: string) => Promise<void>;
  sessionTitle: string;
  onRenameSession: (title: string) => Promise<void>;
  includeReasoning: boolean;
  onToggleReasoning: (value: boolean) => void;
  useHistory: boolean;
  onToggleHistory: (value: boolean) => void;
  datasetName: string;
  hasDataset: boolean;
  isUploading: boolean;
  onUpload: (file: File) => Promise<void>;
  showCode: boolean;
  onToggleShowCode: (value: boolean) => void;
  sortMode: "newest" | "oldest";
  onSortModeChange: (value: "newest" | "oldest") => void;
  dashboardColumns: 2 | 3 | 4;
  onDashboardColumnsChange: (value: 2 | 3 | 4) => void;
  dashboardLayoutLocked: boolean;
  onToggleDashboardLayoutLocked: (value: boolean) => void;
  onResetDashboardLayout: () => void;
  themeMode: "light" | "dark";
  onThemeModeChange: (value: "light" | "dark") => void;
  analysisDepth: AnalysisDepth;
  onAnalysisDepthChange: (value: AnalysisDepth) => void;
  modelName: string;
  runtimeSettings: {
    llm_temperature_chat: number;
    llm_temperature_tool: number;
    llm_max_tokens_default: number;
    llm_max_tokens_reasoning: number;
    backend_query_timeout_sec: number;
    agent_max_steps: number;
    agent_step_timeout_sec: number;
    agent_inner_recursion_limit: number;
  };
  onRuntimeSettingsChange: (next: {
    llm_temperature_chat: number;
    llm_temperature_tool: number;
    llm_max_tokens_default: number;
    llm_max_tokens_reasoning: number;
    backend_query_timeout_sec: number;
    agent_max_steps: number;
    agent_step_timeout_sec: number;
    agent_inner_recursion_limit: number;
  }) => void;
  onSaveRuntimeSettings: (next: {
    llm_temperature_chat: number;
    llm_temperature_tool: number;
    llm_max_tokens_default: number;
    llm_max_tokens_reasoning: number;
    backend_query_timeout_sec: number;
    agent_max_steps: number;
    agent_step_timeout_sec: number;
    agent_inner_recursion_limit: number;
  }) => Promise<void>;
  runtimeSettingsSaving: boolean;
  pinnedCount: number;
  onDownloadChat: () => void;
  onDownloadDashboard: () => void;
  onResetSession: () => Promise<void>;
};

export function SettingsSidebar({
  open,
  pinned,
  onTogglePinned,
  onMouseEnter,
  onMouseLeave,
  onRequestClose,
  currentUser,
  onLogout,
  sessions,
  activeSessionId,
  onSelectSession,
  onDeleteSession,
  sessionTitle,
  onRenameSession,
  includeReasoning,
  onToggleReasoning,
  useHistory,
  onToggleHistory,
  datasetName,
  hasDataset,
  isUploading,
  onUpload,
  showCode,
  onToggleShowCode,
  sortMode,
  onSortModeChange,
  dashboardColumns,
  onDashboardColumnsChange,
  dashboardLayoutLocked,
  onToggleDashboardLayoutLocked,
  onResetDashboardLayout,
  themeMode,
  onThemeModeChange,
  analysisDepth,
  onAnalysisDepthChange,
  modelName,
  runtimeSettings,
  onRuntimeSettingsChange,
  onSaveRuntimeSettings,
  runtimeSettingsSaving,
  pinnedCount,
  onDownloadChat,
  onDownloadDashboard,
  onResetSession
}: SettingsSidebarProps): JSX.Element {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [titleDraft, setTitleDraft] = useState(sessionTitle);
  const [isRenaming, setIsRenaming] = useState(false);
  const [runtimeDraft, setRuntimeDraft] = useState(runtimeSettings);

  useEffect(() => {
    setTitleDraft(sessionTitle);
  }, [sessionTitle, activeSessionId]);

  useEffect(() => {
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, [activeSessionId]);

  useEffect(() => {
    if (!hasDataset && fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, [hasDataset]);

  useEffect(() => {
    setRuntimeDraft(runtimeSettings);
  }, [runtimeSettings]);

  return (
    <aside
      className={`panel sidebar ${open ? "sidebar-open" : "sidebar-closed"}`}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <div className="sidebar-head">
        <h2>Настройки</h2>
        <div className="sidebar-head-actions">
          <button type="button" className="btn-ghost btn-xs" onClick={onTogglePinned}>
            {pinned ? "Открепить" : "Закрепить"}
          </button>
          <button type="button" className="btn-ghost btn-xs" onClick={onRequestClose}>
            Скрыть
          </button>
        </div>
      </div>

      <div className="sidebar-group">
        <div className="kv-row">
          <span>Пользователь</span>
          <strong>{currentUser.username}</strong>
        </div>
        <div className="kv-row">
          <span>Сессия</span>
          <code title={activeSessionId || "init"}>{activeSessionId ? `${activeSessionId.slice(0, 10)}...` : "init"}</code>
        </div>
        <div className="kv-row">
          <span>Название</span>
          <strong title={sessionTitle}>{sessionTitle}</strong>
        </div>
        <div className="kv-row">
          <span>Датасет</span>
          <strong>{hasDataset ? "Загружен" : "Не загружен"}</strong>
        </div>
        <div className="kv-row">
          <span>Хранилище</span>
          <strong>7 дней TTL</strong>
        </div>
        <div className="kv-row">
          <span>Лимит</span>
          <strong>100 МБ</strong>
        </div>
        <div className="kv-row">
          <span>Закреплено</span>
          <strong>{pinnedCount}</strong>
        </div>
      </div>

      <div className="sidebar-group">
        <div className="sidebar-subtitle">Название чата</div>
        <form
          className="rename-form"
          onSubmit={(event: FormEvent) => {
            event.preventDefault();
            if (!titleDraft.trim()) {
              return;
            }
            setIsRenaming(true);
            void onRenameSession(titleDraft)
              .catch(() => undefined)
              .finally(() => setIsRenaming(false));
          }}
        >
          <input
            value={titleDraft}
            onChange={(event) => setTitleDraft(event.target.value)}
            placeholder="Введите название чата"
            maxLength={120}
          />
          <button type="submit" className="btn-ghost btn-xs" disabled={isRenaming || !titleDraft.trim()}>
            {isRenaming ? "..." : "Сохранить"}
          </button>
        </form>
      </div>

      <div className="sidebar-group session-list-group">
        <div className="sidebar-subtitle">История чатов</div>
        <div className="session-list">
          {sessions.length === 0 ? (
            <div className="session-list-empty">Пока нет сессий</div>
          ) : (
            sessions.map((session) => (
              <div key={session.session_id} className={`session-item ${session.session_id === activeSessionId ? "active" : ""}`}>
                <button
                  type="button"
                  className="session-item-main"
                  onClick={() => {
                    void onSelectSession(session.session_id);
                  }}
                >
                  <span className="session-item-title">{session.title || "Новый чат"}</span>
                  <span className="session-item-meta">
                    {session.has_dataset ? "CSV" : "без CSV"} | {new Date(session.last_access).toLocaleString("ru-RU")}
                  </span>
                </button>
                <button
                  type="button"
                  className="session-item-delete"
                  aria-label="Удалить чат"
                  title="Удалить чат"
                  onClick={() => {
                    void onDeleteSession(session.session_id);
                  }}
                >
                  ✕
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="sidebar-group">
        <label className="switch">
          <input type="checkbox" checked={useHistory} onChange={(event) => onToggleHistory(event.target.checked)} />
          <span>Использовать историю чата</span>
        </label>
        <label className="switch">
          <input
            type="checkbox"
            checked={includeReasoning}
            onChange={(event) => onToggleReasoning(event.target.checked)}
          />
          <span>Показывать reasoning</span>
        </label>
        <label className="switch">
          <input type="checkbox" checked={showCode} onChange={(event) => onToggleShowCode(event.target.checked)} />
          <span>Показывать код артефактов</span>
        </label>
      </div>

      <div className="sidebar-group">
        <div className="sidebar-subtitle">Настройки дашборда</div>
        <div className="sidebar-note">
          Перетаскивайте карточки за заголовок и меняйте размер за угол.
          Экспорт HTML сохраняет текущую раскладку.
        </div>
        <span className="sidebar-subtitle no-transform">Порядок добавления</span>
        <label className="switch">
          <input
            type="radio"
            name="sort_mode"
            checked={sortMode === "newest"}
            onChange={() => onSortModeChange("newest")}
          />
          <span>Сначала новые</span>
        </label>
        <label className="switch">
          <input
            type="radio"
            name="sort_mode"
            checked={sortMode === "oldest"}
            onChange={() => onSortModeChange("oldest")}
          />
          <span>Сначала старые</span>
        </label>
        <label className="switch">
          <input
            type="checkbox"
            checked={dashboardLayoutLocked}
            onChange={(event) => onToggleDashboardLayoutLocked(event.target.checked)}
          />
          <span>Зафиксировать layout (без drag/resize)</span>
        </label>
        <div className="row-actions">
          <span className="sidebar-subtitle no-transform">Колонки сетки</span>
          <div className="segmented">
            {[2, 3, 4].map((value) => (
              <button
                key={value}
                type="button"
                className={`btn-segment ${dashboardColumns === value ? "active" : ""}`}
                onClick={() => onDashboardColumnsChange(value as 2 | 3 | 4)}
              >
                {value}
              </button>
            ))}
          </div>
        </div>
        <div className="row-actions">
          <button type="button" className="btn-ghost btn-xs" onClick={onResetDashboardLayout}>
            Сбросить layout
          </button>
        </div>
      </div>

      <div className="sidebar-group">
        <div className="sidebar-subtitle">Внешний вид</div>
        <div className="row-actions">
          <span className="sidebar-subtitle no-transform">Тема</span>
          <div className="segmented">
            <button
              type="button"
              className={`btn-segment ${themeMode === "light" ? "active" : ""}`}
              onClick={() => onThemeModeChange("light")}
            >
              Светлая
            </button>
            <button
              type="button"
              className={`btn-segment ${themeMode === "dark" ? "active" : ""}`}
              onClick={() => onThemeModeChange("dark")}
            >
              Темная
            </button>
          </div>
        </div>
      </div>

      <div className="sidebar-group">
        <div className="sidebar-subtitle">Глубина анализа</div>
        <div className="sidebar-note">
          Влияет на количество шагов, детальность плана и использование оценки результата.
        </div>
        <div className="row-actions">
          <div className="segmented">
            {([
              ["light", "Лёгкий"],
              ["medium", "Средний"],
              ["deep", "Глубокий"],
            ] as const).map(([value, label]) => (
              <button
                key={value}
                type="button"
                className={`btn-segment ${analysisDepth === value ? "active" : ""}`}
                onClick={() => onAnalysisDepthChange(value)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="sidebar-group">
        <div className="sidebar-subtitle">Профиль модели</div>
        <div className="kv-row">
          <span>Модель</span>
          <code title={modelName}>{modelName}</code>
        </div>
        <div className="kv-row">
          <span>Markdown</span>
          <strong>Включён</strong>
        </div>
      </div>

      <div className="sidebar-group">
        <div className="sidebar-subtitle">Runtime агента</div>
        <div className="runtime-grid">
          <label>
            Темп. чата
            <input
              type="number"
              step="0.05"
              min={0}
              max={2}
              value={runtimeDraft.llm_temperature_chat}
              onChange={(event) => setRuntimeDraft((prev) => ({ ...prev, llm_temperature_chat: Number(event.target.value) }))}
            />
          </label>
          <label>
            Темп. инструмента
            <input
              type="number"
              step="0.05"
              min={0}
              max={2}
              value={runtimeDraft.llm_temperature_tool}
              onChange={(event) => setRuntimeDraft((prev) => ({ ...prev, llm_temperature_tool: Number(event.target.value) }))}
            />
          </label>
          <label>
            Макс. токенов
            <input
              type="number"
              min={128}
              max={32768}
              value={runtimeDraft.llm_max_tokens_default}
              onChange={(event) => setRuntimeDraft((prev) => ({ ...prev, llm_max_tokens_default: Number(event.target.value) }))}
            />
          </label>
          <label>
            Токенов рассуждения
            <input
              type="number"
              min={128}
              max={32768}
              value={runtimeDraft.llm_max_tokens_reasoning}
              onChange={(event) => setRuntimeDraft((prev) => ({ ...prev, llm_max_tokens_reasoning: Number(event.target.value) }))}
            />
          </label>
          <label>
            Таймаут запроса, сек
            <input
              type="number"
              min={15}
              max={1800}
              value={runtimeDraft.backend_query_timeout_sec}
              onChange={(event) => setRuntimeDraft((prev) => ({ ...prev, backend_query_timeout_sec: Number(event.target.value) }))}
            />
          </label>
          <label>
            Макс. шагов
            <input
              type="number"
              min={2}
              max={20}
              value={runtimeDraft.agent_max_steps}
              onChange={(event) => setRuntimeDraft((prev) => ({ ...prev, agent_max_steps: Number(event.target.value) }))}
            />
          </label>
          <label>
            Таймаут шага, сек
            <input
              type="number"
              min={5}
              max={600}
              value={runtimeDraft.agent_step_timeout_sec}
              onChange={(event) => setRuntimeDraft((prev) => ({ ...prev, agent_step_timeout_sec: Number(event.target.value) }))}
            />
          </label>
          <label>
            Внутр. рекурсия
            <input
              type="number"
              min={2}
              max={20}
              value={runtimeDraft.agent_inner_recursion_limit}
              onChange={(event) => setRuntimeDraft((prev) => ({ ...prev, agent_inner_recursion_limit: Number(event.target.value) }))}
            />
          </label>
        </div>
        <div className="row-actions">
          <button
            type="button"
            className="btn-ghost btn-xs"
              onClick={() => {
                onRuntimeSettingsChange(runtimeDraft);
                void onSaveRuntimeSettings(runtimeDraft);
              }}
            disabled={runtimeSettingsSaving}
          >
            {runtimeSettingsSaving ? "Сохраняем..." : "Сохранить параметры"}
          </button>
        </div>
      </div>

      <div className="sidebar-group">
        <label className="upload-box sidebar-upload">
          <span>{datasetName ? `CSV: ${datasetName}` : "Загрузите CSV (до 100MB)"}</span>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv"
            disabled={!activeSessionId || isUploading}
            onClick={(event) => {
              event.currentTarget.value = "";
            }}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) {
                void onUpload(file);
              }
            }}
          />
        </label>
      </div>

      <div className="sidebar-group">
        <button type="button" className="btn-ghost" onClick={() => void onResetSession()}>
          Новый чат
        </button>
        <button type="button" className="btn-ghost" onClick={onDownloadDashboard}>
          Скачать дашборд (HTML)
        </button>
        <button type="button" className="btn-ghost" onClick={onDownloadChat}>
          Скачать историю чата (HTML)
        </button>
        <button
          type="button"
          className="btn-ghost"
          onClick={() => {
            void onLogout();
          }}
        >
          Выйти
        </button>
      </div>
    </aside>
  );
}
