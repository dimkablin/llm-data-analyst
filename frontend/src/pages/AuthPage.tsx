import { FormEvent, useEffect, useMemo, useState } from "react";

import {
  changePassword,
  createAdminUser,
  deleteAdminUser,
  getUserSettings,
  listAdminUsers,
  updateAdminUser,
  updateUserSettings
} from "../api";
import { RevealOnScroll } from "../components/RevealOnScroll";
import { SiteNav } from "../components/SiteNav";
import type { AuthUser, UserSettings } from "../types";

type AuthPageProps = {
  currentUser: AuthUser | null;
  onNavigate: (path: "/" | "/user" | "/technical" | "/app" | "/phoenix") => void;
  onLogin: (username: string, password: string) => Promise<void>;
  onRegister: (username: string, password: string) => Promise<void>;
  onLogout: () => Promise<void>;
};

const THEME_STORAGE_KEY = "llm_data_analyst_setting_theme_";

function formatDate(value: string): string {
  const raw = String(value || "").trim();
  if (!raw) {
    return "-";
  }
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) {
    return raw;
  }
  return date.toLocaleString("ru-RU");
}

function normalizeTheme(theme: string): "light" | "dark" {
  return theme === "light" ? "light" : "dark";
}

export function AuthPage({
  currentUser,
  onNavigate,
  onLogin,
  onRegister,
  onLogout
}: AuthPageProps): JSX.Element {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [passwordRepeat, setPasswordRepeat] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [settings, setSettings] = useState<UserSettings>({
    theme: "dark",
    default_include_reasoning: true,
    default_answer_style: "detailed",
    analysis_depth: "light",
    llm_temperature_chat: 0.5,
    llm_temperature_tool: 0.15,
    llm_max_tokens_default: 1200,
    llm_max_tokens_reasoning: 2200,
    backend_query_timeout_sec: 180,
    agent_max_steps: 5,
    agent_step_timeout_sec: 45,
    agent_inner_recursion_limit: 6
  });
  const [settingsLoading, setSettingsLoading] = useState(false);
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [settingsMessage, setSettingsMessage] = useState<string | null>(null);

  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [newPasswordRepeat, setNewPasswordRepeat] = useState("");
  const [passwordSaving, setPasswordSaving] = useState(false);
  const [passwordMessage, setPasswordMessage] = useState<string | null>(null);

  const [adminUsers, setAdminUsers] = useState<AuthUser[]>([]);
  const [adminLoading, setAdminLoading] = useState(false);
  const [adminMessage, setAdminMessage] = useState<string | null>(null);
  const [adminActionInProgress, setAdminActionInProgress] = useState<string | null>(null);
  const [newAdminUsername, setNewAdminUsername] = useState("");
  const [newAdminPassword, setNewAdminPassword] = useState("");
  const [newAdminIsAdmin, setNewAdminIsAdmin] = useState(false);
  const [userRoleDrafts, setUserRoleDrafts] = useState<Record<number, boolean>>({});
  const [userPasswordDrafts, setUserPasswordDrafts] = useState<Record<number, string>>({});

  const modeTitle = useMemo(
    () => (mode === "login" ? "Вход в рабочую область" : "Регистрация нового пользователя"),
    [mode]
  );

  const modeDescription = useMemo(
    () =>
      mode === "login"
        ? "Войдите в аккаунт, чтобы открыть персональные сессии, чат и dashboard."
        : "Создайте учетную запись для самостоятельной аналитики и сохранения артефактов.",
    [mode]
  );

  useEffect(() => {
    if (!currentUser) {
      return;
    }
    let cancelled = false;
    setSettingsLoading(true);
    setSettingsMessage(null);
    void (async () => {
      try {
        const loaded = await getUserSettings();
        if (cancelled) {
          return;
        }
        const normalizedTheme = normalizeTheme(loaded.theme);
        const normalized: UserSettings = {
          theme: normalizedTheme,
          default_include_reasoning: Boolean(loaded.default_include_reasoning),
          default_answer_style: loaded.default_answer_style === "concise" ? "concise" : "detailed",
          analysis_depth: (loaded.analysis_depth as UserSettings["analysis_depth"]) || "light",
          llm_temperature_chat: Number(loaded.llm_temperature_chat),
          llm_temperature_tool: Number(loaded.llm_temperature_tool),
          llm_max_tokens_default: Number(loaded.llm_max_tokens_default),
          llm_max_tokens_reasoning: Number(loaded.llm_max_tokens_reasoning),
          backend_query_timeout_sec: Number(loaded.backend_query_timeout_sec),
          agent_max_steps: Number(loaded.agent_max_steps),
          agent_step_timeout_sec: Number(loaded.agent_step_timeout_sec),
          agent_inner_recursion_limit: Number(loaded.agent_inner_recursion_limit)
        };
        setSettings(normalized);
        document.documentElement.setAttribute("data-theme", normalizedTheme);
        window.localStorage.setItem(`${THEME_STORAGE_KEY}${currentUser.id}`, normalizedTheme);
      } catch (err) {
        if (!cancelled) {
          setSettingsMessage(`Не удалось загрузить настройки: ${String(err)}`);
        }
      } finally {
        if (!cancelled) {
          setSettingsLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [currentUser]);

  useEffect(() => {
    if (!currentUser?.is_admin) {
      setAdminUsers([]);
      setUserRoleDrafts({});
      setUserPasswordDrafts({});
      return;
    }
    let cancelled = false;
    setAdminLoading(true);
    setAdminMessage(null);
    void (async () => {
      try {
        const rows = await listAdminUsers();
        if (cancelled) {
          return;
        }
        setAdminUsers(rows);
        setUserRoleDrafts(
          rows.reduce<Record<number, boolean>>((acc, row) => {
            acc[row.id] = row.is_admin;
            return acc;
          }, {})
        );
      } catch (err) {
        if (!cancelled) {
          setAdminMessage(`Не удалось загрузить список пользователей: ${String(err)}`);
        }
      } finally {
        if (!cancelled) {
          setAdminLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [currentUser]);

  function switchMode(nextMode: "login" | "register"): void {
    setMode(nextMode);
    setError(null);
    setPassword("");
    setPasswordRepeat("");
  }

  async function handleSubmit(event: FormEvent): Promise<void> {
    event.preventDefault();
    if (!username.trim()) {
      setError("Введите логин");
      return;
    }
    if (!password) {
      setError("Введите пароль");
      return;
    }
    if (mode === "register" && password !== passwordRepeat) {
      setError("Пароли не совпадают");
      return;
    }

    setError(null);
    setIsSubmitting(true);
    try {
      if (mode === "login") {
        await onLogin(username.trim(), password);
      } else {
        await onRegister(username.trim(), password);
      }
      onNavigate("/app");
    } catch (err) {
      setError(String(err));
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleSaveSettings(): Promise<void> {
    if (!currentUser) {
      return;
    }
    setSettingsMessage(null);
    setSettingsSaving(true);
    try {
      const normalizedTheme = normalizeTheme(settings.theme);
      const updated = await updateUserSettings({
        theme: normalizedTheme,
        default_include_reasoning: settings.default_include_reasoning,
        default_answer_style: settings.default_answer_style
      });
      const normalized: UserSettings = {
        theme: normalizeTheme(updated.theme),
        default_include_reasoning: Boolean(updated.default_include_reasoning),
        default_answer_style: updated.default_answer_style === "concise" ? "concise" : "detailed",
        analysis_depth: (updated.analysis_depth as UserSettings["analysis_depth"]) || "light",
        llm_temperature_chat: Number(updated.llm_temperature_chat),
        llm_temperature_tool: Number(updated.llm_temperature_tool),
        llm_max_tokens_default: Number(updated.llm_max_tokens_default),
        llm_max_tokens_reasoning: Number(updated.llm_max_tokens_reasoning),
        backend_query_timeout_sec: Number(updated.backend_query_timeout_sec),
        agent_max_steps: Number(updated.agent_max_steps),
        agent_step_timeout_sec: Number(updated.agent_step_timeout_sec),
        agent_inner_recursion_limit: Number(updated.agent_inner_recursion_limit)
      };
      setSettings(normalized);
      document.documentElement.setAttribute("data-theme", normalized.theme);
      window.localStorage.setItem(`${THEME_STORAGE_KEY}${currentUser.id}`, normalized.theme);
      setSettingsMessage("Настройки сохранены");
    } catch (err) {
      setSettingsMessage(`Не удалось сохранить настройки: ${String(err)}`);
    } finally {
      setSettingsSaving(false);
    }
  }

  async function handleChangePasswordSubmit(event: FormEvent): Promise<void> {
    event.preventDefault();
    if (!currentPassword || !newPassword) {
      setPasswordMessage("Заполните все поля пароля");
      return;
    }
    if (newPassword !== newPasswordRepeat) {
      setPasswordMessage("Новый пароль и подтверждение не совпадают");
      return;
    }
    setPasswordSaving(true);
    setPasswordMessage(null);
    try {
      await changePassword(currentPassword, newPassword);
      setCurrentPassword("");
      setNewPassword("");
      setNewPasswordRepeat("");
      setPasswordMessage("Пароль обновлен");
    } catch (err) {
      setPasswordMessage(`Не удалось сменить пароль: ${String(err)}`);
    } finally {
      setPasswordSaving(false);
    }
  }

  async function refreshAdminUsers(): Promise<void> {
    if (!currentUser?.is_admin) {
      return;
    }
    const rows = await listAdminUsers();
    setAdminUsers(rows);
    setUserRoleDrafts(
      rows.reduce<Record<number, boolean>>((acc, row) => {
        acc[row.id] = row.is_admin;
        return acc;
      }, {})
    );
  }

  async function handleAdminCreateUser(event: FormEvent): Promise<void> {
    event.preventDefault();
    if (!newAdminUsername.trim() || !newAdminPassword) {
      setAdminMessage("Для создания пользователя нужен логин и пароль");
      return;
    }
    setAdminMessage(null);
    setAdminActionInProgress("create-user");
    try {
      await createAdminUser(newAdminUsername.trim(), newAdminPassword, newAdminIsAdmin);
      setNewAdminUsername("");
      setNewAdminPassword("");
      setNewAdminIsAdmin(false);
      await refreshAdminUsers();
      setAdminMessage("Пользователь создан");
    } catch (err) {
      setAdminMessage(`Ошибка создания пользователя: ${String(err)}`);
    } finally {
      setAdminActionInProgress(null);
    }
  }

  async function handleAdminSaveRole(userId: number): Promise<void> {
    const nextRole = Boolean(userRoleDrafts[userId]);
    setAdminMessage(null);
    setAdminActionInProgress(`role-${userId}`);
    try {
      await updateAdminUser(userId, { is_admin: nextRole });
      await refreshAdminUsers();
      setAdminMessage("Роль пользователя обновлена");
    } catch (err) {
      setAdminMessage(`Ошибка изменения роли: ${String(err)}`);
    } finally {
      setAdminActionInProgress(null);
    }
  }

  async function handleAdminResetPassword(userId: number): Promise<void> {
    const nextPassword = String(userPasswordDrafts[userId] || "");
    if (!nextPassword) {
      setAdminMessage("Введите новый пароль для сброса");
      return;
    }
    setAdminMessage(null);
    setAdminActionInProgress(`password-${userId}`);
    try {
      await updateAdminUser(userId, { password: nextPassword });
      setUserPasswordDrafts((prev) => ({ ...prev, [userId]: "" }));
      setAdminMessage("Пароль пользователя обновлен");
    } catch (err) {
      setAdminMessage(`Ошибка смены пароля: ${String(err)}`);
    } finally {
      setAdminActionInProgress(null);
    }
  }

  async function handleAdminDeleteUser(userId: number, usernameValue: string): Promise<void> {
    const ok = window.confirm(`Удалить пользователя "${usernameValue}"?`);
    if (!ok) {
      return;
    }
    setAdminMessage(null);
    setAdminActionInProgress(`delete-${userId}`);
    try {
      await deleteAdminUser(userId);
      await refreshAdminUsers();
      setAdminMessage("Пользователь удален");
    } catch (err) {
      setAdminMessage(`Ошибка удаления пользователя: ${String(err)}`);
    } finally {
      setAdminActionInProgress(null);
    }
  }

  return (
    <div className="site-page auth-page auth-page-refresh">
      <SiteNav currentUser={currentUser} onNavigate={onNavigate} />

      <main className={`auth-wrap-pro user-page-wrap ${currentUser ? "user-page-authenticated" : ""}`.trim()}>
        <RevealOnScroll className={`auth-brand-panel ${currentUser ? "user-brand-panel-compact" : ""}`.trim()}>
          {currentUser ? (
            <>
              <span className="landing-chip">Центр пользователя</span>
              <h1>Настройки пользователя</h1>
              <p>
                Управляйте профилем, безопасностью и персональными настройками интерфейса.
                Раздел собран как единый рабочий кабинет пользователя.
              </p>
              <div className="user-brand-meta">
                <span>Логин: {currentUser.username}</span>
                <span>Роль: {currentUser.is_admin ? "Администратор" : "Пользователь"}</span>
              </div>
              <div className="auth-proof-strip" aria-label="Быстрые разделы">
                <span>Профиль</span>
                <span>Настройки</span>
                <span>Безопасность</span>
              </div>
              <nav className="user-quick-links" aria-label="Навигация по разделам User">
                <a href="#user-profile">Профиль</a>
                <a href="#user-settings">Настройки</a>
                <a href="#user-security">Безопасность</a>
                {currentUser.is_admin ? <a href="#user-admin">Управление пользователями</a> : null}
              </nav>
            </>
          ) : (
            <>
              <span className="landing-chip">Управление доступом</span>
              <h1>Единая зона доступа: вход, профиль, безопасность и администрирование</h1>
              <p>
                Здесь пользователь входит в систему, управляет профилем и настройками.
                Администратор дополнительно получает контроль пользователей, ролей и паролей.
              </p>
              <div className="auth-proof-strip" aria-label="Возможности страницы User">
                <span>Вход / Регистрация</span>
                <span>Профиль + Настройки</span>
                <span>Управление пользователями</span>
              </div>
              <div className="auth-brand-mini-cards">
                <article>
                  <strong>Профиль</strong>
                  <span>Роль, дата создания, быстрый переход в рабочую область и выход.</span>
                </article>
                <article>
                  <strong>Настройки</strong>
                  <span>Тема и параметры ответа, влияющие на работу интерфейса.</span>
                </article>
                <article>
                  <strong>Смена пароля</strong>
                  <span>Смена пароля без обращения к администратору.</span>
                </article>
                <article>
                  <strong>Администрирование</strong>
                  <span>Добавление, удаление пользователей, смена пароля и ролей.</span>
                </article>
              </div>
            </>
          )}
        </RevealOnScroll>

        <RevealOnScroll className="auth-form-panel user-page-panel" delayMs={80}>
          {!currentUser ? (
            <>
              <div className="auth-head">
                <span className="auth-head-kicker">Пользователь</span>
                <h2>{modeTitle}</h2>
                <p>{modeDescription}</p>
                <div className="auth-trust-note">
                  <span>Хеширование паролей</span>
                  <span>Изолированные сессии</span>
                  <span>Роли пользователей</span>
                </div>
              </div>

              <div className="auth-mode-toggle" role="tablist" aria-label="Режим авторизации">
                <button
                  type="button"
                  className={mode === "login" ? "auth-mode-btn active" : "auth-mode-btn"}
                  onClick={() => switchMode("login")}
                  disabled={isSubmitting}
                >
                  Вход
                </button>
                <button
                  type="button"
                  className={mode === "register" ? "auth-mode-btn active" : "auth-mode-btn"}
                  onClick={() => switchMode("register")}
                  disabled={isSubmitting}
                >
                  Регистрация
                </button>
              </div>

              <form className="auth-form" onSubmit={(event) => void handleSubmit(event)}>
                <label>
                  <span>Логин</span>
                  <input
                    value={username}
                    onChange={(event) => setUsername(event.target.value)}
                    placeholder="например: analyst_team"
                    autoComplete="username"
                  />
                </label>

                <label>
                  <span>Пароль</span>
                  <input
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    placeholder="Введите пароль"
                    type="password"
                    autoComplete={mode === "login" ? "current-password" : "new-password"}
                  />
                </label>

                {mode === "register" ? (
                  <label>
                    <span>Повторите пароль</span>
                    <input
                      value={passwordRepeat}
                      onChange={(event) => setPasswordRepeat(event.target.value)}
                      placeholder="Повторите пароль"
                      type="password"
                      autoComplete="new-password"
                    />
                  </label>
                ) : null}

                {error ? <div className="alert-error">{error}</div> : null}

                <div className="auth-actions">
                  <button type="submit" disabled={isSubmitting}>
                    {isSubmitting ? "Проверка..." : mode === "login" ? "Войти" : "Создать аккаунт"}
                  </button>
                </div>

                <p className="auth-form-hint">
                  {mode === "login"
                    ? "Нет аккаунта? Переключитесь на регистрацию."
                    : "Уже есть аккаунт? Переключитесь на вход."}
                </p>
                <p className="auth-form-hint user-forgot-hint">
                  Забыли пароль? Обратитесь к администратору системы.
                </p>
              </form>
            </>
          ) : (
            <section className="user-page-sections">
              <article className="user-card user-card-profile" id="user-profile">
                <h2>Профиль</h2>
                <p className="user-meta-row">
                  Логин: <strong>{currentUser.username}</strong>
                </p>
                <p className="user-meta-row">
                  Роль: <strong>{currentUser.is_admin ? "Администратор" : "Пользователь"}</strong>
                </p>
                <p className="user-meta-row">Создан: {formatDate(currentUser.created_at)}</p>
                <div className="auth-actions">
                  <button type="button" onClick={() => onNavigate("/app")}>Открыть workspace</button>
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
                <p className="auth-form-hint user-forgot-hint">
                  Если забыли пароль и не можете войти, обратитесь к администратору.
                </p>
              </article>

              <article className="user-card user-card-settings" id="user-settings">
                <h3>Настройки</h3>
                <div className="user-settings-grid">
                  <label>
                    <span>Тема</span>
                    <select
                      value={settings.theme}
                      onChange={(event) =>
                        setSettings((prev) => ({
                          ...prev,
                          theme: normalizeTheme(event.target.value)
                        }))
                      }
                      disabled={settingsLoading || settingsSaving}
                    >
                      <option value="dark">Темная</option>
                      <option value="light">Светлая</option>
                    </select>
                  </label>
                  <label className="user-switch-row">
                    <input
                      type="checkbox"
                      checked={settings.default_include_reasoning}
                      onChange={(event) =>
                        setSettings((prev) => ({
                          ...prev,
                          default_include_reasoning: event.target.checked
                        }))
                      }
                      disabled={settingsLoading || settingsSaving}
                    />
                    <span>Показывать reasoning по умолчанию</span>
                  </label>
                  <label>
                    <span>Стиль ответа</span>
                    <select
                      value={settings.default_answer_style}
                      onChange={(event) =>
                        setSettings((prev) => ({
                          ...prev,
                          default_answer_style: event.target.value === "concise" ? "concise" : "detailed"
                        }))
                      }
                      disabled={settingsLoading || settingsSaving}
                    >
                      <option value="detailed">Развернутый</option>
                      <option value="concise">Краткий</option>
                    </select>
                  </label>
                </div>
                <div className="auth-actions">
                  <button type="button" onClick={() => void handleSaveSettings()} disabled={settingsSaving}>
                    {settingsSaving ? "Сохранение..." : "Сохранить настройки"}
                  </button>
                </div>
                {settingsMessage ? <p className="user-inline-message">{settingsMessage}</p> : null}
              </article>

              <article className="user-card user-card-password" id="user-security">
                <h3>Смена пароля</h3>
                <form className="user-inline-form" onSubmit={(event) => void handleChangePasswordSubmit(event)}>
                  <label>
                    <span>Текущий пароль</span>
                    <input
                      type="password"
                      value={currentPassword}
                      onChange={(event) => setCurrentPassword(event.target.value)}
                      autoComplete="current-password"
                    />
                  </label>
                  <label>
                    <span>Новый пароль</span>
                    <input
                      type="password"
                      value={newPassword}
                      onChange={(event) => setNewPassword(event.target.value)}
                      autoComplete="new-password"
                    />
                  </label>
                  <label>
                    <span>Повторите новый пароль</span>
                    <input
                      type="password"
                      value={newPasswordRepeat}
                      onChange={(event) => setNewPasswordRepeat(event.target.value)}
                      autoComplete="new-password"
                    />
                  </label>
                  <div className="auth-actions">
                    <button type="submit" disabled={passwordSaving}>
                      {passwordSaving ? "Обновление..." : "Обновить пароль"}
                    </button>
                  </div>
                </form>
                {passwordMessage ? <p className="user-inline-message">{passwordMessage}</p> : null}
              </article>

              {currentUser.is_admin ? (
                <article className="user-card admin-panel" id="user-admin">
                  <h3>Админ-панель пользователей</h3>
                  <p className="auth-form-hint">
                    Управление доступом: добавить, удалить, сменить пароль, назначить роль.
                  </p>

                  <form className="admin-create-form" onSubmit={(event) => void handleAdminCreateUser(event)}>
                    <label>
                      <span>Новый логин</span>
                      <input
                        value={newAdminUsername}
                        onChange={(event) => setNewAdminUsername(event.target.value)}
                        placeholder="например: bi_manager"
                      />
                    </label>
                    <label>
                      <span>Временный пароль</span>
                      <input
                        type="password"
                        value={newAdminPassword}
                        onChange={(event) => setNewAdminPassword(event.target.value)}
                        placeholder="не менее 4 символов"
                      />
                    </label>
                    <label className="user-switch-row">
                      <input
                        type="checkbox"
                        checked={newAdminIsAdmin}
                        onChange={(event) => setNewAdminIsAdmin(event.target.checked)}
                      />
                      <span>Назначить администратором</span>
                    </label>
                    <div className="auth-actions">
                      <button type="submit" disabled={adminActionInProgress === "create-user"}>
                        {adminActionInProgress === "create-user" ? "Создание..." : "Добавить пользователя"}
                      </button>
                    </div>
                  </form>

                  {adminLoading ? (
                    <p className="user-inline-message">Загрузка пользователей...</p>
                  ) : (
                    <div className="admin-users-table-wrap">
                      <table className="admin-users-table">
                        <thead>
                          <tr>
                            <th>ID</th>
                            <th>Логин</th>
                            <th>Роль</th>
                            <th>Смена роли</th>
                            <th>Новый пароль</th>
                            <th>Удаление</th>
                          </tr>
                        </thead>
                        <tbody>
                          {adminUsers.map((row) => (
                            <tr key={row.id}>
                              <td>{row.id}</td>
                              <td>{row.username}</td>
                              <td>{row.is_admin ? "Администратор" : "Пользователь"}</td>
                              <td>
                                <div className="admin-role-actions">
                                  <select
                                    value={userRoleDrafts[row.id] ? "admin" : "user"}
                                    onChange={(event) =>
                                      setUserRoleDrafts((prev) => ({
                                        ...prev,
                                        [row.id]: event.target.value === "admin"
                                      }))
                                    }
                                  >
                                    <option value="user">Пользователь</option>
                                    <option value="admin">Администратор</option>
                                  </select>
                                  <button
                                    type="button"
                                    onClick={() => void handleAdminSaveRole(row.id)}
                                    disabled={adminActionInProgress === `role-${row.id}` || row.id === currentUser.id}
                                  >
                                    Сохранить
                                  </button>
                                </div>
                              </td>
                              <td>
                                <div className="admin-password-actions">
                                  <input
                                    type="password"
                                    value={userPasswordDrafts[row.id] ?? ""}
                                    onChange={(event) =>
                                      setUserPasswordDrafts((prev) => ({
                                        ...prev,
                                        [row.id]: event.target.value
                                      }))
                                    }
                                    placeholder="Новый пароль"
                                  />
                                  <button
                                    type="button"
                                    onClick={() => void handleAdminResetPassword(row.id)}
                                    disabled={adminActionInProgress === `password-${row.id}`}
                                  >
                                    Сбросить
                                  </button>
                                </div>
                              </td>
                              <td>
                                <button
                                  type="button"
                                  className="btn-ghost danger"
                                  onClick={() => void handleAdminDeleteUser(row.id, row.username)}
                                  disabled={adminActionInProgress === `delete-${row.id}` || row.id === currentUser.id}
                                >
                                  Удалить
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  {adminMessage ? <p className="user-inline-message">{adminMessage}</p> : null}
                </article>
              ) : null}
            </section>
          )}
        </RevealOnScroll>
      </main>
    </div>
  );
}
