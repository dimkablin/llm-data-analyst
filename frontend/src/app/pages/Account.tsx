import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link, Navigate, useNavigate } from "react-router";
import { useTheme } from "next-themes";
import {
  BadgeCheck,
  Bell,
  Brush,
  Check,
  Cpu,
  Globe,
  KeyRound,
  LogOut,
  Palette,
  Save,
  Shield,
  User,
  Users,
} from "lucide-react";
import { Navigation } from "../components/Navigation";
import { ToolAccessSection } from "../components/account/ToolAccessSection";
import { UserMemorySection } from "../components/account/UserMemorySection";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "../components/ui/dropdown-menu";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { useAppSession } from "../context/AppSessionContext";
import { changePassword, createAdminUser, deleteAdminUser, listAdminUsers, updateAdminUser } from "../lib/backend-api";
import type { AccentId } from "../lib/accent";
import { getStoredAccent, setStoredAccent } from "../lib/accent";
import {
  ANALYSIS_DEPTH_STEP_CEILING,
  clampAgentMaxStepsForDepth,
  type AuthUser,
  type UserSettings,
} from "../lib/backend-types";
import { formatDateTime, summarizeError } from "../lib/format";

type AccountTab = "general" | "notifications" | "security" | "users" | "account";

const TABS: Array<{ id: AccountTab; label: string; icon: ReactNode }> = [
  { id: "general", label: "Общее", icon: <Brush className="h-4 w-4" /> },
  { id: "notifications", label: "Уведомления", icon: <Bell className="h-4 w-4" /> },
  { id: "security", label: "Безопасность", icon: <Shield className="h-4 w-4" /> },
  { id: "users", label: "Пользователи", icon: <Users className="h-4 w-4" /> },
  { id: "account", label: "Аккаунт", icon: <User className="h-4 w-4" /> },
];

const ACCENTS: Array<{ id: AccentId; label: string; swatch: string }> = [
  { id: "default", label: "По умолчанию", swatch: "bg-zinc-300" },
  { id: "blue", label: "Синий", swatch: "bg-blue-500" },
  { id: "green", label: "Зеленый", swatch: "bg-green-500" },
  { id: "yellow", label: "Желтый", swatch: "bg-yellow-400" },
  { id: "pink", label: "Розовый", swatch: "bg-pink-500" },
  { id: "orange", label: "Оранжевый", swatch: "bg-orange-500" },
  { id: "violet", label: "Фиолетовый", swatch: "bg-violet-500" },
];

export function Account() {
  const navigate = useNavigate();
  const { setTheme } = useTheme();
  const { user, settings, saveSettings, logout } = useAppSession();

  const [activeTab, setActiveTab] = useState<AccountTab>("general");
  const [settingsDraft, setSettingsDraft] = useState<UserSettings>(settings);
  const [settingsMessage, setSettingsMessage] = useState<string | null>(null);
  const [themeMode, setThemeMode] = useState<"system" | "light" | "dark">(settings.theme);
  const [accent, setAccent] = useState<AccentId>(() => getStoredAccent());
  const [language, setLanguage] = useState("Русский");

  const [browserNotifications, setBrowserNotifications] = useState(true);
  const [emailNotifications, setEmailNotifications] = useState(false);
  const [systemAlerts, setSystemAlerts] = useState(true);

  const [currentPassword, setCurrentPassword] = useState("");
  const [nextPassword, setNextPassword] = useState("");
  const [nextPasswordRepeat, setNextPasswordRepeat] = useState("");
  const [passwordMessage, setPasswordMessage] = useState<string | null>(null);

  const [adminUsers, setAdminUsers] = useState<AuthUser[]>([]);
  const [adminMessage, setAdminMessage] = useState<string | null>(null);
  const [newUsername, setNewUsername] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [newIsAdmin, setNewIsAdmin] = useState(false);

  const selectedAccent = useMemo(
    () => ACCENTS.find((option) => option.id === accent) ?? ACCENTS[0],
    [accent],
  );

  useEffect(() => {
    setSettingsDraft(settings);
    setThemeMode(settings.theme);
    setSettingsMessage(null);
  }, [settings]);

  useEffect(() => {
    setStoredAccent(accent);
  }, [accent]);

  useEffect(() => {
    if (user?.is_admin) void refreshUsers();
  }, [user?.is_admin]);

  if (!user) return <Navigate to="/auth" replace />;

  async function refreshUsers(): Promise<void> {
    try {
      setAdminUsers(await listAdminUsers());
    } catch (error) {
      setAdminMessage(summarizeError(error));
    }
  }

  async function handleSaveSettings(): Promise<void> {
    try {
      const updated = await saveSettings(settingsDraft);
      setSettingsDraft(updated);
      setTheme(themeMode);
      setSettingsMessage("Настройки сохранены.");
    } catch (error) {
      setSettingsMessage(summarizeError(error));
    }
  }

  async function handlePasswordChange(): Promise<void> {
    if (!currentPassword || !nextPassword) {
      setPasswordMessage("Заполните все поля.");
      return;
    }
    if (nextPassword !== nextPasswordRepeat) {
      setPasswordMessage("Новый пароль и подтверждение не совпадают.");
      return;
    }
    try {
      await changePassword(currentPassword, nextPassword);
      setCurrentPassword("");
      setNextPassword("");
      setNextPasswordRepeat("");
      setPasswordMessage("Пароль обновлен.");
    } catch (error) {
      setPasswordMessage(summarizeError(error));
    }
  }

  async function handleAddUser(): Promise<void> {
    if (!newUsername.trim() || !newPassword.trim()) {
      setAdminMessage("Введите логин и пароль.");
      return;
    }
    try {
      await createAdminUser(newUsername.trim(), newPassword, newIsAdmin);
      setNewUsername("");
      setNewPassword("");
      setNewIsAdmin(false);
      await refreshUsers();
    } catch (error) {
      setAdminMessage(summarizeError(error));
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navigation />
      <main className="mx-auto max-w-[1600px] px-6 py-8">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
          <aside className="h-fit rounded-[28px] border border-border/50 bg-card/45 p-6 shadow-xl backdrop-blur-xl lg:sticky lg:top-24">
            <div className="mb-5 inline-flex items-center rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-[11px] font-bold uppercase tracking-wider text-emerald-400">Аккаунт</div>
            <h1 className="text-3xl font-bold leading-tight tracking-tight">Настройки аккаунта</h1>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">Разделы сгруппированы по смыслу: внешний вид, уведомления, безопасность и управление доступом.</p>
            <div className="mt-6 space-y-3">
              <InfoField label="Логин" value={user.username} />
              <InfoField label="Роль" value={user.is_admin ? "Администратор" : "Пользователь"} />
            </div>
            <div className="mt-7 space-y-2">
              {TABS.map((tab) => (
                <button key={tab.id} type="button" onClick={() => setActiveTab(tab.id)} className={`flex w-full items-center gap-2 rounded-xl px-3 py-2.5 text-left text-sm font-medium transition-colors ${activeTab === tab.id ? "bg-primary/15 text-primary ring-1 ring-primary/30" : "border border-border/50 bg-secondary/30 text-muted-foreground hover:bg-secondary/60 hover:text-foreground"}`}>
                  {tab.icon}
                  {tab.label}
                </button>
              ))}
            </div>
          </aside>

          <section className="space-y-5">
            {activeTab === "general" ? (
              <SectionBlock title="Общее" subtitle="Внешний вид, язык и базовые параметры интерфейса." icon={<Brush className="h-4 w-4" />}>
                <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
                  <Card title="Внешний вид" subtitle="Сочетание темы и акцента применяется ко всему интерфейсу." icon={<Palette className="h-4 w-4" />}>
                    <Field label="Тема">
                      <Select
                        value={themeMode}
                        onValueChange={(value) => {
                          const next = value === "system" ? "system" : value === "light" ? "light" : "dark";
                          setThemeMode(next);
                          setTheme(next);
                          if (next !== "system") setSettingsDraft((prev) => ({ ...prev, theme: next }));
                        }}
                      >
                        <SelectTrigger className="h-11 w-full rounded-xl border border-border/60 bg-secondary/70 px-4 text-sm">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="system">Системная</SelectItem>
                          <SelectItem value="light">Светлая</SelectItem>
                          <SelectItem value="dark">Темная</SelectItem>
                        </SelectContent>
                      </Select>
                    </Field>

                    <Field label="Акцентный цвет">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <button type="button" className="flex h-11 w-full items-center justify-between rounded-xl border border-border/60 bg-secondary/70 px-4 text-sm">
                            <span className="flex items-center gap-3">
                              <span className={`h-3.5 w-3.5 rounded-full ${selectedAccent.swatch}`}></span>
                              <span>{selectedAccent.label}</span>
                            </span>
                            <Palette className="h-4 w-4 text-muted-foreground" />
                          </button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="start" className="w-[240px] rounded-2xl border-border/60 bg-popover/95 p-2 shadow-2xl backdrop-blur-xl">
                          {ACCENTS.map((option) => (
                            <DropdownMenuItem key={option.id} onSelect={() => setAccent(option.id)} className={`rounded-xl px-3 py-2.5 ${accent === option.id ? "bg-accent text-accent-foreground" : ""}`}>
                              <span className="flex min-w-0 flex-1 items-center gap-3">
                                <span className={`h-3.5 w-3.5 rounded-full ${option.swatch}`}></span>
                                <span className="font-medium">{option.label}</span>
                              </span>
                              {accent === option.id ? <Check className="ml-2 h-4 w-4 text-primary" /> : null}
                            </DropdownMenuItem>
                          ))}
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </Field>
                  </Card>

                  <Card title="Локализация и ответы" subtitle="Язык интерфейса и формат вывода аналитики." icon={<Globe className="h-4 w-4" />}>
                    <Field label="Язык">
                      <Select value={language} onValueChange={setLanguage}>
                        <SelectTrigger className="h-11 w-full rounded-xl border border-border/60 bg-secondary/70 px-4 text-sm">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="Русский">Русский</SelectItem>
                          <SelectItem value="English">English</SelectItem>
                        </SelectContent>
                      </Select>
                    </Field>
                    <label className="inline-flex items-center gap-2 text-sm font-medium">
                      <input type="checkbox" checked={settingsDraft.default_include_reasoning} onChange={(event) => setSettingsDraft((prev) => ({ ...prev, default_include_reasoning: event.target.checked }))} className="h-4 w-4 accent-primary" />
                      Показывать reasoning по умолчанию
                    </label>
                    <Field label="Стиль ответа">
                      <Select value={settingsDraft.default_answer_style} onValueChange={(value) => setSettingsDraft((prev) => ({ ...prev, default_answer_style: value === "concise" ? "concise" : "detailed" }))}>
                        <SelectTrigger className="h-11 w-full rounded-xl border border-border/60 bg-secondary/70 px-4 text-sm">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="detailed">Развернутый</SelectItem>
                          <SelectItem value="concise">Краткий</SelectItem>
                        </SelectContent>
                      </Select>
                    </Field>
                    <Field label="Глубина анализа">
                      <Select
                        value={settingsDraft.analysis_depth}
                        onValueChange={(value) => {
                          const depth = value === "medium" ? "medium" : value === "deep" ? "deep" : "light";
                          setSettingsDraft((prev) => ({
                            ...prev,
                            analysis_depth: depth,
                            agent_max_steps: clampAgentMaxStepsForDepth(depth, prev.agent_max_steps),
                          }));
                        }}
                      >
                        <SelectTrigger className="h-11 w-full rounded-xl border border-border/60 bg-secondary/70 px-4 text-sm">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="light">Легкий</SelectItem>
                          <SelectItem value="medium">Средний</SelectItem>
                          <SelectItem value="deep">Глубокий</SelectItem>
                        </SelectContent>
                      </Select>
                    </Field>
                    <button type="button" onClick={() => void handleSaveSettings()} className="inline-flex items-center gap-2 rounded-xl bg-primary px-5 py-2.5 text-sm font-bold text-primary-foreground shadow-lg shadow-primary/20">
                      <Save className="h-4 w-4" />
                      Сохранить настройки
                    </button>
                    {settingsMessage ? <p className="text-sm text-muted-foreground">{settingsMessage}</p> : null}
                  </Card>

                  <div className="xl:col-span-2">
                    <Card title="Параметры backend" subtitle="Тонкие настройки из /auth/settings для runtime-профиля агента." icon={<Cpu className="h-4 w-4" />}>
                      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
                        <NumField label="Темп. чата" value={settingsDraft.llm_temperature_chat} step={0.05} onChange={(value) => setSettingsDraft((prev) => ({ ...prev, llm_temperature_chat: value }))} />
                        <NumField label="Темп. инструментов" value={settingsDraft.llm_temperature_tool} step={0.05} onChange={(value) => setSettingsDraft((prev) => ({ ...prev, llm_temperature_tool: value }))} />
                        <NumField label="Макс. токенов" value={settingsDraft.llm_max_tokens_default} step={128} onChange={(value) => setSettingsDraft((prev) => ({ ...prev, llm_max_tokens_default: Math.round(value) }))} />
                        <NumField label="Токены reasoning" value={settingsDraft.llm_max_tokens_reasoning} step={128} onChange={(value) => setSettingsDraft((prev) => ({ ...prev, llm_max_tokens_reasoning: Math.round(value) }))} />
                        <NumField label="Timeout backend, сек" value={settingsDraft.backend_query_timeout_sec} step={5} onChange={(value) => setSettingsDraft((prev) => ({ ...prev, backend_query_timeout_sec: Math.round(value) }))} />
                        <NumField
                          label="Макс. шагов"
                          value={settingsDraft.agent_max_steps}
                          min={2}
                          max={ANALYSIS_DEPTH_STEP_CEILING[settingsDraft.analysis_depth]}
                          hint={`Потолок с уровнем: ${ANALYSIS_DEPTH_STEP_CEILING[settingsDraft.analysis_depth]}. Раньше — по решению модели.`}
                          step={1}
                          onChange={(value) =>
                            setSettingsDraft((prev) => ({
                              ...prev,
                              agent_max_steps: clampAgentMaxStepsForDepth(prev.analysis_depth, value),
                            }))
                          }
                        />
                        <NumField label="Timeout шага, сек" value={settingsDraft.agent_step_timeout_sec} step={5} onChange={(value) => setSettingsDraft((prev) => ({ ...prev, agent_step_timeout_sec: Math.round(value) }))} />
                        <NumField label="Внутр. рекурсия" value={settingsDraft.agent_inner_recursion_limit} step={1} onChange={(value) => setSettingsDraft((prev) => ({ ...prev, agent_inner_recursion_limit: Math.round(value) }))} />
                      </div>
                    </Card>
                  </div>

                  <div className="xl:col-span-2">
                    <ToolAccessSection />
                  </div>
                </div>
              </SectionBlock>
            ) : null}

            {activeTab === "notifications" ? (
              <SectionBlock title="Уведомления" subtitle="Настройка каналов уведомлений и оповещений." icon={<Bell className="h-4 w-4" />}>
                <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                  <ToggleCard title="Браузерные уведомления" desc="Оповещения о завершении анализа и готовности артефактов." enabled={browserNotifications} onChange={setBrowserNotifications} />
                  <ToggleCard title="Email-уведомления" desc="Дублировать важные события на привязанный email." enabled={emailNotifications} onChange={setEmailNotifications} />
                  <ToggleCard title="Системные оповещения" desc="Показывать сообщения о состоянии runtime и инфраструктуры." enabled={systemAlerts} onChange={setSystemAlerts} />
                </div>
              </SectionBlock>
            ) : null}

            {activeTab === "security" ? (
              <SectionBlock title="Безопасность" subtitle="Смена пароля и контроль текущей сессии." icon={<Shield className="h-4 w-4" />}>
                <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
                  <Card title="Смена пароля" subtitle="Подключено к backend /auth/change-password." icon={<KeyRound className="h-4 w-4" />}>
                    <Input label="Текущий пароль" value={currentPassword} onChange={setCurrentPassword} type="password" />
                    <Input label="Новый пароль" value={nextPassword} onChange={setNextPassword} type="password" />
                    <Input label="Повторите новый пароль" value={nextPasswordRepeat} onChange={setNextPasswordRepeat} type="password" />
                    <button type="button" onClick={() => void handlePasswordChange()} className="inline-flex items-center gap-2 rounded-xl bg-primary px-5 py-2.5 text-sm font-bold text-primary-foreground">
                      <KeyRound className="h-4 w-4" />
                      Обновить пароль
                    </button>
                    {passwordMessage ? <p className="text-sm text-muted-foreground">{passwordMessage}</p> : null}
                  </Card>
                  <Card title="Сессия" subtitle="Выход и переход в рабочую область." icon={<LogOut className="h-4 w-4" />}>
                    <div className="rounded-xl border border-border/50 bg-secondary/25 p-3 text-sm text-muted-foreground">Текущая backend-сессия связана с bearer token и ownership моделей/чатов.</div>
                    <Link to="/workspace" className="rounded-xl bg-secondary px-5 py-2.5 text-sm font-bold hover:bg-muted">Открыть workspace</Link>
                  </Card>
                </div>
              </SectionBlock>
            ) : null}

            {activeTab === "users" ? (
              <SectionBlock title="Управление пользователями" subtitle="Рабочая admin-функциональность старого frontend, встроенная в новый интерфейс." icon={<Users className="h-4 w-4" />}>
                {!user.is_admin ? (
                  <div className="rounded-2xl border border-border/50 bg-secondary/25 p-5 text-sm text-muted-foreground">Раздел сохранен в UI, но доступен только администраторам.</div>
                ) : (
                  <div className="space-y-5">
                    <div className="grid grid-cols-1 gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] md:items-end">
                      <Input label="Новый логин" value={newUsername} onChange={setNewUsername} />
                      <Input label="Временный пароль" value={newPassword} onChange={setNewPassword} type="password" />
                      <button type="button" onClick={() => void handleAddUser()} className="h-11 rounded-xl bg-primary px-5 text-sm font-bold text-primary-foreground">Создать</button>
                    </div>
                    <label className="inline-flex items-center gap-2 text-sm font-medium">
                      <input type="checkbox" checked={newIsAdmin} onChange={(event) => setNewIsAdmin(event.target.checked)} className="h-4 w-4 accent-primary" />
                      Назначить администратором
                    </label>
                    <div className="overflow-x-auto rounded-2xl border border-border/60">
                      <table className="w-full min-w-[760px] border-collapse text-left">
                        <thead className="bg-secondary/60">
                          <tr>
                            <th className="px-4 py-3 text-xs font-bold uppercase tracking-wider text-muted-foreground">ID</th>
                            <th className="px-4 py-3 text-xs font-bold uppercase tracking-wider text-muted-foreground">Логин</th>
                            <th className="px-4 py-3 text-xs font-bold uppercase tracking-wider text-muted-foreground">Роль</th>
                            <th className="px-4 py-3 text-xs font-bold uppercase tracking-wider text-muted-foreground">Сменить пароль</th>
                            <th className="px-4 py-3 text-xs font-bold uppercase tracking-wider text-muted-foreground">Действия</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border/40">
                          {adminUsers.map((row) => (
                            <AdminUserRow key={row.id} row={row} currentUserId={user.id} onRefresh={refreshUsers} />
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {adminMessage ? <p className="text-sm text-muted-foreground">{adminMessage}</p> : null}
                  </div>
                )}
              </SectionBlock>
            ) : null}

            {activeTab === "account" ? (
              <>
                <SectionBlock title="Аккаунт" subtitle="Профиль, дата создания и быстрые переходы." icon={<User className="h-4 w-4" />}>
                  <div className="space-y-5">
                    <div className="rounded-2xl border border-border/50 bg-secondary/30 p-5">
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div className="space-y-1">
                          <p className="text-sm text-muted-foreground">Логин: <span className="font-semibold text-foreground">{user.username}</span></p>
                          <p className="text-sm text-muted-foreground">Роль: <span className="font-semibold text-foreground">{user.is_admin ? "Администратор" : "Пользователь"}</span></p>
                          <p className="text-sm text-muted-foreground">Создан: {formatDateTime(user.created_at)}</p>
                        </div>
                        <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-xs font-bold uppercase tracking-wider text-emerald-400">
                          <BadgeCheck className="h-4 w-4" />
                          active
                        </div>
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-3">
                      <Link to="/workspace" className="inline-flex items-center justify-center rounded-xl bg-primary px-5 py-2.5 text-sm font-bold text-primary-foreground">Открыть workspace</Link>
                      <button type="button" onClick={() => { void logout().then(() => navigate("/auth", { replace: true })); }} className="inline-flex items-center justify-center gap-2 rounded-xl border border-border/60 bg-secondary px-5 py-2.5 text-sm font-bold hover:bg-muted">
                        <LogOut className="h-4 w-4" />
                        Выйти из аккаунта
                      </button>
                    </div>
                  </div>
                </SectionBlock>

                <SectionBlock title="Память" subtitle="Персональный контекст агента для текущего пользователя." icon={<Cpu className="h-4 w-4" />}>
                  <UserMemorySection />
                </SectionBlock>
              </>
            ) : null}
          </section>
        </div>
      </main>
    </div>
  );
}

function AdminUserRow({
  row,
  currentUserId,
  onRefresh,
}: {
  row: AuthUser;
  currentUserId: number;
  onRefresh: () => Promise<void>;
}) {
  const [role, setRole] = useState(row.is_admin);
  const [password, setPassword] = useState("");

  return (
    <tr className="bg-background/10">
      <td className="px-4 py-3 text-sm font-semibold">{row.id}</td>
      <td className="px-4 py-3 text-sm font-semibold">{row.username}</td>
      <td className="px-4 py-3">
        <Select value={role ? "admin" : "user"} onValueChange={(value) => setRole(value === "admin")}>
          <SelectTrigger className="h-10 rounded-lg border border-border/60 bg-secondary/60 px-3 text-sm">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="user">Пользователь</SelectItem>
            <SelectItem value="admin">Администратор</SelectItem>
          </SelectContent>
        </Select>
      </td>
      <td className="px-4 py-3">
        <input value={password} onChange={(event) => setPassword(event.target.value)} className="h-10 w-full rounded-lg border border-border/60 bg-secondary/60 px-3 text-sm" />
      </td>
      <td className="px-4 py-3">
        <div className="flex gap-2">
          <button type="button" onClick={() => void updateAdminUser(row.id, { is_admin: role, password: password || undefined }).then(onRefresh)} className="rounded-lg bg-primary/80 px-3 py-2 text-xs font-bold text-primary-foreground">Сохранить</button>
          <button type="button" disabled={row.id === currentUserId} onClick={() => void deleteAdminUser(row.id).then(onRefresh)} className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs font-bold text-rose-400 disabled:opacity-50">Удалить</button>
        </div>
      </td>
    </tr>
  );
}

function SectionBlock({ title, subtitle, icon, children }: { title: string; subtitle: string; icon: ReactNode; children: ReactNode }) {
  return (
    <section className="rounded-[30px] border border-border/55 bg-card/42 p-6 shadow-xl backdrop-blur-xl">
      <div className="mb-5 border-b border-border/35 pb-4">
        <div className="flex items-center gap-2">
          <span className="rounded-lg bg-primary/10 p-1.5 text-primary">{icon}</span>
          <h2 className="text-2xl font-bold tracking-tight">{title}</h2>
        </div>
        <p className="mt-2 text-sm text-muted-foreground">{subtitle}</p>
      </div>
      {children}
    </section>
  );
}

function Card({ title, subtitle, icon, children }: { title: string; subtitle: string; icon: ReactNode; children: ReactNode }) {
  return (
    <div className="space-y-4 rounded-2xl border border-border/50 bg-secondary/25 p-5">
      <div>
        <div className="flex items-center gap-2">
          <span className="rounded-lg bg-primary/10 p-1.5 text-primary">{icon}</span>
          <h3 className="text-lg font-bold tracking-tight">{title}</h3>
        </div>
        <p className="mt-2 text-sm text-muted-foreground">{subtitle}</p>
      </div>
      {children}
    </div>
  );
}

function ToggleCard({ title, desc, enabled, onChange }: { title: string; desc: string; enabled: boolean; onChange: (value: boolean) => void }) {
  return (
    <div className="rounded-2xl border border-border/50 bg-secondary/25 p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-[15px] font-semibold tracking-tight">{title}</h3>
          <p className="mt-2 text-sm text-muted-foreground">{desc}</p>
        </div>
        <button type="button" onClick={() => onChange(!enabled)} className={`relative mt-1 inline-flex h-6 w-11 flex-shrink-0 items-center rounded-full transition-colors ${enabled ? "bg-primary" : "bg-border/70"}`} aria-pressed={enabled} aria-label={title}>
          <span className={`inline-block h-5 w-5 transform rounded-full bg-white transition-transform ${enabled ? "translate-x-5" : "translate-x-0.5"}`} />
        </button>
      </div>
    </div>
  );
}

function NumField({
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
      <label className="text-xs font-bold uppercase tracking-widest text-muted-foreground">{label}</label>
      {hint ? <p className="text-[10px] leading-snug text-muted-foreground">{hint}</p> : null}
      <input
        type="number"
        value={Number.isFinite(value) ? value : 0}
        step={step}
        min={min}
        max={max}
        onChange={(event) => {
          const parsed = parseFloat(event.target.value);
          onChange(Number.isFinite(parsed) ? parsed : 0);
        }}
        className="h-11 w-full rounded-xl border border-border/60 bg-secondary/70 px-4 text-sm outline-none transition-colors focus:border-primary/50"
      />
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="space-y-2">
      <label className="text-xs font-bold uppercase tracking-widest text-muted-foreground">{label}</label>
      {children}
    </div>
  );
}

function Input({ label, value, onChange, type = "text" }: { label: string; value: string; onChange: (value: string) => void; type?: string }) {
  return (
    <div className="space-y-2">
      <label className="text-xs font-bold uppercase tracking-widest text-muted-foreground">{label}</label>
      <input value={value} onChange={(event) => onChange(event.target.value)} type={type} className="h-11 w-full rounded-xl border border-border/60 bg-secondary/70 px-4 text-sm outline-none transition-colors focus:border-primary/50" />
    </div>
  );
}

function InfoField({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-border/50 bg-secondary/45 px-4 py-2.5">
      <div className="text-xs font-bold uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="text-sm font-semibold">{value}</div>
    </div>
  );
}
