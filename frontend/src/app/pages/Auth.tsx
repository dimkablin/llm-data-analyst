import { useState, type ReactNode } from "react";
import { Link, Navigate, useNavigate } from "react-router";
import {
  ArrowRight,
  Lock,
  Mail,
  Settings2,
  Shield,
  UserCog,
  UserPlus,
  Users,
  Zap,
} from "lucide-react";
import { motion } from "motion/react";
import { Navigation } from "../components/Navigation";
import { useAppSession } from "../context/AppSessionContext";
import { summarizeError } from "../lib/format";

export function Auth() {
  const navigate = useNavigate();
  const { user, login, register } = useAppSession();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [passwordRepeat, setPasswordRepeat] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (user) {
    return <Navigate to="/workspace" replace />;
  }

  async function handleSubmit(event: React.FormEvent): Promise<void> {
    event.preventDefault();
    if (!username.trim() || !password.trim()) {
      setError("Введите логин и пароль.");
      return;
    }
    if (mode === "register" && password !== passwordRepeat) {
      setError("Пароли не совпадают.");
      return;
    }

    setIsSubmitting(true);
    setError(null);
    try {
      if (mode === "login") {
        await login(username.trim(), password);
      } else {
        await register(username.trim(), password);
      }
      navigate("/workspace", { replace: true });
    } catch (submitError) {
      setError(summarizeError(submitError));
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navigation />

      <div className="mx-auto flex min-h-[calc(100vh-4rem)] w-full max-w-[1380px] items-center justify-center px-8 py-10">
        <div className="grid w-full grid-cols-1 items-stretch gap-6 xl:grid-cols-2">
          <motion.section
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex min-h-[720px] flex-col rounded-[36px] border border-border/60 bg-card/80 p-8 shadow-xl backdrop-blur-xl sm:p-9 lg:p-10 dark:bg-card/35"
          >
            <div className="mb-5 inline-flex items-center self-start rounded-full border border-primary/20 bg-primary/10 px-3.5 py-1.5 text-[12px] font-bold tracking-wide text-primary">
              Управление доступом
            </div>

            <div className="max-w-[560px]">
              <h1 className="min-h-[4.2em] text-[2.35rem] font-bold leading-[1.02] tracking-tight text-foreground sm:text-[2.9rem] lg:text-[3.35rem]">
                Единая зона доступа:
                <br />
                вход, профиль,
                <br />
                безопасность и
                <br />
                администрирование
              </h1>
              <p className="mt-4 max-w-[520px] text-[15px] leading-relaxed text-muted-foreground sm:text-[16px]">
                Пользователь входит в систему, управляет профилем и настройками.
                Администратор дополнительно получает контроль над пользователями, ролями
                и паролями.
              </p>
            </div>

            <div className="mt-6 flex flex-wrap gap-2">
              {[
                "Вход / регистрация",
                "Профиль + настройки",
                "Управление пользователями",
              ].map((item) => (
                <div
                  key={item}
                  className="rounded-full border border-border/50 bg-secondary/45 px-3 py-1.5 text-[11px] font-bold uppercase tracking-wide text-foreground/80"
                >
                  {item}
                </div>
              ))}
            </div>

            <div className="mt-6 grid flex-1 content-end gap-3 sm:grid-cols-2">
              <PromoCard
                icon={<Shield className="h-5 w-5" />}
                title="Профиль"
                text="Роль, дата создания, переход в рабочую область и выход."
              />
              <PromoCard
                icon={<Settings2 className="h-5 w-5" />}
                title="Настройки"
                text="Тема и параметры ответа, влияющие на работу интерфейса."
              />
              <PromoCard
                icon={<Lock className="h-5 w-5" />}
                title="Смена пароля"
                text="Обновление пароля без обращения к администратору."
              />
              <PromoCard
                icon={<Users className="h-5 w-5" />}
                title="Администрирование"
                text="Добавление, удаление пользователей, смена ролей и паролей."
              />
            </div>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex min-h-[720px] flex-col rounded-[36px] border border-border/60 bg-card/95 p-8 shadow-[0_24px_60px_rgba(15,23,42,0.14)] sm:p-9 lg:p-10 dark:bg-card"
          >
            <div className="mb-8 flex items-center gap-4 rounded-[28px] border border-border/50 bg-secondary/45 px-5 py-4">
              <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-lg shadow-primary/20">
                <Zap className="h-6 w-6 fill-current" />
              </div>
              <div className="min-w-0">
                <div className="text-[12px] font-bold uppercase tracking-[0.24em] text-muted-foreground">
                  Access Layer
                </div>
                <div className="mt-1 text-lg font-bold tracking-tight text-foreground">
                  {mode === "login" ? "Вход в систему" : "Регистрация"}
                </div>
              </div>
            </div>

            <div className="mb-8 text-center">
              <h2 className="mb-3 text-[2rem] font-bold tracking-tight sm:text-[2.3rem]">
                {mode === "login" ? "Вход в систему" : "Регистрация"}
              </h2>
              <p className="mx-auto max-w-md text-[15px] leading-relaxed text-muted-foreground">
                Реальная backend-аутентификация уже подключена. Эта зона работает
                как production-вход и одновременно показывает access layer продукта
                на демо.
              </p>
            </div>

            <div className="mb-8 flex rounded-2xl border border-border/50 bg-secondary/50 p-1">
              <button
                type="button"
                onClick={() => setMode("login")}
                className={`flex-1 rounded-xl px-4 py-3 text-sm font-bold transition-all ${
                  mode === "login"
                    ? "bg-card shadow-sm"
                    : "text-muted-foreground"
                }`}
              >
                Вход
              </button>
              <button
                type="button"
                onClick={() => setMode("register")}
                className={`flex-1 rounded-xl px-4 py-3 text-sm font-bold transition-all ${
                  mode === "register"
                    ? "bg-card shadow-sm"
                    : "text-muted-foreground"
                }`}
              >
                Регистрация
              </button>
            </div>

            <form
              onSubmit={(event) => void handleSubmit(event)}
              className="flex flex-1 flex-col justify-between"
            >
              <div className="space-y-5">
                <div className="space-y-2">
                  <label className="px-4 text-[12px] font-bold uppercase tracking-widest text-muted-foreground">
                    Логин
                  </label>
                  <div className="group relative">
                    <Mail className="absolute left-5 top-1/2 h-5 w-5 -translate-y-1/2 text-muted-foreground transition-colors group-focus-within:text-primary" />
                    <input
                      type="text"
                      value={username}
                      onChange={(event) => setUsername(event.target.value)}
                      placeholder="admin"
                      className="w-full rounded-2xl border border-border/50 bg-secondary py-4 pl-14 pr-5 text-[15px] font-medium transition-all focus:border-primary/50 focus:outline-none focus:ring-4 focus:ring-primary/10"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="px-4 text-[12px] font-bold uppercase tracking-widest text-muted-foreground">
                    Пароль
                  </label>
                  <div className="group relative">
                    <Lock className="absolute left-5 top-1/2 h-5 w-5 -translate-y-1/2 text-muted-foreground transition-colors group-focus-within:text-primary" />
                    <input
                      type="password"
                      value={password}
                      onChange={(event) => setPassword(event.target.value)}
                      placeholder="********"
                      className="w-full rounded-2xl border border-border/50 bg-secondary py-4 pl-14 pr-5 text-[15px] font-medium transition-all focus:border-primary/50 focus:outline-none focus:ring-4 focus:ring-primary/10"
                    />
                  </div>
                </div>

                {mode === "register" ? (
                  <div className="space-y-2">
                    <label className="px-4 text-[12px] font-bold uppercase tracking-widest text-muted-foreground">
                      Повторите пароль
                    </label>
                    <div className="group relative">
                      <Lock className="absolute left-5 top-1/2 h-5 w-5 -translate-y-1/2 text-muted-foreground transition-colors group-focus-within:text-primary" />
                      <input
                        type="password"
                        value={passwordRepeat}
                        onChange={(event) => setPasswordRepeat(event.target.value)}
                        placeholder="********"
                        className="w-full rounded-2xl border border-border/50 bg-secondary py-4 pl-14 pr-5 text-[15px] font-medium transition-all focus:border-primary/50 focus:outline-none focus:ring-4 focus:ring-primary/10"
                      />
                    </div>
                  </div>
                ) : null}
              </div>

              <div className="mt-8 space-y-5">
                {error ? (
                  <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
                    {error}
                  </div>
                ) : null}

                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="flex w-full items-center justify-center gap-3 rounded-2xl bg-primary py-4 font-bold text-primary-foreground shadow-xl shadow-primary/20 transition-all active:scale-[0.98] disabled:opacity-60"
                >
                  {mode === "login" ? (
                    <ArrowRight className="h-5 w-5" />
                  ) : (
                    <UserPlus className="h-5 w-5" />
                  )}
                  <span>
                    {isSubmitting
                      ? "Подключение..."
                      : mode === "login"
                        ? "Войти в аккаунт"
                        : "Создать аккаунт"}
                  </span>
                </button>
              </div>
            </form>

            <div className="mt-8 border-t border-border/40 pt-8 text-center">
              <div className="inline-flex items-center gap-2 text-[13px] font-medium text-muted-foreground">
                <UserCog className="h-4 w-4" />
                Backend auth, user settings и session ownership активны
              </div>
              <div className="mt-4 text-sm text-muted-foreground">
                После входа доступны{" "}
                <Link to="/workspace" className="font-semibold text-primary">
                  workspace
                </Link>
                ,{" "}
                <Link to="/sessions" className="font-semibold text-primary">
                  sessions
                </Link>{" "}
                и{" "}
                <Link to="/account" className="font-semibold text-primary">
                  account
                </Link>
                .
              </div>
            </div>
          </motion.section>
        </div>
      </div>
    </div>
  );
}

function PromoCard({
  icon,
  title,
  text,
}: {
  icon: ReactNode;
  title: string;
  text: string;
}) {
  return (
    <div className="rounded-[20px] border border-border/50 bg-secondary/30 p-4 backdrop-blur-sm">
      <div className="mb-3 flex items-center gap-3 text-foreground">
        <div className="rounded-xl border border-primary/15 bg-primary/10 p-2 text-primary">
          {icon}
        </div>
        <h3 className="text-lg font-bold tracking-tight">{title}</h3>
      </div>
      <p className="text-[14px] leading-relaxed text-muted-foreground">{text}</p>
    </div>
  );
}
