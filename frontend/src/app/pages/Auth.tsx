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

      <div className="mx-auto flex min-h-[calc(100svh-3.5rem)] w-full max-w-[1380px] items-start justify-center px-4 py-4 sm:px-6 sm:py-6 lg:px-8 xl:items-center xl:py-4 2xl:py-10 [@media(max-height:800px)]:py-3">
        <div className="grid w-full grid-cols-1 items-stretch gap-4 lg:gap-5 2xl:gap-6 xl:grid-cols-[1.04fr_0.96fr]">
          <motion.section
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col rounded-[28px] border border-border/60 bg-card/80 p-5 shadow-xl backdrop-blur-xl sm:p-6 lg:p-7 xl:min-h-[600px] 2xl:min-h-[720px] 2xl:rounded-[36px] 2xl:p-10 [@media(max-height:800px)]:p-4 [@media(max-height:800px)]:xl:min-h-[540px] dark:bg-card/35"
          >
            <div className="mb-3 inline-flex items-center self-start rounded-full border border-primary/20 bg-primary/10 px-3 py-1 text-[10px] font-bold tracking-wide text-primary sm:px-3.5 sm:py-1.5 sm:text-[12px] [@media(max-height:800px)]:mb-2.5">
              Управление доступом
            </div>

            <div className="max-w-[560px] xl:max-w-[520px] 2xl:max-w-[560px]">
              <h1 className="text-balance text-[1.85rem] font-bold leading-[0.98] tracking-tight text-foreground sm:text-[2.25rem] lg:text-[2.45rem] xl:text-[2.5rem] 2xl:text-[3.35rem] [@media(max-height:800px)]:text-[1.7rem] [@media(max-height:800px)]:xl:text-[2.15rem]">
                Единая зона доступа: вход, профиль, безопасность и администрирование
              </h1>
              <p className="mt-3 max-w-[520px] text-[13px] leading-relaxed text-muted-foreground sm:mt-4 sm:text-[15px] 2xl:text-[16px] [@media(max-height:800px)]:mt-2.5 [@media(max-height:800px)]:text-[12px]">
                Пользователь входит в систему, управляет профилем и настройками.
                Администратор дополнительно получает контроль над пользователями, ролями
                и паролями.
              </p>
            </div>

            <div className="mt-4 flex flex-wrap gap-2 2xl:mt-6 [@media(max-height:800px)]:mt-3 [@media(max-height:800px)]:gap-1.5">
              {[
                "Вход / регистрация",
                "Профиль + настройки",
                "Управление пользователями",
              ].map((item) => (
                <div
                  key={item}
                  className="rounded-full border border-border/50 bg-secondary/45 px-2.5 py-1 text-[9px] font-bold uppercase tracking-wide text-foreground/80 sm:px-3 sm:py-1.5 sm:text-[11px]"
                >
                  {item}
                </div>
              ))}
            </div>

            <div className="mt-4 grid flex-1 content-end gap-2.5 sm:grid-cols-2 2xl:mt-6 2xl:gap-3 [@media(max-height:800px)]:mt-3">
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
            className="flex flex-col rounded-[28px] border border-border/60 bg-card/95 p-5 shadow-[0_24px_60px_rgba(15,23,42,0.14)] sm:p-6 lg:p-7 xl:min-h-[600px] 2xl:min-h-[720px] 2xl:rounded-[36px] 2xl:p-10 [@media(max-height:800px)]:p-4 [@media(max-height:800px)]:xl:min-h-[540px] dark:bg-card"
          >
            <div className="mb-4 flex items-center gap-3 rounded-[24px] border border-border/50 bg-secondary/45 px-4 py-3 sm:gap-4 sm:px-5 sm:py-4 2xl:mb-8 2xl:rounded-[28px] [@media(max-height:800px)]:mb-3 [@media(max-height:800px)]:py-2.5">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-lg shadow-primary/20 sm:h-12 sm:w-12 2xl:h-14 2xl:w-14 2xl:rounded-2xl">
                <Zap className="h-5 w-5 fill-current sm:h-6 sm:w-6" />
              </div>
              <div className="min-w-0">
                <div className="text-[11px] font-bold uppercase tracking-[0.22em] text-muted-foreground sm:text-[12px]">
                  Слой доступа
                </div>
                <div className="mt-0.5 text-[15px] font-bold tracking-tight text-foreground sm:mt-1 sm:text-lg">
                  {mode === "login" ? "Вход в систему" : "Регистрация"}
                </div>
              </div>
            </div>

            <div className="mb-4 text-center 2xl:mb-8 [@media(max-height:800px)]:mb-3">
              <h2 className="mb-2 text-[1.55rem] font-bold tracking-tight sm:mb-3 sm:text-[1.85rem] 2xl:text-[2.3rem] [@media(max-height:800px)]:text-[1.4rem]">
                {mode === "login" ? "Вход в систему" : "Регистрация"}
              </h2>
              <p className="mx-auto max-w-md text-[13px] leading-relaxed text-muted-foreground sm:text-[15px] [@media(max-height:800px)]:text-[12px]">
                Реальная аутентификация бэкенда уже подключена. Эта зона работает
                как промышленный вход и одновременно показывает слой доступа продукта
                на демо.
              </p>
            </div>

            <div className="mb-4 flex rounded-2xl border border-border/50 bg-secondary/50 p-1 2xl:mb-8 [@media(max-height:800px)]:mb-3">
              <button
                type="button"
                onClick={() => setMode("login")}
                className={`flex-1 rounded-xl px-3 py-2.5 text-sm font-bold transition-all sm:px-4 2xl:py-3 ${
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
                className={`flex-1 rounded-xl px-3 py-2.5 text-sm font-bold transition-all sm:px-4 2xl:py-3 ${
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
              <div className="space-y-3.5 2xl:space-y-5 [@media(max-height:800px)]:space-y-3">
                <div className="space-y-2">
                  <label className="px-3 text-[11px] font-bold uppercase tracking-widest text-muted-foreground sm:px-4 sm:text-[12px]">
                    Логин
                  </label>
                  <div className="group relative">
                    <Mail className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground transition-colors group-focus-within:text-primary sm:left-5 sm:h-5 sm:w-5" />
                    <input
                      type="text"
                      value={username}
                      onChange={(event) => setUsername(event.target.value)}
                      placeholder="admin"
                      className="w-full rounded-2xl border border-border/50 bg-secondary py-3 pl-12 pr-4 text-sm font-medium transition-all focus:border-primary/50 focus:outline-none focus:ring-4 focus:ring-primary/10 sm:py-4 sm:pl-14 sm:pr-5 sm:text-[15px] [@media(max-height:800px)]:py-2.5"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="px-3 text-[11px] font-bold uppercase tracking-widest text-muted-foreground sm:px-4 sm:text-[12px]">
                    Пароль
                  </label>
                  <div className="group relative">
                    <Lock className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground transition-colors group-focus-within:text-primary sm:left-5 sm:h-5 sm:w-5" />
                    <input
                      type="password"
                      value={password}
                      onChange={(event) => setPassword(event.target.value)}
                      placeholder="********"
                      className="w-full rounded-2xl border border-border/50 bg-secondary py-3 pl-12 pr-4 text-sm font-medium transition-all focus:border-primary/50 focus:outline-none focus:ring-4 focus:ring-primary/10 sm:py-4 sm:pl-14 sm:pr-5 sm:text-[15px] [@media(max-height:800px)]:py-2.5"
                    />
                  </div>
                </div>

                {mode === "register" ? (
                  <div className="space-y-2">
                    <label className="px-3 text-[11px] font-bold uppercase tracking-widest text-muted-foreground sm:px-4 sm:text-[12px]">
                      Повторите пароль
                    </label>
                    <div className="group relative">
                      <Lock className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground transition-colors group-focus-within:text-primary sm:left-5 sm:h-5 sm:w-5" />
                      <input
                        type="password"
                        value={passwordRepeat}
                        onChange={(event) => setPasswordRepeat(event.target.value)}
                        placeholder="********"
                        className="w-full rounded-2xl border border-border/50 bg-secondary py-3 pl-12 pr-4 text-sm font-medium transition-all focus:border-primary/50 focus:outline-none focus:ring-4 focus:ring-primary/10 sm:py-4 sm:pl-14 sm:pr-5 sm:text-[15px] [@media(max-height:800px)]:py-2.5"
                      />
                    </div>
                  </div>
                ) : null}
              </div>

              <div className="mt-4 space-y-3.5 2xl:mt-8 2xl:space-y-5 [@media(max-height:800px)]:mt-3 [@media(max-height:800px)]:space-y-3">
                {error ? (
                  <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
                    {error}
                  </div>
                ) : null}

                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="flex w-full items-center justify-center gap-3 rounded-2xl bg-primary py-3 text-sm font-bold text-primary-foreground shadow-xl shadow-primary/20 transition-all active:scale-[0.98] disabled:opacity-60 sm:text-base 2xl:py-4 [@media(max-height:800px)]:py-2.5"
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

            <div className="mt-4 border-t border-border/40 pt-4 text-center 2xl:mt-8 2xl:pt-8 [@media(max-height:800px)]:mt-3 [@media(max-height:800px)]:pt-3">
              <div className="inline-flex items-center gap-2 text-[11px] font-medium text-muted-foreground sm:text-[13px]">
                <UserCog className="h-4 w-4" />
                Аутентификация бэкенда, настройки пользователя и владение сессиями активны
              </div>
              <div className="mt-2 text-[13px] text-muted-foreground sm:mt-4 sm:text-sm">
                После входа доступны{" "}
                <Link to="/workspace" className="font-semibold text-primary">
                  рабочая область
                </Link>
                ,{" "}
                <Link to="/sessions" className="font-semibold text-primary">
                  сессии
                </Link>{" "}
                и{" "}
                <Link to="/account" className="font-semibold text-primary">
                  аккаунт
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
    <div className="rounded-[18px] border border-border/50 bg-secondary/30 p-3 backdrop-blur-sm sm:p-4 2xl:rounded-[20px] [@media(max-height:800px)]:p-2.5">
      <div className="mb-2 flex items-center gap-2.5 text-foreground sm:mb-3 sm:gap-3">
        <div className="rounded-lg border border-primary/15 bg-primary/10 p-1.5 text-primary sm:rounded-xl sm:p-2">
          {icon}
        </div>
        <h3 className="text-[15px] font-bold tracking-tight sm:text-lg">{title}</h3>
      </div>
      <p className="text-[12px] leading-relaxed text-muted-foreground sm:text-[14px]">{text}</p>
    </div>
  );
}
