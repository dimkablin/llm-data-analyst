import { useMemo } from "react";
import { Link, useLocation } from "react-router";
import { Bell, Search, User } from "lucide-react";
import { motion } from "motion/react";
import { ThemeToggle } from "./ThemeToggle";
import { useAppSession } from "../context/AppSessionContext";

export function Navigation() {
  const location = useLocation();
  const { user } = useAppSession();
  const signedIn = Boolean(user);

  const navItems = [
    { path: "/", label: "Платформа" },
    { path: "/technical", label: "Архитектура" },
    { path: "/workspace", label: "Рабочая область" },
    { path: "/sessions", label: "Сессии" },
    { path: "/tracing", label: "Трассировка" },
  ];

  const accountLabel = useMemo(() => {
    if (!user) {
      return "Войти";
    }
    return user.is_admin ? `${user.username} • admin` : user.username;
  }, [user]);

  return (
    <nav className="sticky top-0 z-50 border-b border-border/50 bg-background/60 backdrop-blur-xl">
      <div className="mx-auto grid h-16 max-w-[1600px] grid-cols-[1fr_auto_1fr] items-center px-4 sm:h-20 sm:px-6 lg:px-8">
        <div className="flex min-w-0 items-center gap-3 sm:gap-4">
          <Link to="/" className="group flex shrink-0 items-center gap-3 transition-all">
            <div className="relative flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden rounded-xl bg-primary shadow-lg shadow-primary/25 transition-all group-hover:shadow-primary/40">
              <svg viewBox="83 130 320 230" fill="currentColor" className="h-6 w-6 text-primary-foreground">
                <rect x="175.67" y="135" width="44.662" height="220.553"/>
                <rect x="353.215" y="135" width="44.662" height="220.553"/>
                <rect x="88" y="223.221" width="220.553" height="44.662"/>
              </svg>
            </div>
            <div className="hidden min-w-0 flex-col sm:flex">
              <span className="whitespace-nowrap text-[16px] font-semibold leading-tight tracking-tight text-foreground transition-colors group-hover:text-primary lg:text-[17px]">
                Генеративная аналитика
              </span>
              <span className="mt-0.5 whitespace-nowrap text-[11px] font-medium leading-none text-muted-foreground">
                БЮ ИИ-консалтинг, Домен ИИ
              </span>
            </div>
          </Link>
        </div>

        <div className="hidden items-center overflow-x-auto rounded-xl border border-border/40 bg-secondary/50 p-1.5 backdrop-blur-md [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden lg:flex">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className="relative shrink-0 whitespace-nowrap rounded-lg px-4 py-2 text-[14px] font-medium transition-colors duration-200 xl:px-5 xl:text-[14px] 2xl:px-6"
              >
                {isActive && (
                  <motion.span
                    layoutId="nav-pill"
                    className="absolute inset-0 rounded-lg bg-card shadow-sm ring-1 ring-border/20"
                    transition={{ type: "spring", stiffness: 400, damping: 35 }}
                  />
                )}
                <span className={`relative z-10 transition-colors duration-200 ${isActive ? "text-foreground" : "text-muted-foreground hover:text-foreground"}`}>
                  {item.label}
                </span>
              </Link>
            );
          })}
        </div>

        <div className="flex shrink-0 items-center justify-end gap-2 sm:gap-2.5 lg:gap-3">
          <Link
            to="/sessions?focus=search"
            aria-label="Поиск по чатам"
            className="rounded-full border border-transparent p-2.5 text-muted-foreground transition-all hover:border-border/50 hover:bg-secondary hover:text-foreground sm:p-3"
          >
            <Search className="h-5 w-5" />
          </Link>
          <ThemeToggle />
          <button className="relative rounded-full border border-transparent p-2.5 text-muted-foreground transition-all hover:border-border/50 hover:bg-secondary hover:text-foreground sm:p-3">
            <Bell className="h-5 w-5" />
            <span className="absolute right-2.5 top-2.5 h-2 w-2 rounded-full bg-primary ring-2 ring-background"></span>
          </button>
          <div className="mx-1 hidden h-7 w-[1px] bg-border/50 sm:block"></div>
          <Link
            to={signedIn ? "/account" : "/auth"}
            className="flex items-center gap-2.5 rounded-xl border border-border/50 bg-secondary px-4 py-2.5 transition-all hover:border-border hover:bg-muted active:scale-95 sm:px-5"
          >
            <div className="flex h-7 w-7 items-center justify-center rounded-full bg-primary/20">
              <User className="h-4 w-4 text-primary" />
            </div>
            <span className="max-w-[120px] truncate text-[14px] font-medium text-foreground sm:max-w-[160px] lg:max-w-[180px]">
              {accountLabel}
            </span>
          </Link>
        </div>
      </div>
    </nav>
  );
}
