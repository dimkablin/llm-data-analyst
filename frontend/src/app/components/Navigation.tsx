import { useMemo } from "react";
import { Link, useLocation } from "react-router";
import { Bell, Search } from "lucide-react";
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
    { path: "/sessions", label: "Мои дашборды" },
    { path: "/tracing", label: "Phoenix" },
  ];

  const accountLabel = useMemo(() => {
    if (!user) {
      return { title: "Войти", subtitle: "" };
    }
    return {
      title: "Глава городского округа",
      subtitle: "Иванов И. И.",
    };
  }, [user]);

  return (
    <nav className="sticky top-0 z-50 border-b border-border/50 bg-background/60 backdrop-blur-xl">
      <div className="mx-auto grid h-14 max-w-[1600px] grid-cols-[1fr_auto_1fr] items-center px-3 sm:h-16 sm:px-4 lg:px-6">
        <div className="flex min-w-0 items-center gap-3 sm:gap-4">
          <Link to="/" className="group flex shrink-0 items-center gap-2 transition-all">
            <div className="relative flex h-8 w-8 shrink-0 items-center justify-center overflow-hidden rounded-xl bg-primary shadow-lg shadow-primary/25 transition-all group-hover:shadow-primary/40">
              <svg viewBox="83 130 320 230" fill="currentColor" className="h-5 w-5 text-primary-foreground">
                <rect x="175.67" y="135" width="44.662" height="220.553"/>
                <rect x="353.215" y="135" width="44.662" height="220.553"/>
                <rect x="88" y="223.221" width="220.553" height="44.662"/>
              </svg>
            </div>
            <span className="hidden whitespace-nowrap text-[15px] font-semibold tracking-tight text-foreground transition-colors group-hover:text-primary sm:block lg:text-[16px]">
              ИИ Монитор
            </span>
          </Link>
        </div>

        <div className="hidden items-center overflow-x-auto rounded-xl border border-border/40 bg-secondary/50 p-1 backdrop-blur-md [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden lg:flex">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className="relative shrink-0 whitespace-nowrap rounded-lg px-3 py-1.5 text-[13px] font-medium transition-colors duration-200 xl:px-4 xl:text-[14px] 2xl:px-5"
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

        <div className="flex shrink-0 items-center justify-end gap-1.5 sm:gap-2 lg:gap-3">
          <Link
            to="/sessions?focus=search"
            aria-label="Поиск по чатам"
            className="rounded-full border border-transparent p-2 text-muted-foreground transition-all hover:border-border/50 hover:bg-secondary hover:text-foreground sm:p-2.5"
          >
            <Search className="h-5 w-5" />
          </Link>
          <ThemeToggle />
          <button className="relative rounded-full border border-transparent p-2 text-muted-foreground transition-all hover:border-border/50 hover:bg-secondary hover:text-foreground sm:p-2.5">
            <Bell className="h-5 w-5" />
            <span className="absolute right-2.5 top-2.5 h-2 w-2 rounded-full bg-primary ring-2 ring-background"></span>
          </button>
          <div className="mx-1 hidden h-6 w-[1px] bg-border/50 sm:block"></div>
          <Link
            to={signedIn ? "/account" : "/auth"}
            className="flex items-center gap-3 rounded-xl border border-border/50 bg-secondary px-3 py-2 transition-all hover:border-border hover:bg-muted active:scale-95 sm:px-4"
          >
            <div className="flex h-8 w-8 items-center justify-center rounded-full border border-border/40 bg-white text-[16px] shadow-sm">
              <span aria-hidden="true">🇷🇺</span>
            </div>
            <div className="min-w-0">
              <div className="max-w-[180px] truncate text-[12px] font-semibold uppercase tracking-[0.08em] text-muted-foreground sm:max-w-[220px]">
                {accountLabel.title}
              </div>
              <div className="max-w-[140px] truncate text-[13px] font-medium text-foreground sm:max-w-[180px] sm:text-[14px] lg:max-w-[220px]">
                {accountLabel.subtitle}
              </div>
            </div>
          </Link>
        </div>
      </div>
    </nav>
  );
}
