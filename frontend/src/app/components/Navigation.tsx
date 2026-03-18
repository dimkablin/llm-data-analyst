import { useMemo } from "react";
import { Link, useLocation } from "react-router";
import { Bell, Search, User } from "lucide-react";
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
    { path: "/phoenix", label: "Phoenix" },
  ];

  const accountLabel = useMemo(() => {
    if (!user) {
      return "Войти";
    }
    return user.is_admin ? `${user.username} • admin` : user.username;
  }, [user]);

  return (
    <nav className="sticky top-0 z-50 border-b border-border/50 bg-background/60 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-[1600px] items-center justify-between px-6">
        <div className="flex min-w-0 items-center gap-6 lg:gap-10">
          <Link to="/" className="group flex shrink-0 items-center gap-2 transition-all">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-xl font-bold text-primary-foreground shadow-lg shadow-primary/20 transition-transform group-hover:scale-105">
              A
            </div>
            <span className="whitespace-nowrap text-[17px] font-semibold tracking-tight text-foreground transition-colors group-hover:text-primary">
              Генеративная аналитика
            </span>
          </Link>

          <div className="hidden min-w-0 flex-1 items-center overflow-x-auto rounded-xl border border-border/40 bg-secondary/50 p-1 backdrop-blur-md [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden md:flex">
            {navItems.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`shrink-0 whitespace-nowrap rounded-lg px-4 py-1.5 text-[14px] font-medium transition-all duration-200 lg:px-5 ${
                    isActive
                      ? "bg-card text-foreground shadow-sm ring-1 ring-border/20"
                      : "text-muted-foreground hover:bg-white/5 hover:text-foreground"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-3">
          <Link
            to="/sessions?focus=search"
            aria-label="Поиск по чатам"
            className="rounded-full border border-transparent p-2.5 text-muted-foreground transition-all hover:border-border/50 hover:bg-secondary hover:text-foreground"
          >
            <Search className="h-5 w-5" />
          </Link>
          <ThemeToggle />
          <button className="relative rounded-full border border-transparent p-2.5 text-muted-foreground transition-all hover:border-border/50 hover:bg-secondary hover:text-foreground">
            <Bell className="h-5 w-5" />
            <span className="absolute right-2.5 top-2.5 h-2 w-2 rounded-full bg-primary ring-2 ring-background"></span>
          </button>
          <div className="mx-2 h-6 w-[1px] bg-border/50"></div>
          <Link
            to={signedIn ? "/account" : "/auth"}
            className="flex items-center gap-2.5 rounded-xl border border-border/50 bg-secondary px-4 py-2 transition-all hover:border-border hover:bg-muted active:scale-95"
          >
            <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/20">
              <User className="h-3.5 w-3.5 text-primary" />
            </div>
            <span className="max-w-[180px] truncate text-[14px] font-medium text-foreground">{accountLabel}</span>
          </Link>
        </div>
      </div>
    </nav>
  );
}
