import { MoonStar, SunMedium } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const isDark = resolvedTheme === "dark";

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <button
      type="button"
      onClick={() => mounted && setTheme(isDark ? "light" : "dark")}
      className="p-2.5 rounded-full hover:bg-secondary text-muted-foreground hover:text-foreground transition-all border border-transparent hover:border-border/50"
      aria-label={isDark ? "Переключить на светлую тему" : "Переключить на темную тему"}
      title={isDark ? "Светлая тема" : "Темная тема"}
    >
      {mounted && isDark ? (
        <SunMedium className="w-5 h-5 text-amber-400" />
      ) : (
        <MoonStar className="w-5 h-5 text-primary" />
      )}
    </button>
  );
}
