import { ThemeProvider as NextThemesProvider } from "next-themes";
import type { PropsWithChildren } from "react";
import { useEffect } from "react";
import { applyAccent, getStoredAccent } from "../lib/accent";

export function ThemeProvider({ children }: PropsWithChildren) {
  useEffect(() => {
    applyAccent(getStoredAccent());
  }, []);

  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
    >
      {children}
    </NextThemesProvider>
  );
}
