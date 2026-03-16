import { useCallback, useEffect, useMemo, useState } from "react";

import {
  clearStoredToken,
  getCurrentUser,
  getStoredToken,
  loginUser,
  logoutUser,
  registerUser,
  setStoredToken
} from "./api";
import { stripBasePath, withBasePath } from "./basePath";
import { AuthPage } from "./pages/AuthPage";
import { LandingPage } from "./pages/LandingPage";
import { PhoenixPage } from "./pages/PhoenixPage";
import { TechnicalPage } from "./pages/TechnicalPage";
import { WorkspacePage } from "./pages/WorkspacePage";
import type { AuthUser } from "./types";

type AppRoute = "/" | "/user" | "/technical" | "/app" | "/phoenix";

function normalizeRoute(pathname: string): AppRoute {
  if (pathname === "/user" || pathname === "/auth") {
    return "/user";
  }
  if (pathname === "/technical") {
    return "/technical";
  }
  if (pathname === "/app") {
    return "/app";
  }
  if (pathname === "/phoenix") {
    return "/phoenix";
  }
  return "/";
}

export default function App(): JSX.Element {
  const [route, setRoute] = useState<AppRoute>(() => normalizeRoute(stripBasePath(window.location.pathname)));
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isAuthLoading, setIsAuthLoading] = useState(true);

  const navigate = useCallback((nextRoute: AppRoute, replace = false): void => {
    const target = withBasePath(nextRoute);
    if (replace) {
      window.history.replaceState({}, "", target);
    } else {
      window.history.pushState({}, "", target);
    }
    setRoute(nextRoute);
  }, []);

  useEffect(() => {
    function handlePopState(): void {
      setRoute(normalizeRoute(stripBasePath(window.location.pathname)));
    }
    window.addEventListener("popstate", handlePopState);
    return () => {
      window.removeEventListener("popstate", handlePopState);
    };
  }, []);

  useEffect(() => {
    let mounted = true;
    async function bootstrapAuth(): Promise<void> {
      const token = getStoredToken();
      if (!token) {
        if (mounted) {
          setUser(null);
          setIsAuthLoading(false);
        }
        return;
      }
      try {
        const me = await getCurrentUser();
        if (mounted) {
          setUser(me);
        }
      } catch {
        clearStoredToken();
        if (mounted) {
          setUser(null);
        }
      } finally {
        if (mounted) {
          setIsAuthLoading(false);
        }
      }
    }
    void bootstrapAuth();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (!isAuthLoading && (route === "/app" || route === "/phoenix") && !user) {
      navigate("/user", true);
    }
  }, [isAuthLoading, navigate, route, user]);

  useEffect(() => {
    if (!isAuthLoading && route === "/phoenix" && user && !user.is_admin) {
      navigate("/app", true);
    }
  }, [isAuthLoading, navigate, route, user]);

  const authActions = useMemo(
    () => ({
      login: async (username: string, password: string): Promise<void> => {
        const result = await loginUser(username, password);
        setStoredToken(result.access_token);
        setUser(result.user);
      },
      register: async (username: string, password: string): Promise<void> => {
        const result = await registerUser(username, password);
        setStoredToken(result.access_token);
        setUser(result.user);
      },
      logout: async (): Promise<void> => {
        try {
          await logoutUser();
        } catch {
          // Ignore network errors on logout and clear local auth state anyway.
        }
        clearStoredToken();
        setUser(null);
      }
    }),
    []
  );

  if (isAuthLoading) {
    return (
      <div className="app-loader-wrap">
        <div className="app-loader-card">
          <h1>Генеративная аналитика</h1>
          <p>Проверка сессии пользователя...</p>
        </div>
      </div>
    );
  }

  if (route === "/user") {
    return (
      <AuthPage
        currentUser={user}
        onNavigate={navigate}
        onLogin={authActions.login}
        onRegister={authActions.register}
        onLogout={authActions.logout}
      />
    );
  }

  if (route === "/technical") {
    return (
      <TechnicalPage
        currentUser={user}
        onNavigate={navigate}
      />
    );
  }

  if (route === "/app" && user) {
    return <WorkspacePage user={user} onLogout={authActions.logout} onNavigate={navigate} />;
  }

  if (route === "/phoenix" && user?.is_admin) {
    return <PhoenixPage currentUser={user} onNavigate={navigate} />;
  }

  return (
    <LandingPage
      currentUser={user}
      onNavigate={navigate}
    />
  );
}
