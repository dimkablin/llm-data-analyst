import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import {
  clearStoredToken,
  getCurrentUser,
  getUserSettings,
  hasStoredToken,
  loginUser,
  logoutUser,
  registerUser,
  setStoredToken,
  updateUserSettings,
} from "../lib/backend-api";
import type { AuthUser, UserSettings } from "../lib/backend-types";

const DEFAULT_SETTINGS: UserSettings = {
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
  agent_inner_recursion_limit: 6,
};

type AppSessionContextValue = {
  user: AuthUser | null;
  settings: UserSettings;
  isAuthLoading: boolean;
  isAuthenticated: boolean;
  refreshAuth: () => Promise<void>;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  saveSettings: (payload: Partial<UserSettings>) => Promise<UserSettings>;
  setLocalSettings: (next: UserSettings) => void;
};

const AppSessionContext = createContext<AppSessionContextValue | null>(null);

export function AppSessionProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [settings, setSettings] = useState<UserSettings>(DEFAULT_SETTINGS);
  const [isAuthLoading, setIsAuthLoading] = useState(true);

  const refreshAuth = useCallback(async () => {
    if (!hasStoredToken()) {
      setUser(null);
      setSettings(DEFAULT_SETTINGS);
      setIsAuthLoading(false);
      return;
    }

    setIsAuthLoading(true);
    try {
      const me = await getCurrentUser();
      setUser(me);
      try {
        const nextSettings = await getUserSettings();
        setSettings(nextSettings);
      } catch {
        setSettings(DEFAULT_SETTINGS);
      }
    } catch {
      clearStoredToken();
      setUser(null);
      setSettings(DEFAULT_SETTINGS);
    } finally {
      setIsAuthLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshAuth();
  }, [refreshAuth]);

  const login = useCallback(async (username: string, password: string) => {
    const result = await loginUser(username, password);
    setStoredToken(result.access_token);
    setUser(result.user);
    try {
      setSettings(await getUserSettings());
    } catch {
      setSettings(DEFAULT_SETTINGS);
    }
  }, []);

  const register = useCallback(async (username: string, password: string) => {
    const result = await registerUser(username, password);
    setStoredToken(result.access_token);
    setUser(result.user);
    try {
      setSettings(await getUserSettings());
    } catch {
      setSettings(DEFAULT_SETTINGS);
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      await logoutUser();
    } catch {
      // Local logout should still proceed.
    }
    clearStoredToken();
    setUser(null);
    setSettings(DEFAULT_SETTINGS);
  }, []);

  const saveSettings = useCallback(async (payload: Partial<UserSettings>) => {
    const updated = await updateUserSettings(payload);
    setSettings(updated);
    return updated;
  }, []);

  const value = useMemo<AppSessionContextValue>(
    () => ({
      user,
      settings,
      isAuthLoading,
      isAuthenticated: Boolean(user),
      refreshAuth,
      login,
      register,
      logout,
      saveSettings,
      setLocalSettings: setSettings,
    }),
    [isAuthLoading, login, logout, refreshAuth, register, saveSettings, settings, user],
  );

  return <AppSessionContext.Provider value={value}>{children}</AppSessionContext.Provider>;
}

export function useAppSession(): AppSessionContextValue {
  const value = useContext(AppSessionContext);
  if (!value) {
    throw new Error("useAppSession must be used within AppSessionProvider");
  }
  return value;
}
