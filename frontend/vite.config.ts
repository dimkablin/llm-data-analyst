import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

function normalizeBasePath(value?: string): string {
  if (!value || value === "/") {
    return "/";
  }
  const trimmed = value.trim();
  const withLeadingSlash = trimmed.charAt(0) === "/" ? trimmed : `/${trimmed}`;
  return withLeadingSlash.slice(-1) === "/" ? withLeadingSlash : `${withLeadingSlash}/`;
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const frontendPort = Number(env.FRONTEND_PORT || 8603);
  const backendUrl = `http://localhost:${env.BACKEND_PORT || 8000}`;
  const processEnv =
    typeof globalThis !== "undefined" && "process" in globalThis
      ? (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env
      : undefined;
  const appBasePath = normalizeBasePath(processEnv?.APP_BASE_PATH || env.APP_BASE_PATH);

  const backendProxy = {
    target: backendUrl,
    changeOrigin: true,
  };

  return {
    base: appBasePath,
    plugins: [react()],
    server: {
      host: "0.0.0.0",
      port: frontendPort,
      proxy: {
        "/auth": backendProxy,
        "/admin": backendProxy,
        "/sessions": backendProxy,
        "/runtime": backendProxy,
        "/health": backendProxy,
      },
    },
    preview: {
      host: "0.0.0.0",
      port: frontendPort
    }
  };
});
