import { defineConfig, loadEnv } from "vite";
import path from "path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const localEnv = loadEnv(mode, ".", "");
  const projectEnv = loadEnv(mode, path.resolve(__dirname, ".."), "");
  const env = {
    ...projectEnv,
    ...localEnv,
  };
  const backendPort = env.BACKEND_PORT || "8605";
  const backendUrl = env.VITE_API_BASE || `http://localhost:${backendPort}`;
  const phoenixHost = env.PHOENIX_HOST || "localhost";
  const phoenixUiPort = env.PHOENIX_UI_PORT || "8607";
  const phoenixUrl = env.VITE_PHOENIX_URL || `http://${phoenixHost}:${phoenixUiPort}`;
  const frontendPort = env.FRONTEND_PORT ? Number(env.FRONTEND_PORT) : undefined;
  const backendProxy = {
    target: backendUrl,
    changeOrigin: true,
  };
  const phoenixProxy = {
    target: phoenixUrl,
    changeOrigin: true,
  };
  const phoenixAssetProxy = {
    target: phoenixUrl,
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/phoenix\/assets\//, "/assets/"),
  };
  const phoenixApiProxy = {
    target: phoenixUrl,
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/phoenix\/api(\/|$)/, "/api$1"),
  };
  const phoenixV1Proxy = {
    target: phoenixUrl,
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/phoenix\/v1(\/|$)/, "/v1$1"),
  };
  const phoenixGraphqlProxy = {
    target: phoenixUrl,
    changeOrigin: true,
    rewrite: () => "/graphql",
  };
  const phoenixModernizrProxy = {
    target: phoenixUrl,
    changeOrigin: true,
    rewrite: () => "/modernizr.js",
  };
  const phoenixFaviconProxy = {
    target: phoenixUrl,
    changeOrigin: true,
    rewrite: () => "/favicon.ico",
  };

  return {
    plugins: [
      react(),
      tailwindcss(),
    ],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    server: {
      host: "0.0.0.0",
      ...(frontendPort ? { port: frontendPort } : {}),
      proxy: {
        "/auth": backendProxy,
        "/admin": backendProxy,
        "/db-connections": backendProxy,
        "/sessions": backendProxy,
        "/runtime": backendProxy,
        "/observability": backendProxy,
        "/health": backendProxy,
        "/phoenix/assets/": phoenixAssetProxy,
        "/phoenix/api": phoenixApiProxy,
        "/phoenix/api/": phoenixApiProxy,
        "/phoenix/v1/": phoenixV1Proxy,
        "/phoenix/graphql": phoenixGraphqlProxy,
        "/phoenix/modernizr.js": phoenixModernizrProxy,
        "/phoenix/favicon.ico": phoenixFaviconProxy,
        "/phoenix/": phoenixProxy,
        "/api": phoenixProxy,
        "/v1": phoenixProxy,
        "/graphql": phoenixProxy,
      },
    },
    preview: {
      host: "0.0.0.0",
      ...(frontendPort ? { port: frontendPort } : {}),
    },
    assetsInclude: ["**/*.svg", "**/*.csv"],
  };
});
