import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const frontendPort = Number(env.FRONTEND_PORT || 8603);
  const backendUrl = `http://localhost:${env.BACKEND_PORT || 8000}`;

  const backendProxy = {
    target: backendUrl,
    changeOrigin: true,
  };

  return {
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
