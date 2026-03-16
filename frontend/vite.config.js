import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig(function (_a) {
    var mode = _a.mode;
    var env = loadEnv(mode, ".", "");
    var frontendPort = Number(env.FRONTEND_PORT || 8603);
    return {
        plugins: [react()],
        server: {
            host: "0.0.0.0",
            port: frontendPort
        },
        preview: {
            host: "0.0.0.0",
            port: frontendPort
        }
    };
});
