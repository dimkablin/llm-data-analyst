import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
function normalizeBasePath(value) {
    if (!value || value === "/") {
        return "/";
    }
    var trimmed = value.trim();
    var withLeadingSlash = trimmed.charAt(0) === "/" ? trimmed : "/".concat(trimmed);
    return withLeadingSlash.slice(-1) === "/" ? withLeadingSlash : "".concat(withLeadingSlash, "/");
}
export default defineConfig(function (_a) {
    var _b;
    var mode = _a.mode;
    var env = loadEnv(mode, ".", "");
    var frontendPort = Number(env.FRONTEND_PORT || 8603);
    var backendUrl = "http://localhost:".concat(env.BACKEND_PORT || 8000);
    var processEnv = typeof globalThis !== "undefined" && "process" in globalThis
        ? (_b = globalThis.process) === null || _b === void 0 ? void 0 : _b.env
        : undefined;
    var appBasePath = normalizeBasePath((processEnv === null || processEnv === void 0 ? void 0 : processEnv.APP_BASE_PATH) || env.APP_BASE_PATH);
    var backendProxy = {
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
