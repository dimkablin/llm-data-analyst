export type PhoenixUrlEnv = {
  BASE_URL?: string;
  VITE_PHOENIX_BASE_PATH?: string;
  VITE_PHOENIX_PUBLIC_URL?: string;
};

const ABSOLUTE_URL_PATTERN = /^[a-z][a-z\d+.-]*:\/\//i;

function withTrailingSlash(value: string): string {
  return value.endsWith("/") ? value : `${value}/`;
}

function normalizePathOrUrl(value: string | undefined): string {
  const clean = value?.trim() || "/";
  if (ABSOLUTE_URL_PATTERN.test(clean)) {
    return withTrailingSlash(clean);
  }

  const rootedPath = clean.startsWith("/") ? clean : `/${clean}`;
  return withTrailingSlash(rootedPath);
}

function getExplicitPhoenixUrl(env: PhoenixUrlEnv): string | null {
  const explicitUrl = env.VITE_PHOENIX_PUBLIC_URL?.trim();
  if (explicitUrl) {
    return normalizePathOrUrl(explicitUrl);
  }

  const explicitBasePath = env.VITE_PHOENIX_BASE_PATH?.trim();
  if (explicitBasePath) {
    return normalizePathOrUrl(explicitBasePath);
  }

  return null;
}

export function resolvePhoenixUiBaseUrl(env: PhoenixUrlEnv): string {
  const explicitUrl = getExplicitPhoenixUrl(env);
  if (explicitUrl) {
    return explicitUrl;
  }

  return `${normalizePathOrUrl(env.BASE_URL)}phoenix/`;
}

export function buildPhoenixProjectTraceUrl(
  env: PhoenixUrlEnv,
  projectId: string,
  traceId: string,
): string {
  const phoenixBaseUrl = resolvePhoenixUiBaseUrl(env);
  return `${phoenixBaseUrl}projects/${encodeURIComponent(projectId)}/traces/${encodeURIComponent(traceId)}`;
}
