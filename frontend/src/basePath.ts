const FALLBACK_BASE_PATH = "/";

function normalizeBasePath(value?: string): string {
  if (!value || value === "/") {
    return FALLBACK_BASE_PATH;
  }

  const trimmed = value.trim();
  const withLeadingSlash = trimmed.charAt(0) === "/" ? trimmed : `/${trimmed}`;
  return withLeadingSlash.slice(-1) === "/" ? withLeadingSlash : `${withLeadingSlash}/`;
}

export const appBasePath = normalizeBasePath(import.meta.env.BASE_URL);

export function withBasePath(path: string): string {
  if (!path || path === "/") {
    return appBasePath;
  }

  const normalizedPath = path.charAt(0) === "/" ? path.slice(1) : path;
  return `${appBasePath}${normalizedPath}`;
}

export function stripBasePath(pathname: string): string {
  if (appBasePath === "/") {
    return pathname || "/";
  }

  if (pathname === appBasePath.slice(0, -1)) {
    return "/";
  }

  if (pathname.indexOf(appBasePath) === 0) {
    const stripped = pathname.slice(appBasePath.length - 1);
    return stripped || "/";
  }

  return pathname || "/";
}
