export const KNOWN_ARTIFACT_TYPES = [
  "plot",
  "table",
  "value",
  "json",
  "note",
] as const;

export type KnownArtifactType = (typeof KNOWN_ARTIFACT_TYPES)[number];

export const ARTIFACT_ICON_KEYS = [
  "artifact",
  "chart",
  "table",
  "metric",
  "json",
  "note",
] as const;

export type ArtifactIconKey = (typeof ARTIFACT_ICON_KEYS)[number];

export const DEFAULT_ARTIFACT_ICON_KEY: ArtifactIconKey = "artifact";

const ARTIFACT_TYPE_ICON_KEY: Record<KnownArtifactType, ArtifactIconKey> = {
  plot: "chart",
  table: "table",
  value: "metric",
  json: "json",
  note: "note",
};

const ARTIFACT_TYPE_ALIASES: Record<string, KnownArtifactType> = {
  chart: "plot",
  visualization: "plot",
  dataframe: "table",
  sql_result: "table",
  scalar: "value",
  number: "value",
  metric: "value",
  markdown: "note",
  text: "note",
};

const KNOWN_ARTIFACT_TYPE_SET = new Set<string>(KNOWN_ARTIFACT_TYPES);

export function normalizeArtifactType(type: string | null | undefined): KnownArtifactType | null {
  const normalized = String(type ?? "").trim().toLowerCase();
  if (!normalized) {
    return null;
  }
  if (KNOWN_ARTIFACT_TYPE_SET.has(normalized)) {
    return normalized as KnownArtifactType;
  }
  return ARTIFACT_TYPE_ALIASES[normalized] ?? null;
}

export function getArtifactIconKey(type: string | null | undefined): ArtifactIconKey {
  const normalized = normalizeArtifactType(type);
  if (!normalized) {
    return DEFAULT_ARTIFACT_ICON_KEY;
  }
  return ARTIFACT_TYPE_ICON_KEY[normalized];
}
