export function normalizeRagStatus(status: string | null | undefined): string {
  return String(status || "unknown").replace(/^DocStatus\./, "").toLowerCase();
}

export function ragStatusLabel(status: string | null | undefined): string {
  const normalized = normalizeRagStatus(status);
  if (normalized === "processed") {
    return "processed";
  }
  if (normalized === "processing" || normalized === "pending") {
    return "processing";
  }
  if (normalized === "failed" || normalized === "failure") {
    return "failed";
  }
  return normalized;
}

export function isRagProcessing(status: string | null | undefined): boolean {
  const normalized = normalizeRagStatus(status);
  return normalized === "processing" || normalized === "pending";
}
