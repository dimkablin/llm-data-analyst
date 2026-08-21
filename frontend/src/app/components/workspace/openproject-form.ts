import type { OpenProjectSyncRequest } from "../../lib/backend-types";

export type OpenProjectFormState = {
  baseUrl: string;
  apiKey: string;
  project: string;
  days: string;
};

export const DEFAULT_OPENPROJECT_FORM: OpenProjectFormState = {
  baseUrl: "http://localhost:8080",
  apiKey: "",
  project: "",
  days: "90",
};

export function buildOpenProjectPayload(
  form: OpenProjectFormState,
): OpenProjectSyncRequest {
  const project = form.project.trim();
  const days = Number.parseInt(form.days.trim(), 10);
  return {
    base_url: form.baseUrl.trim() || null,
    api_key: form.apiKey.trim() || null,
    project: project || null,
    all_projects: !project,
    days: Number.isFinite(days) ? days : null,
  };
}
