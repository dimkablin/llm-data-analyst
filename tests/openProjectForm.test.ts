import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_OPENPROJECT_FORM,
  buildOpenProjectPayload,
} from "../frontend/src/app/components/workspace/openproject-form.ts";

test("openproject form defaults target env-driven project sync", () => {
  const payload = buildOpenProjectPayload(DEFAULT_OPENPROJECT_FORM);

  assert.equal(payload.base_url, "http://localhost:8080");
  assert.equal(payload.api_key, null);
  assert.equal(payload.project, null);
  assert.equal(payload.all_projects, true);
  assert.equal(payload.days, 90);
});

test("openproject payload trims inputs and handles invalid days", () => {
  const payload = buildOpenProjectPayload({
    baseUrl: " https://openproject.local ",
    apiKey: " token ",
    project: " alpha ",
    days: "not-a-number",
  });

  assert.equal(payload.base_url, "https://openproject.local");
  assert.equal(payload.api_key, "token");
  assert.equal(payload.project, "alpha");
  assert.equal(payload.all_projects, false);
  assert.equal(payload.days, null);
});
