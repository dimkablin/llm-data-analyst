import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_ARTIFACT_ICON_KEY,
  getArtifactIconKey,
} from "../frontend/src/app/lib/artifact-icons.ts";

test("maps supported artifact types to contextual icon keys", () => {
  assert.equal(getArtifactIconKey("json"), "json");
  assert.equal(getArtifactIconKey("plot"), "chart");
  assert.equal(getArtifactIconKey("table"), "table");
  assert.equal(getArtifactIconKey("value"), "metric");
  assert.equal(getArtifactIconKey("note"), "note");
});

test("normalizes backend artifact aliases before selecting an icon", () => {
  assert.equal(getArtifactIconKey("DATAFRAME"), "table");
  assert.equal(getArtifactIconKey(" sql_result "), "table");
  assert.equal(getArtifactIconKey("scalar"), "metric");
  assert.equal(getArtifactIconKey("chart"), "chart");
  assert.equal(getArtifactIconKey("markdown"), "note");
});

test("uses a neutral artifact icon for unknown or empty artifact types", () => {
  assert.equal(DEFAULT_ARTIFACT_ICON_KEY, "artifact");
  assert.equal(getArtifactIconKey(""), DEFAULT_ARTIFACT_ICON_KEY);
  assert.equal(getArtifactIconKey(null), DEFAULT_ARTIFACT_ICON_KEY);
  assert.equal(getArtifactIconKey("custom_payload"), DEFAULT_ARTIFACT_ICON_KEY);
});
