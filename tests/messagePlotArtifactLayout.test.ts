import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const chatPanel = readFileSync(
  new URL("../frontend/src/app/components/workspace/ChatPanel.tsx", import.meta.url),
  "utf8",
);
const artifactSurface = readFileSync(
  new URL("../frontend/src/app/components/workspace/ArtifactSurface.tsx", import.meta.url),
  "utf8",
);

test("plot artifacts in chat use the inner artifact card header for the dashboard action", () => {
  assert.doesNotMatch(
    chatPanel,
    /rounded-2xl border border-border\/40 bg-background\/30 p-3/,
  );
  assert.match(chatPanel, /<ArtifactSurface\s+key=\{artifact\.id\}[\s\S]*headerAction=\{/);
  assert.match(artifactSurface, /headerAction\?: ReactNode/);
  assert.match(
    artifactSurface,
    /\{headerAction \? <div className="shrink-0">\{headerAction\}<\/div> : null\}/,
  );
});
