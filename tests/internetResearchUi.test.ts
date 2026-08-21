import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL("../frontend/src/app/components/workspace/ChatPanel.tsx", import.meta.url),
  "utf8",
);

test("deep research action selects the internet research skill", () => {
  const handler = source.slice(
    source.indexOf("function handleMenuAction"),
    source.indexOf("async function handleSend"),
  );
  assert.match(handler, /type: "skill",\s+id: "internet_research"/);
});
