import assert from "node:assert/strict";
import test from "node:test";

import {
  isRagProcessing,
  normalizeRagStatus,
  ragStatusLabel,
} from "../frontend/src/app/components/workspace/rag-status.ts";

test("rag status helpers normalize backend enum strings", () => {
  assert.equal(normalizeRagStatus("DocStatus.PROCESSED"), "processed");
  assert.equal(normalizeRagStatus("DocStatus.pending"), "pending");
  assert.equal(normalizeRagStatus(null), "unknown");
});

test("rag status helpers group canonical UI states", () => {
  assert.equal(ragStatusLabel("DocStatus.PROCESSED"), "processed");
  assert.equal(ragStatusLabel("pending"), "processing");
  assert.equal(ragStatusLabel("processing"), "processing");
  assert.equal(ragStatusLabel("failure"), "failed");
  assert.equal(ragStatusLabel("failed"), "failed");

  assert.equal(isRagProcessing("pending"), true);
  assert.equal(isRagProcessing("processing"), true);
  assert.equal(isRagProcessing("processed"), false);
});
