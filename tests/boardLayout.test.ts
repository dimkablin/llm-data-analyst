import assert from "node:assert/strict";
import test from "node:test";

import type { ArtifactPayload } from "../frontend/src/app/lib/backend-types.ts";
import {
  BOARD_TURN_HEADER_HEIGHT_PX,
  computeBoardLayouts,
  estimateAutoHeight,
} from "../frontend/src/app/components/workspace/board-layout.ts";

function artifact(id: string, type = "note"): ArtifactPayload {
  return {
    id,
    type,
    role: "assistant",
    meta: {},
    timestamp: "2026-01-01T00:00:00Z",
    data: {
      format: type === "table" ? "split" : "markdown",
      data: type === "table" ? { columns: ["A"], data: [[1], [2]] } : { content: "one\ntwo" },
    },
  };
}

test("board layout honors turn headers and preferred columns", () => {
  const result = computeBoardLayouts(
    [artifact("a"), artifact("b", "table")],
    [{ turnKey: "turn-1", label: "Turn 1", firstArtifactId: "a" }],
    { a: 6, b: 6 },
    {},
    { b: 7 },
    {},
    false,
    estimateAutoHeight,
  );

  assert.equal(result.turnHeaderLayouts.length, 1);
  assert.equal(result.turnHeaderLayouts[0]?.topPx, 0);

  const first = result.layouts.get("a");
  const second = result.layouts.get("b");
  assert.ok(first);
  assert.ok(second);
  assert.equal(first.colStart, 1);
  assert.equal(second.colStart, 7);
  assert.ok(first.topPx >= BOARD_TURN_HEADER_HEIGHT_PX);
  assert.ok(result.boardHeight > second.topPx);
});

test("auto height estimates are bounded for compact note artifacts", () => {
  const height = estimateAutoHeight(artifact("note"));

  assert.ok(height >= 220);
  assert.ok(height <= 800);
});

test("full-width artifacts ignore saved narrow widths", () => {
  const fullWidth = artifact("planfact", "table");
  fullWidth.meta = { full_width: true };

  const result = computeBoardLayouts(
    [fullWidth],
    [],
    { planfact: 4 },
    {},
    {},
    {},
    false,
    estimateAutoHeight,
  );

  assert.equal(result.layouts.get("planfact")?.widthUnits, 12);
});

test("compact donut and wide waterfall share one dashboard row", () => {
  const donut = artifact("donut", "plot");
  const waterfall = artifact("waterfall", "plot");
  donut.meta = { board_width_units: 4 };
  waterfall.meta = { board_width_units: 8 };

  const result = computeBoardLayouts(
    [donut, waterfall],
    [],
    {},
    {},
    {},
    {},
    false,
    estimateAutoHeight,
  );

  assert.equal(result.layouts.get("donut")?.colStart, 1);
  assert.equal(result.layouts.get("donut")?.widthUnits, 4);
  assert.equal(result.layouts.get("waterfall")?.colStart, 5);
  assert.equal(result.layouts.get("waterfall")?.widthUnits, 8);
  assert.equal(result.layouts.get("donut")?.topPx, result.layouts.get("waterfall")?.topPx);
});
