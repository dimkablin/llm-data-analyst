import assert from "node:assert/strict";
import test from "node:test";

import { plotlySequence } from "../frontend/src/app/lib/plotly-data.ts";

test("decodes Plotly typed-array payloads", () => {
  assert.deepEqual(
    plotlySequence({
      dtype: "f8",
      bdata: "AAAAAAAAJEAAAAAAAAA0QAAAAAAAAD5A",
    }),
    [10, 20, 30],
  );
});
