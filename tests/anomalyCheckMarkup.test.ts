import assert from "node:assert/strict";
import test from "node:test";

import { splitCheckedNumbers } from "../frontend/src/app/lib/anomaly-check.ts";
import type { AnomalyCheck } from "../frontend/src/app/lib/backend-types.ts";

test("splits checked numbers without mistaking a shorter overlapping value", () => {
  const items: AnomalyCheck["items"] = [
    { id: "short", text: "66,18%", normalized_value: 66.18, status: "matched", sources: [] },
    { id: "signed", text: "+66,18%", normalized_value: 66.18, status: "matched", sources: [] },
  ];

  const parts = splitCheckedNumbers("Отклонение: +66,18%.", items);

  assert.equal(parts[1].text, "+66,18%");
  assert.equal(parts[1].item?.id, "signed");
});

test("does not highlight a short number inside a year", () => {
  const items: AnomalyCheck["items"] = [
    { id: "two", text: "2", normalized_value: 2, status: "matched", sources: [] },
  ];

  assert.deepEqual(splitCheckedNumbers("Период: март 2026.", items), [
    { text: "Период: март 2026." },
  ]);
});
