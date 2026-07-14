import assert from "node:assert/strict";
import test from "node:test";

import {
  buildPhoenixProjectTraceUrl,
  resolvePhoenixUiBaseUrl,
} from "../frontend/src/app/lib/phoenix-url.ts";

test("uses the proxied Phoenix route by default", () => {
  assert.equal(resolvePhoenixUiBaseUrl({ BASE_URL: "/" }), "/phoenix/");
});

test("keeps Phoenix under the configured app base path", () => {
  assert.equal(resolvePhoenixUiBaseUrl({ BASE_URL: "/analyst/" }), "/analyst/phoenix/");
  assert.equal(resolvePhoenixUiBaseUrl({ BASE_URL: "/analyst" }), "/analyst/phoenix/");
});

test("allows an explicit frontend Phoenix URL override", () => {
  assert.equal(
    resolvePhoenixUiBaseUrl({
      BASE_URL: "/",
      VITE_PHOENIX_PUBLIC_URL: "http://localhost:9607",
    }),
    "http://localhost:9607/",
  );
});

test("normalizes an explicit frontend Phoenix base path override", () => {
  assert.equal(
    resolvePhoenixUiBaseUrl({
      BASE_URL: "/",
      VITE_PHOENIX_BASE_PATH: "local-phoenix",
    }),
    "/local-phoenix/",
  );
});

test("builds encoded Phoenix trace links from the same base URL contract", () => {
  assert.equal(
    buildPhoenixProjectTraceUrl(
      { BASE_URL: "/app/" },
      "project with spaces",
      "trace/with/slash",
    ),
    "/app/phoenix/projects/project%20with%20spaces/traces/trace%2Fwith%2Fslash",
  );
});
