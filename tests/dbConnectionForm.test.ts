import assert from "node:assert/strict";
import test from "node:test";

import type { DBConnection } from "../frontend/src/app/lib/backend-types.ts";
import {
  buildPayload,
  defaultPortFor,
  toFormState,
} from "../frontend/src/app/components/workspace/db-connection-form.ts";

const existingConnection: DBConnection = {
  id: "conn-1",
  name: "Warehouse",
  db_type: "postgresql",
  host: "db.local",
  port: 5432,
  database: "analytics",
  username: "reader",
  options_json: { sslmode: "require", schema: "mart" },
  password_present: true,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

test("connection form derives defaults from db type and existing connection", () => {
  assert.equal(defaultPortFor("postgresql"), "5432");
  assert.equal(defaultPortFor("clickhouse"), "8123");

  const form = toFormState(existingConnection);

  assert.equal(form.name, "Warehouse");
  assert.equal(form.port, "5432");
  assert.equal(form.secretMode, "keep");
  assert.equal(form.password, "");
  assert.equal(form.sslmode, "require");
});

test("connection payload preserves schema and only sends password when requested", () => {
  const keepPayload = buildPayload(
    {
      ...toFormState(existingConnection),
      name: " Warehouse ",
      host: " db.local ",
    },
    true,
    existingConnection,
  );

  assert.equal(keepPayload.name, "Warehouse");
  assert.equal(keepPayload.host, "db.local");
  assert.deepEqual(keepPayload.options_json, { sslmode: "require", schema: "mart" });
  assert.equal(Object.hasOwn(keepPayload, "password"), false);
  assert.equal(Object.hasOwn(keepPayload, "clear_password"), false);

  const replacePayload = buildPayload(
    {
      ...toFormState(existingConnection),
      secretMode: "replace",
      password: "new-secret",
    },
    true,
    existingConnection,
  );
  assert.equal(replacePayload.password, "new-secret");

  const clearPayload = buildPayload(
    {
      ...toFormState(existingConnection),
      secretMode: "clear",
    },
    true,
    existingConnection,
  );
  assert.equal(clearPayload.clear_password, true);
});
