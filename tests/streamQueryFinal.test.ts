import assert from "node:assert/strict";
import test from "node:test";

test("streamQuery stops after final SSE event", async () => {
  const globals = globalThis as typeof globalThis & {
    window?: { localStorage: { getItem: () => string | null } };
  };
  const originalFetch = globalThis.fetch;
  const originalWindow = globals.window;
  const payload = {
    session_id: "session-1",
    text: "done",
    artifacts: [],
    metrics: {
      duration_ms: 1,
      artifact_count: 0,
      table_count: 0,
      plot_count: 0,
      value_count: 0,
      model: "test",
    },
  };
  let reads = 0;

  globals.window = { localStorage: { getItem: () => null } };
  globalThis.fetch = async () =>
    new Response(
      new ReadableStream<Uint8Array>({
        pull(controller) {
          reads += 1;
          if (reads === 1) {
            controller.enqueue(
              new TextEncoder().encode(`event: final\ndata: ${JSON.stringify(payload)}\n\n`),
            );
            return;
          }
          controller.error(new TypeError("network error"));
        },
      }),
      { status: 200 },
    );

  try {
    const { streamQuery } = await import("../frontend/src/app/lib/backend-api.ts");
    const errors: string[] = [];
    let finalText = "";

    await streamQuery("session-1", "query", false, true, {
      onToken: () => {},
      onFinal: (finalPayload) => {
        finalText = finalPayload.text;
      },
      onReasoning: () => {},
      onError: (error) => errors.push(error),
    });

    assert.equal(finalText, "done");
    assert.deepEqual(errors, []);
  } finally {
    globalThis.fetch = originalFetch;
    globals.window = originalWindow;
  }
});

test("streamQuery sends one-shot skill and tool options", async () => {
  const globals = globalThis as typeof globalThis & {
    window?: { localStorage: { getItem: () => string | null } };
  };
  const originalFetch = globalThis.fetch;
  const originalWindow = globals.window;
  let requestBody: Record<string, unknown> = {};

  globals.window = { localStorage: { getItem: () => null } };
  globalThis.fetch = async (_input, init) => {
    requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
    const payload = {
      session_id: "session-1",
      text: "done",
      artifacts: [],
      metrics: {
        duration_ms: 1,
        artifact_count: 0,
        table_count: 0,
        plot_count: 0,
        value_count: 0,
        model: "test",
      },
    };
    return new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(
            new TextEncoder().encode(`event: final\ndata: ${JSON.stringify(payload)}\n\n`),
          );
          controller.close();
        },
      }),
      { status: 200 },
    );
  };

  try {
    const { streamQuery } = await import("../frontend/src/app/lib/backend-api.ts");
    await streamQuery(
      "session-1", "query", false, true,
      { onToken: () => {}, onFinal: () => {}, onReasoning: () => {}, onError: () => {} },
      undefined, undefined, ["skill-1"], "sql_tool",
    );
    assert.deepEqual(requestBody.selected_skill_ids, ["skill-1"]);
    assert.equal(requestBody.requested_tool_key, "sql_tool");
  } finally {
    globalThis.fetch = originalFetch;
    globals.window = originalWindow;
  }
});
