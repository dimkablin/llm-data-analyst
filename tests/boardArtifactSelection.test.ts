import assert from "node:assert/strict";
import test from "node:test";

import type { ArtifactPayload, ChatMessage } from "../frontend/src/app/lib/backend-types.ts";
import {
  applyBoardTurnTitleOverrides,
  selectDefaultHighlightedBoardArtifactIds,
  selectVisibleBoardArtifactIds,
} from "../frontend/src/app/lib/board-artifacts.ts";

function artifact(id: string, type: string, timestamp: string): ArtifactPayload {
  return {
    id,
    type,
    text: id,
    role: "assistant",
    meta: {},
    timestamp,
    data: {
      format: type === "plot" ? "plotly-json" : "json",
      data: {},
    },
  };
}

function noteArtifact(sessionId: string, messageId: string, timestamp: string): ArtifactPayload {
  return {
    ...artifact(`msg_note_${sessionId}_${messageId}`, "note", timestamp),
    data: {
      format: "markdown",
      data: { content: "answer" },
    },
  };
}

const sessionId = "session-1";

const messages: ChatMessage[] = [
  {
    id: "user-1",
    role: "user",
    content: "first question",
    timestamp: "2026-06-16T10:00:00.000Z",
  },
  {
    id: "assistant-1",
    role: "assistant",
    content: "first answer",
    timestamp: "2026-06-16T10:01:00.000Z",
    artifacts: [
      artifact("plot-old", "plot", "2026-06-16T10:01:01.000Z"),
      artifact("table-old", "table", "2026-06-16T10:01:02.000Z"),
    ],
  },
  {
    id: "user-2",
    role: "user",
    content: "latest question",
    timestamp: "2026-06-16T10:05:00.000Z",
  },
  {
    id: "assistant-2",
    role: "assistant",
    content: "latest answer",
    timestamp: "2026-06-16T10:06:00.000Z",
    artifacts: [
      artifact("plot-latest", "plot", "2026-06-16T10:06:01.000Z"),
      artifact("table-latest", "table", "2026-06-16T10:06:02.000Z"),
    ],
  },
];

const artifacts: ArtifactPayload[] = [
  artifact("plot-old", "plot", "2026-06-16T10:01:01.000Z"),
  artifact("table-old", "table", "2026-06-16T10:01:02.000Z"),
  noteArtifact(sessionId, "assistant-1", "2026-06-16T10:01:03.000Z"),
  artifact("plot-latest", "plot", "2026-06-16T10:06:01.000Z"),
  artifact("table-latest", "table", "2026-06-16T10:06:02.000Z"),
  noteArtifact(sessionId, "assistant-2", "2026-06-16T10:06:03.000Z"),
];

test("shows only plot artifacts by default until the user explicitly pins an artifact", () => {
  const selected = selectVisibleBoardArtifactIds({
    artifacts,
    messages,
    sessionId,
    autoPinnedArtifactIds: ["plot-old"],
    userPinnedArtifactIds: [],
    hiddenArtifactIds: [],
  });

  assert.deepEqual(selected, [
    "plot-old",
    "plot-latest",
  ]);
});

test("falls back to json artifacts by default when there are no plots", () => {
  const jsonOnlyArtifacts = [
    artifact("table-result", "table", "2026-06-16T10:01:01.000Z"),
    artifact("json-result", "json", "2026-06-16T10:01:02.000Z"),
    noteArtifact(sessionId, "assistant-1", "2026-06-16T10:01:03.000Z"),
  ];

  const selected = selectVisibleBoardArtifactIds({
    artifacts: jsonOnlyArtifacts,
    messages,
    sessionId,
    autoPinnedArtifactIds: [],
    userPinnedArtifactIds: [],
    hiddenArtifactIds: [],
  });

  assert.deepEqual(selected, ["json-result"]);
});

test("shows no default board artifacts when there are no plots or json artifacts", () => {
  const nonHighlightedArtifacts = [
    artifact("table-result", "table", "2026-06-16T10:01:01.000Z"),
    artifact("value-result", "value", "2026-06-16T10:01:02.000Z"),
    noteArtifact(sessionId, "assistant-1", "2026-06-16T10:01:03.000Z"),
  ];

  const selected = selectVisibleBoardArtifactIds({
    artifacts: nonHighlightedArtifacts,
    messages,
    sessionId,
    autoPinnedArtifactIds: [],
    userPinnedArtifactIds: [],
    hiddenArtifactIds: [],
  });

  assert.deepEqual(selected, []);
});

test("identifies highlighted board artifacts with plot before json fallback", () => {
  assert.deepEqual(selectDefaultHighlightedBoardArtifactIds(artifacts), [
    "plot-old",
    "plot-latest",
  ]);

  assert.deepEqual(
    selectDefaultHighlightedBoardArtifactIds([
      artifact("table-result", "table", "2026-06-16T10:01:01.000Z"),
      artifact("json-result", "json", "2026-06-16T10:01:02.000Z"),
    ]),
    ["json-result"],
  );
});

test("shows user pinned artifacts plus the default highlighted board artifacts", () => {
  const selected = selectVisibleBoardArtifactIds({
    artifacts,
    messages,
    sessionId,
    autoPinnedArtifactIds: ["plot-old", "plot-latest"],
    userPinnedArtifactIds: ["table-old"],
    hiddenArtifactIds: [],
  });

  assert.deepEqual(selected, [
    "plot-old",
    "table-old",
    "plot-latest",
  ]);
});

test("does not expand a manually pinned artifact to every artifact in its assistant turn", () => {
  const selected = selectVisibleBoardArtifactIds({
    artifacts,
    messages,
    sessionId,
    autoPinnedArtifactIds: ["plot-old", "plot-latest"],
    userPinnedArtifactIds: ["table-latest"],
    hiddenArtifactIds: [],
  });

  assert.deepEqual(selected, [
    "plot-old",
    "plot-latest",
    "table-latest",
  ]);
});

test("keeps hidden artifacts out of the default highlighted board", () => {
  const selected = selectVisibleBoardArtifactIds({
    artifacts,
    messages,
    sessionId,
    autoPinnedArtifactIds: [],
    userPinnedArtifactIds: [],
    hiddenArtifactIds: ["table-old"],
  });

  assert.equal(selected.includes("table-old"), false);
});

test("applies persisted board turn title overrides by stable turn key", () => {
  const headers = applyBoardTurnTitleOverrides(
    [
      { turnKey: "assistant-1", label: "Question 1: first question", firstArtifactId: "plot-old" },
      { turnKey: "assistant-2", label: "Question 2: latest question", firstArtifactId: "plot-latest" },
    ],
    {
      "assistant-1": "Finance summary",
      "assistant-2": "   ",
    },
  );

  assert.deepEqual(headers, [
    { turnKey: "assistant-1", label: "Finance summary", firstArtifactId: "plot-old" },
    { turnKey: "assistant-2", label: "Question 2: latest question", firstArtifactId: "plot-latest" },
  ]);
});
