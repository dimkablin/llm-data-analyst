import type { ArtifactPayload, ChatMessage } from "./backend-types";

type TurnBucket = {
  plots: string[];
  other: string[];
  notes: string[];
};

function getAssistantTurns(messages: ChatMessage[]): Array<{ messageId: string; timestamp: string }> {
  return messages
    .filter((message) => message.role === "assistant")
    .map((message) => ({
      messageId: message.id,
      timestamp: message.timestamp,
    }));
}

function buildArtifactTurnIndex(
  messages: ChatMessage[],
  sessionId: string,
  artifacts: ArtifactPayload[],
): Map<string, string> {
  const index = new Map<string, string>();

  for (const message of messages) {
    if (message.role !== "assistant") {
      continue;
    }
    for (const artifact of message.artifacts ?? []) {
      if (artifact.id) {
        index.set(artifact.id, message.id);
      }
    }
    index.set(messageNoteArtifactId(sessionId, message.id), message.id);
  }

  const notePrefix = `msg_note_${sessionId}_`;
  for (const artifact of artifacts) {
    if (artifact.type === "note" && artifact.id.startsWith(notePrefix)) {
      index.set(artifact.id, artifact.id.slice(notePrefix.length));
    }
  }

  return index;
}

function resolveOrphanTurnId(
  artifact: ArtifactPayload,
  turns: Array<{ messageId: string; timestamp: string }>,
): string | null {
  if (!turns.length) {
    return null;
  }
  const artifactTs = Date.parse(artifact.timestamp);
  if (!Number.isFinite(artifactTs)) {
    return turns[turns.length - 1]?.messageId ?? null;
  }

  let resolved = turns[0].messageId;
  for (const turn of turns) {
    const turnTs = Date.parse(turn.timestamp);
    if (Number.isFinite(turnTs) && turnTs <= artifactTs) {
      resolved = turn.messageId;
    }
  }
  return resolved;
}

/** Board order: per assistant answer → plots, then other, then note; turns chronological. */
export function sortBoardArtifactIdsByTurn(
  ids: string[],
  artifacts: ArtifactPayload[],
  messages: ChatMessage[],
  sessionId: string,
): string[] {
  const artifactById = new Map(artifacts.map((artifact) => [artifact.id, artifact]));
  const turns = getAssistantTurns(messages);
  const turnIndex = buildArtifactTurnIndex(messages, sessionId, artifacts);

  const buckets = new Map<string, TurnBucket>();
  for (const turn of turns) {
    buckets.set(turn.messageId, { plots: [], other: [], notes: [] });
  }
  const orphanBucket: TurnBucket = { plots: [], other: [], notes: [] };

  for (const id of ids) {
    const artifact = artifactById.get(id);
    if (!artifact) {
      continue;
    }
    const turnId = turnIndex.get(id) ?? resolveOrphanTurnId(artifact, turns);
    const bucket =
      turnId && buckets.has(turnId) ? buckets.get(turnId)! : orphanBucket;

    if (artifact.type === "plot") {
      bucket.plots.push(id);
    } else if (artifact.type === "note") {
      bucket.notes.push(id);
    } else {
      bucket.other.push(id);
    }
  }

  const ordered: string[] = [];
  for (const turn of turns) {
    const bucket = buckets.get(turn.messageId);
    if (!bucket) {
      continue;
    }
    ordered.push(...bucket.plots, ...bucket.other, ...bucket.notes);
  }
  ordered.push(...orphanBucket.plots, ...orphanBucket.other, ...orphanBucket.notes);

  for (const id of ids) {
    if (!ordered.includes(id)) {
      ordered.push(id);
    }
  }
  return ordered;
}

/** @deprecated Use sortBoardArtifactIdsByTurn — kept as alias for imports. */
export function sortBoardArtifactIds(
  ids: string[],
  artifacts: ArtifactPayload[],
  messages: ChatMessage[] = [],
  sessionId = "",
): string[] {
  if (messages.length > 0 && sessionId) {
    return sortBoardArtifactIdsByTurn(ids, artifacts, messages, sessionId);
  }
  return ids;
}

export type BoardTurnSectionHeader = {
  turnKey: string;
  label: string;
  firstArtifactId: string;
};

export type BoardTurnTitleOverrides = Record<string, string>;

export type BoardArtifactSelectionInput = {
  artifacts: ArtifactPayload[];
  messages: ChatMessage[];
  sessionId: string;
  autoPinnedArtifactIds: string[];
  userPinnedArtifactIds: string[];
  hiddenArtifactIds?: string[];
};

export function selectDefaultHighlightedBoardArtifactIds(
  artifacts: ArtifactPayload[],
): string[] {
  const plotIds = artifacts
    .filter((artifact) => artifact.type === "plot")
    .map((artifact) => artifact.id);
  if (plotIds.length > 0) {
    return plotIds;
  }

  return artifacts
    .filter((artifact) => artifact.type === "json")
    .map((artifact) => artifact.id);
}

function uniqueValidArtifactIds(
  ids: string[],
  artifactById: Map<string, ArtifactPayload>,
  hiddenIds: Set<string>,
): string[] {
  const selected: string[] = [];
  for (const id of ids) {
    if (hiddenIds.has(id) || !artifactById.has(id) || selected.includes(id)) {
      continue;
    }
    selected.push(id);
  }
  return selected;
}

export function selectVisibleBoardArtifactIds({
  artifacts,
  messages,
  sessionId,
  userPinnedArtifactIds,
  hiddenArtifactIds = [],
}: BoardArtifactSelectionInput): string[] {
  const hiddenIds = new Set(hiddenArtifactIds);
  const visibleArtifacts = artifacts.filter((artifact) => !hiddenIds.has(artifact.id));
  const artifactById = new Map(visibleArtifacts.map((artifact) => [artifact.id, artifact]));
  const explicitPinnedIds = uniqueValidArtifactIds(userPinnedArtifactIds, artifactById, hiddenIds);
  const defaultHighlightedIds = uniqueValidArtifactIds(
    selectDefaultHighlightedBoardArtifactIds(visibleArtifacts),
    artifactById,
    hiddenIds,
  );

  if (explicitPinnedIds.length === 0) {
    return sortBoardArtifactIdsByTurn(
      defaultHighlightedIds,
      visibleArtifacts,
      messages,
      sessionId,
    );
  }

  return sortBoardArtifactIdsByTurn(
    uniqueValidArtifactIds([...defaultHighlightedIds, ...explicitPinnedIds], artifactById, hiddenIds),
    visibleArtifacts,
    messages,
    sessionId,
  );
}

export function applyBoardTurnTitleOverrides(
  headers: BoardTurnSectionHeader[],
  overrides: BoardTurnTitleOverrides,
): BoardTurnSectionHeader[] {
  return headers.map((header) => {
    const override = overrides[header.turnKey]?.trim();
    if (!override) {
      return header;
    }
    return { ...header, label: override };
  });
}

function findUserPromptBeforeAssistant(
  messages: ChatMessage[],
  assistantMessageId: string,
): string {
  const assistantIdx = messages.findIndex((message) => message.id === assistantMessageId);
  if (assistantIdx >= 0) {
    for (let index = assistantIdx - 1; index >= 0; index -= 1) {
      const message = messages[index];
      if (message?.role === "user") {
        return message.content.trim().replace(/\s+/g, " ");
      }
    }
  }
  const lastUser = [...messages].reverse().find((message) => message.role === "user");
  return lastUser?.content.trim().replace(/\s+/g, " ") ?? "";
}

function buildTurnLabel(
  turnMessageId: string,
  turnIndex: number,
  messages: ChatMessage[],
  userQuestionHint = "",
): string {
  if (turnMessageId.startsWith("orphan")) {
    return userQuestionHint
      ? `Вопрос ${turnIndex}: ${userQuestionHint}`
      : `Блок ${turnIndex}`;
  }
  const userLine =
    userQuestionHint || findUserPromptBeforeAssistant(messages, turnMessageId);
  if (userLine) {
    return `Вопрос ${turnIndex}: ${userLine}`;
  }
  const assistantIdx = messages.findIndex((message) => message.id === turnMessageId);
  if (assistantIdx < 0) {
    return `Ответ ${turnIndex}`;
  }
  return `Ответ ${turnIndex}`;
}

/** Section labels for the board — one header before each assistant answer group. */
export function buildBoardTurnHeaders(
  orderedArtifactIds: string[],
  artifacts: ArtifactPayload[],
  messages: ChatMessage[],
  sessionId: string,
): BoardTurnSectionHeader[] {
  const artifactById = new Map(artifacts.map((artifact) => [artifact.id, artifact]));
  const turns = getAssistantTurns(messages);
  const turnIndex = buildArtifactTurnIndex(messages, sessionId, artifacts);
  const headers: BoardTurnSectionHeader[] = [];
  let lastTurnKey: string | null = null;
  let turnNumber = 0;

  for (const artifactId of orderedArtifactIds) {
    const artifact = artifactById.get(artifactId);
    if (!artifact) {
      continue;
    }
    const assistantMessageId =
      turnIndex.get(artifactId) ?? resolveOrphanTurnId(artifact, turns);
    const turnKey = assistantMessageId ?? `orphan-${artifactId}`;
    if (turnKey === lastTurnKey) {
      continue;
    }
    turnNumber += 1;
    lastTurnKey = turnKey;
    const userQuestionHint =
      typeof artifact.meta?.user_question === "string"
        ? String(artifact.meta.user_question).trim()
        : "";
    headers.push({
      turnKey,
      label: buildTurnLabel(turnKey, turnNumber, messages, userQuestionHint),
      firstArtifactId: artifactId,
    });
  }

  return headers;
}

export function mergePinnedIdsForBoard(
  prev: string[],
  newcomerIds: string[],
  artifacts: ArtifactPayload[],
  messages: ChatMessage[],
  sessionId: string,
): string[] {
  const merged = [...prev];
  for (const id of newcomerIds) {
    if (!merged.includes(id)) {
      merged.push(id);
    }
  }
  return sortBoardArtifactIdsByTurn(merged, artifacts, messages, sessionId);
}

export function messageNoteArtifactId(sessionId: string, messageId: string): string {
  return `msg_note_${sessionId}_${messageId}`;
}

export type BoardExportSectionPayload = {
  label: string;
  artifact_ids: string[];
};

/** Group visible board artifacts by «Вопрос N» headers for DOCX/PDF export. */
export function buildBoardExportSections(
  orderedArtifacts: Array<{ id: string }>,
  turnHeaders: BoardTurnSectionHeader[],
): BoardExportSectionPayload[] {
  if (!orderedArtifacts.length) {
    return [];
  }
  if (!turnHeaders.length) {
    return [{ label: "", artifact_ids: orderedArtifacts.map((artifact) => artifact.id) }];
  }

  const headerByFirstId = new Map(
    turnHeaders.map((header) => [header.firstArtifactId, header]),
  );
  const sections: BoardExportSectionPayload[] = [];
  let current: BoardExportSectionPayload | null = null;

  for (const artifact of orderedArtifacts) {
    const header = headerByFirstId.get(artifact.id);
    if (header) {
      if (current && current.artifact_ids.length > 0) {
        sections.push(current);
      }
      current = { label: header.label, artifact_ids: [] };
    }
    if (!current) {
      current = { label: "", artifact_ids: [] };
    }
    current.artifact_ids.push(artifact.id);
  }

  if (current && current.artifact_ids.length > 0) {
    sections.push(current);
  }

  return sections;
}

export function buildMessageNoteArtifact(
  sessionId: string,
  messageId: string,
  content: string,
  timestamp: string,
  auto = false,
  userQuestion = "",
): ArtifactPayload | null {
  const trimmed = content.trim();
  if (!trimmed) {
    return null;
  }
  const id = messageNoteArtifactId(sessionId, messageId);
  const title = userQuestion.trim()
    ? `Ответ: ${userQuestion.trim().slice(0, 72)}`
    : "Аналитическая записка";
  return {
    id,
    type: "note",
    text: title.slice(0, 90),
    role: "assistant",
    meta: {
      source: auto ? "chat_message_auto" : "chat_message",
      message_id: messageId,
      user_question: userQuestion.trim() || undefined,
    },
    timestamp,
    data: {
      format: "markdown",
      data: {
        content: trimmed,
      },
    },
  };
}
