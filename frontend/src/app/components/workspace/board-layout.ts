import type { ArtifactPayload } from "../../lib/backend-types";

export const MIN_ARTIFACT_HEIGHT = 220;
export const MAX_ARTIFACT_HEIGHT = 800;
export const GRID_COLUMNS = 12;
const BOARD_CARD_HEADER_PX = 54;
const BOARD_CARD_BODY_PADDING_PX = 12;
const BOARD_CARD_CODE_PX = 52;
const BOARD_COLUMN_GAP_PX = 12;
export const BOARD_GAP_PX = BOARD_COLUMN_GAP_PX;
export const BOARD_TURN_HEADER_HEIGHT_PX = 48;
export const DEFAULT_ARTIFACT_WIDTH_UNITS = 6;
export const MIN_ARTIFACT_WIDTH_UNITS = 4;
export const MAX_ARTIFACT_WIDTH_UNITS = 12;

export type ArtifactBoardLayout = {
  colStart: number;
  widthUnits: number;
  topPx: number;
  heightPx: number;
};

export type TurnHeaderBoardLayout = {
  turnKey: string;
  label: string;
  topPx: number;
};

function clampWidthUnitsValue(value: number): number {
  return Math.max(
    MIN_ARTIFACT_WIDTH_UNITS,
    Math.min(MAX_ARTIFACT_WIDTH_UNITS, Math.round(value)),
  );
}

function estimateBoardCardHeight(
  artifact: ArtifactPayload,
  contentHeight: number,
  showCode: boolean,
): number {
  const hasCode =
    showCode && typeof artifact.meta?.code === "string" && artifact.meta.code.length > 0;
  return (
    BOARD_CARD_HEADER_PX +
    BOARD_CARD_BODY_PADDING_PX +
    contentHeight +
    (hasCode ? BOARD_CARD_CODE_PX : 0)
  );
}

export function computeBoardLayouts(
  artifacts: ArtifactPayload[],
  turnHeaders: Array<{ turnKey: string; label: string; firstArtifactId: string }>,
  widthMap: Record<string, number>,
  heightMap: Record<string, number>,
  colStartMap: Record<string, number>,
  measuredHeights: Record<string, number>,
  showCode: boolean,
  estimateHeight: (artifact: ArtifactPayload) => number,
): {
  layouts: Map<string, ArtifactBoardLayout>;
  turnHeaderLayouts: TurnHeaderBoardLayout[];
  boardHeight: number;
} {
  const columnBottom = Array.from({ length: GRID_COLUMNS }, () => 0);
  const layouts = new Map<string, ArtifactBoardLayout>();
  const turnHeaderLayouts: TurnHeaderBoardLayout[] = [];
  const headerByFirstArtifactId = new Map(
    turnHeaders.map((header) => [header.firstArtifactId, header]),
  );

  for (const artifact of artifacts) {
    const sectionHeader = headerByFirstArtifactId.get(artifact.id);
    if (sectionHeader) {
      const headerTopPx = Math.max(0, ...columnBottom);
      turnHeaderLayouts.push({
        turnKey: sectionHeader.turnKey,
        label: sectionHeader.label,
        topPx: headerTopPx,
      });
      const belowHeader =
        headerTopPx + BOARD_TURN_HEADER_HEIGHT_PX + BOARD_COLUMN_GAP_PX;
      for (let col = 0; col < GRID_COLUMNS; col += 1) {
        columnBottom[col] = belowHeader;
      }
    }
    const requestedWidth = Number(artifact.meta?.board_width_units);
    const widthUnits = Number.isFinite(requestedWidth)
      ? clampWidthUnitsValue(requestedWidth)
      : artifact.meta?.full_width === true
        ? GRID_COLUMNS
        : clampWidthUnitsValue(widthMap[artifact.id] ?? DEFAULT_ARTIFACT_WIDTH_UNITS);
    const contentHeight = heightMap[artifact.id] ?? estimateHeight(artifact);
    const heightPx =
      measuredHeights[artifact.id] ??
      estimateBoardCardHeight(artifact, contentHeight, showCode);
    const preferredColStart = colStartMap[artifact.id];

    let colStart = 0;
    let topPx = 0;

    if (preferredColStart != null) {
      colStart = Math.max(
        0,
        Math.min(GRID_COLUMNS - widthUnits, Math.round(preferredColStart) - 1),
      );
      for (let col = colStart; col < colStart + widthUnits; col += 1) {
        topPx = Math.max(topPx, columnBottom[col]);
      }
    } else {
      let bestTop = Number.POSITIVE_INFINITY;
      for (let candidateCol = 0; candidateCol <= GRID_COLUMNS - widthUnits; candidateCol += 1) {
        let candidateTop = 0;
        for (let col = candidateCol; col < candidateCol + widthUnits; col += 1) {
          candidateTop = Math.max(candidateTop, columnBottom[col]);
        }
        if (
          candidateTop < bestTop ||
          (candidateTop === bestTop && candidateCol < colStart)
        ) {
          bestTop = candidateTop;
          colStart = candidateCol;
        }
      }
      topPx = bestTop;
    }

    layouts.set(artifact.id, {
      colStart: colStart + 1,
      widthUnits,
      topPx,
      heightPx,
    });

    const nextBottom = topPx + heightPx + BOARD_COLUMN_GAP_PX;
    for (let col = colStart; col < colStart + widthUnits; col += 1) {
      columnBottom[col] = nextBottom;
    }
  }

  const boardHeight = Math.max(0, ...columnBottom);
  return { layouts, turnHeaderLayouts, boardHeight };
}

export function estimateAutoHeight(artifact: ArtifactPayload): number {
  if (artifact.type === "plot") {
    return 440;
  }

  if (artifact.type === "note" && artifact.data.format === "markdown") {
    const content = String((artifact.data.data as { content?: unknown })?.content ?? "");
    const lines = Math.max(1, content.split("\n").length);
    return Math.max(MIN_ARTIFACT_HEIGHT, Math.min(MAX_ARTIFACT_HEIGHT, 180 + lines * 18));
  }

  if (artifact.type === "table" && artifact.data.format === "split") {
    const raw = artifact.data.data as { data?: unknown[][]; columns?: unknown[] };
    const rows = Array.isArray(raw.data) ? raw.data.length : 0;
    const cols = Array.isArray(raw.columns) ? raw.columns.length : 0;
    const headerAndPadding = 120;
    const rowHeight = 34;
    return Math.max(
      140,
      Math.min(
        MAX_ARTIFACT_HEIGHT,
        headerAndPadding + Math.min(rows, 20) * rowHeight + Math.min(cols, 12) * 4,
      ),
    );
  }

  if (artifact.type === "value" && artifact.data.format === "value") {
    const data = artifact.data.data as Record<string, unknown>;
    const entries = Object.keys(data ?? {}).length;
    return Math.max(MIN_ARTIFACT_HEIGHT, Math.min(MAX_ARTIFACT_HEIGHT, 170 + entries * 40));
  }

  const jsonLength = JSON.stringify(artifact.data.data ?? "").length;
  return Math.max(MIN_ARTIFACT_HEIGHT, Math.min(MAX_ARTIFACT_HEIGHT, 200 + Math.min(jsonLength, 4500) / 14));
}
