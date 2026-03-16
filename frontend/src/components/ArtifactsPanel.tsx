import { useEffect, useMemo, type RefObject } from "react";
import { GridLayout, useContainerWidth, type Layout, type LayoutItem } from "react-grid-layout";

import type { ArtifactPayload } from "../types";
import { ArtifactCard } from "./ArtifactCard";

export type DashboardLayoutItem = {
  i: string;
  x: number;
  y: number;
  w: number;
  h: number;
  minW?: number;
  minH?: number;
  maxW?: number;
};

type ArtifactsPanelProps = {
  artifacts: ArtifactPayload[];
  showCode: boolean;
  gridColumns: 2 | 3 | 4;
  themeMode: "light" | "dark";
  layoutLocked: boolean;
  layout: DashboardLayoutItem[];
  onLayoutChange: (nextLayout: DashboardLayoutItem[]) => void;
  onResetLayout: () => void;
  onUnpinArtifact: (artifactId: string) => void;
};

function clampLayoutItem(
  item: DashboardLayoutItem,
  gridColumns: number
): DashboardLayoutItem {
  const maxCols = Math.max(1, gridColumns);
  const nextW = Math.min(maxCols, Math.max(1, Math.round(item.w)));
  const nextX = Math.max(0, Math.min(Math.round(item.x), Math.max(0, maxCols - nextW)));

  return {
    ...item,
    x: nextX,
    y: Math.max(0, Math.round(item.y)),
    w: nextW,
    h: Math.max(2, Math.round(item.h)),
    minW: item.minW ? Math.max(1, Math.round(item.minW)) : undefined,
    minH: item.minH ? Math.max(2, Math.round(item.minH)) : undefined,
    maxW: item.maxW ? Math.max(1, Math.round(item.maxW)) : undefined
  };
}

function defaultLayoutForArtifact(
  artifact: ArtifactPayload,
  index: number,
  gridColumns: number
): DashboardLayoutItem {
  const cols = Math.max(1, gridColumns);
  if (artifact.type === "plot") {
    const width = Math.min(cols, cols >= 3 ? 2 : cols);
    return {
      i: artifact.id,
      x: index % cols,
      y: Number.POSITIVE_INFINITY,
      w: width,
      h: 9,
      minW: 1,
      minH: 6
    };
  }
  if (artifact.type === "table") {
    return {
      i: artifact.id,
      x: index % cols,
      y: Number.POSITIVE_INFINITY,
      w: Math.min(cols, 2),
      h: 8,
      minW: 1,
      minH: 5
    };
  }
  if (artifact.type === "value") {
    return {
      i: artifact.id,
      x: index % cols,
      y: Number.POSITIVE_INFINITY,
      w: 1,
      h: 6,
      minW: 1,
      maxW: Math.min(cols, 2),
      minH: 4
    };
  }
  return {
    i: artifact.id,
    x: index % cols,
    y: Number.POSITIVE_INFINITY,
    w: 1,
    h: 6,
    minW: 1,
    minH: 4
  };
}

function sameLayout(a: DashboardLayoutItem[], b: DashboardLayoutItem[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  const sortById = (left: DashboardLayoutItem, right: DashboardLayoutItem) =>
    left.i.localeCompare(right.i);
  const left = [...a].sort(sortById);
  const right = [...b].sort(sortById);
  for (let idx = 0; idx < left.length; idx += 1) {
    const l = left[idx];
    const r = right[idx];
    if (
      l.i !== r.i ||
      l.x !== r.x ||
      l.y !== r.y ||
      l.w !== r.w ||
      l.h !== r.h ||
      l.minW !== r.minW ||
      l.minH !== r.minH ||
      l.maxW !== r.maxW
    ) {
      return false;
    }
  }
  return true;
}

export function ArtifactsPanel({
  artifacts,
  showCode,
  gridColumns,
  themeMode,
  layoutLocked,
  layout,
  onLayoutChange,
  onResetLayout,
  onUnpinArtifact
}: ArtifactsPanelProps): JSX.Element {
  const counters = useMemo(
    () => ({
      total: artifacts.length,
      tables: artifacts.filter((item) => item.type === "table").length,
      plots: artifacts.filter((item) => item.type === "plot").length,
      values: artifacts.filter((item) => item.type === "value").length
    }),
    [artifacts]
  );
  const effectiveLayout = useMemo(() => {
    const byId = new Map(layout.map((item) => [item.i, item]));
    return artifacts.map((artifact, index) => {
      const fromState = byId.get(artifact.id);
      const seed = fromState ?? defaultLayoutForArtifact(artifact, index, gridColumns);
      return clampLayoutItem(
        {
          ...seed,
          i: artifact.id
        },
        gridColumns
      );
    });
  }, [artifacts, gridColumns, layout]);

  useEffect(() => {
    if (!sameLayout(layout, effectiveLayout)) {
      onLayoutChange(effectiveLayout);
    }
  }, [effectiveLayout, layout, onLayoutChange]);

  function handleGridLayoutChange(nextLayout: Layout): void {
    const byId = new Map(effectiveLayout.map((item) => [item.i, item]));
    const next = nextLayout
      .filter((item) => byId.has(item.i))
      .map((item) => {
        const base = byId.get(item.i);
        const merged: DashboardLayoutItem = {
          i: item.i,
          x: item.x,
          y: item.y,
          w: item.w,
          h: item.h,
          minW: base?.minW,
          minH: base?.minH,
          maxW: base?.maxW
        };
        return clampLayoutItem(merged, gridColumns);
      });
    if (!sameLayout(next, effectiveLayout)) {
      onLayoutChange(next);
    }
  }
  const { width, mounted, containerRef } = useContainerWidth({
    initialWidth: 1280
  });

  return (
    <section className="panel panel-dashboard">
      <div className="panel-head">
        <h2>Дашборд</h2>
        <div className="artifacts-head-tools">
          <span className="session-id">{counters.total} артефактов</span>
          <span className="session-id">{layoutLocked ? "макет: зафиксирован" : "макет: свободный"}</span>
          {artifacts.length > 0 ? (
            <button type="button" className="btn-ghost btn-xs" onClick={onResetLayout}>
              Сбросить макет
            </button>
          ) : null}
        </div>
      </div>
      <div className="eval-summary">
        <span>таблиц: {counters.tables}</span>
        <span>графиков: {counters.plots}</span>
        <span>значений: {counters.values}</span>
      </div>

      <div ref={containerRef as RefObject<HTMLDivElement>} className="artifacts dashboard-grid-shell">
        {artifacts.length === 0 ? (
          <div className="empty-state">
            Пока пусто. Добавляйте артефакты в дашборд кнопкой в чате рядом с каждым артефактом.
          </div>
        ) : (
          mounted ? (
            <GridLayout
              className="dashboard-grid-layout"
              width={width}
              layout={effectiveLayout as LayoutItem[]}
              gridConfig={{
                cols: gridColumns,
                rowHeight: 46,
                margin: [10, 10],
                containerPadding: [0, 0],
                maxRows: 120
              }}
              dragConfig={{
                enabled: artifacts.length > 1 && !layoutLocked,
                bounded: true,
                handle: ".artifact > header",
                cancel:
                  ".btn-ghost, .btn-inline, iframe, .plot-frame, .table-wrap, .artifact-code, details, summary, pre, code, input, textarea, select, button, a",
                threshold: 4
              }}
              resizeConfig={{
                enabled: !layoutLocked,
                handles: ["se"]
              }}
              onLayoutChange={handleGridLayoutChange}
            >
              {artifacts.map((artifact) => (
                <div key={artifact.id} className="dashboard-grid-item">
                  <ArtifactCard
                    artifact={artifact}
                    showCode={showCode}
                    themeMode={themeMode}
                    actionLabel="Убрать"
                    onAction={(item) => onUnpinArtifact(item.id)}
                  />
                </div>
              ))}
            </GridLayout>
          ) : null
        )}
      </div>
    </section>
  );
}
