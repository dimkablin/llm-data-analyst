import React, { useState } from "react";
import { ChevronRight } from "lucide-react";

type Props = {
  content: string;
  defaultCollapsed?: boolean;
  /** Имя инструмента, перед которым идет рассуждение, или undefined для финальных блоков. */
  sourceLabel?: string;
};

function firstMeaningfulLine(text: string): string {
  const lines = text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  const line =
    lines.find((l) => !l.startsWith("#") && !l.startsWith("```")) ??
    lines[0] ??
    "";
  return line.replace(/^[∴•\-*>]+\s*/, "").trim();
}

export function ThinkingBlock({ content, defaultCollapsed = true, sourceLabel }: Props) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const summary = firstMeaningfulLine(content);

  if (!content.trim()) return null;

  return (
    <div className="group/thinking">
      <button
        type="button"
        onClick={() => setCollapsed((v) => !v)}
        className="flex w-full items-center gap-1.5 rounded-lg px-2 py-1 text-left transition-colors hover:bg-muted/30"
      >
        <ChevronRight
          className={`h-3 w-3 shrink-0 text-muted-foreground/40 transition-transform ${
            collapsed ? "" : "rotate-90"
          }`}
        />
        <span className="text-[12px] font-medium text-muted-foreground/50 select-none">
          Рассуждение
        </span>
        {sourceLabel ? (
          <span className="shrink-0 rounded px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground/40 bg-muted/30 select-none">
            {sourceLabel}
          </span>
        ) : null}
        {collapsed && summary ? (
          <span className="truncate text-[12px] text-muted-foreground/40 italic">
            {summary}
          </span>
        ) : null}
      </button>
      {!collapsed ? (
        <div className="ml-5 mt-1 rounded-lg border border-border/20 bg-muted/10 px-3 py-2 font-mono text-[11px] leading-relaxed text-muted-foreground/60 whitespace-pre-wrap">
          {content}
        </div>
      ) : null}
    </div>
  );
}

/** Streaming variant — shows live thinking text with animated indicator. */
export function LiveThinkingBlock({ content }: { content: string }) {
  const summary = firstMeaningfulLine(content);

  return (
    <div className="flex items-center gap-1.5 px-2 py-1">
      <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-primary/40" />
      <span className="text-[12px] font-medium text-muted-foreground/50 select-none">
        Рассуждение
      </span>
      {summary ? (
        <span className="truncate text-[12px] text-muted-foreground/40 italic">
          {summary}
        </span>
      ) : (
        <span className="text-[12px] text-muted-foreground/30">…</span>
      )}
    </div>
  );
}
