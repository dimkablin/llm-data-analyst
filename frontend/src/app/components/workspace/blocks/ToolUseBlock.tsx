import React, { useEffect, useState } from "react";
import { ChevronRight, Loader2 } from "lucide-react";

type Props = {
  tool_name: string;
  input_summary: string;
  input_code?: string;
  status: "running" | "done" | "error";
  started_at: number;
  /** Result summary shown inline after tool completes. */
  result_summary?: string;
  output_preview?: string;
  artifact_keys?: string[];
};

export function ToolUseBlock({
  tool_name,
  input_summary,
  input_code,
  status,
  started_at,
  result_summary,
  output_preview,
  artifact_keys,
}: Props) {
  const [elapsed, setElapsed] = useState(0);
  const [expanded, setExpanded] = useState(false);
  const isRunning = status === "running";

  useEffect(() => {
    if (!isRunning) return;
    const t = setInterval(
      () => setElapsed(Math.floor((Date.now() - started_at) / 1000)),
      1000,
    );
    return () => clearInterval(t);
  }, [isRunning, started_at]);

  const hasDetail = !!(input_code || output_preview || (artifact_keys?.length));

  const statusIcon = isRunning ? (
    <Loader2 className="h-3 w-3 shrink-0 animate-spin text-primary" />
  ) : status === "error" ? (
    <span className="flex h-3 w-3 shrink-0 items-center justify-center text-[10px] text-destructive">●</span>
  ) : (
    <span className="flex h-3 w-3 shrink-0 items-center justify-center text-[10px] text-emerald-500">●</span>
  );

  return (
    <div className="flex flex-col">
      {/* Main tool call row */}
      <div
        className={`flex items-center gap-2 rounded-lg px-2 py-1 transition-colors ${
          hasDetail ? "cursor-pointer hover:bg-muted/20" : ""
        }`}
        onClick={hasDetail ? () => setExpanded((v) => !v) : undefined}
      >
        {statusIcon}
        <span
          className={`font-mono text-[12px] font-semibold ${
            isRunning ? "text-foreground" : "text-muted-foreground"
          }`}
        >
          {tool_name}
        </span>
        {input_summary ? (
          <span className="truncate font-mono text-[12px] text-muted-foreground/40">
            {input_summary}
          </span>
        ) : null}

        {/* Elapsed timer */}
        {isRunning && elapsed >= 2 ? (
          <span className="ml-auto shrink-0 font-mono text-[11px] text-muted-foreground/30">
            {elapsed}s
          </span>
        ) : null}

        {/* Artifact badges */}
        {!isRunning && artifact_keys?.length ? (
          <span className="ml-auto shrink-0 font-mono text-[11px] text-emerald-500/60">
            → {artifact_keys.join(", ")}
          </span>
        ) : null}

        {/* Result summary inline */}
        {!isRunning && !artifact_keys?.length && result_summary ? (
          <span className={`ml-auto shrink-0 truncate font-mono text-[11px] ${status === "error" ? "text-destructive/70" : "text-muted-foreground/40"}`}>
            {result_summary}
          </span>
        ) : null}

        {hasDetail ? (
          <ChevronRight
            className={`ml-auto h-3 w-3 shrink-0 text-muted-foreground/20 transition-transform ${
              expanded ? "rotate-90" : ""
            }`}
          />
        ) : null}
      </div>

      {/* Expanded detail */}
      {expanded ? (
        <div className="ml-7 mt-1 flex flex-col gap-2">
          {input_code ? (
            <div className="flex flex-col gap-0.5">
              <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground/30 select-none">
                Input
              </span>
              <div className="overflow-x-auto rounded border border-border/20 bg-muted/15 px-3 py-2 font-mono text-[11px] leading-relaxed text-muted-foreground/60 whitespace-pre">
                {input_code}
              </div>
            </div>
          ) : null}

          {output_preview ? (
            <div className="flex flex-col gap-0.5">
              <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground/30 select-none">
                {status === "error" ? "Error" : "Output"}
              </span>
              <div className={`overflow-x-auto rounded border px-3 py-2 font-mono text-[11px] leading-relaxed whitespace-pre-wrap ${status === "error" ? "border-destructive/20 bg-destructive/5 text-destructive/80" : "border-border/20 bg-muted/15 text-muted-foreground/60"}`}>
                {output_preview}
              </div>
            </div>
          ) : !output_preview && artifact_keys?.length ? (
            <div className="flex flex-col gap-0.5">
              <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground/30 select-none">
                Output
              </span>
              <div className="rounded border border-border/20 bg-muted/15 px-3 py-2 font-mono text-[11px] text-emerald-500/60">
                {artifact_keys.join(", ")}
              </div>
            </div>
          ) : null}

          {result_summary && !output_preview ? (
            <div className="font-mono text-[11px] text-muted-foreground/50">
              {result_summary}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
