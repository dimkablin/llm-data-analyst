import React, { useEffect, useState } from "react";
import { motion } from "motion/react";
import type { StreamToolCall } from "../../lib/backend-types";
import { getStoredSpinner, getSpinnerFrames, SPINNER_CHANGED_EVENT } from "../../lib/spinner";

// ─── Spinner (выбирается в настройках аккаунта) ───────────────────────────────

export function useSpinnerFrame() {
  const [frames, setFrames] = useState<string[]>(() => getSpinnerFrames(getStoredSpinner()));
  const [frame, setFrame] = useState(0);

  useEffect(() => {
    const onChanged = () => setFrames(getSpinnerFrames(getStoredSpinner()));
    window.addEventListener(SPINNER_CHANGED_EVENT, onChanged);
    return () => window.removeEventListener(SPINNER_CHANGED_EVENT, onChanged);
  }, []);

  useEffect(() => {
    setFrame(0);
    const t = setInterval(() => setFrame((f) => (f + 1) % frames.length), 120);
    return () => clearInterval(t);
  }, [frames]);

  return frames[frame] ?? frames[0] ?? "◐";
}

// ─── Text helpers ─────────────────────────────────────────────────────────────

function stripLeadingSymbols(line: string): string {
  return line.replace(/^[∴•\-*>]+\s*/, "").trim();
}

function firstMeaningfulLine(text: string): string {
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  const line = lines.find((l) => !l.startsWith("#")) ?? lines[0] ?? "";
  return stripLeadingSymbols(line);
}

function InlineMarkdown({ text }: { text: string }) {
  const parts = text.split(/`([^`]+)`/);
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <code
            key={i}
            className="rounded bg-muted/40 px-[3px] py-[1px] font-mono text-[11px] not-italic text-foreground/70"
          >
            {part}
          </code>
        ) : (
          part
        ),
      )}
    </>
  );
}

function parseToolInputCode(preview: string): string | null {
  const trimmed = preview.trim();
  if (!trimmed) return null;
  try {
    const parsed = JSON.parse(trimmed) as Record<string, unknown>;
    if (typeof parsed.code === "string") return parsed.code;
    if (typeof parsed.query === "string") return parsed.query;
    return JSON.stringify(parsed, null, 2);
  } catch {
    return trimmed;
  }
}

// ─── Reasoning text between steps ────────────────────────────────────────────

function ReasoningText({ text }: { text: string }) {
  const line = firstMeaningfulLine(text);
  if (!line) return null;
  return (
    <p className="pl-1 text-[13px] text-muted-foreground/75 leading-5 select-none">
      <InlineMarkdown text={line} />
    </p>
  );
}

function LiveReasoningText({ text }: { text: string }) {
  const line = firstMeaningfulLine(text);
  return (
    <p className="pl-1 text-[13px] text-muted-foreground/75 leading-5 select-none">
      {line ? <InlineMarkdown text={line} /> : "…"}
    </p>
  );
}

// ─── Tool row ─────────────────────────────────────────────────────────────────

function ToolRow({ call, isLast }: { call: StreamToolCall; isLast: boolean }) {
  const spinner = useSpinnerFrame();
  const [elapsed, setElapsed] = useState(0);
  const [expanded, setExpanded] = useState(false);

  const isRunning = call.status === "running";

  useEffect(() => {
    if (!isRunning) return;
    const t = setInterval(
      () => setElapsed(Math.floor((Date.now() - call.started_at) / 1000)),
      1000,
    );
    return () => clearInterval(t);
  }, [isRunning, call.started_at]);

  const connector = isLast ? "└─" : "├─";
  const inputCode = call.input_preview ? parseToolInputCode(call.input_preview) : null;
  const hasDetail = !!(inputCode || call.output_preview || call.artifact_keys?.length);

  const dot = isRunning ? (
    <span className="text-primary">{spinner}</span>
  ) : call.status === "error" ? (
    <span className="text-destructive">●</span>
  ) : (
    <span className="text-emerald-500">●</span>
  );

  return (
    <div className="flex flex-col">
      {/* Main row */}
      <div
        className={`flex items-baseline gap-1.5 font-mono text-[12px] leading-5 ${hasDetail ? "cursor-pointer" : ""}`}
        onClick={hasDetail ? () => setExpanded((v) => !v) : undefined}
      >
        <span className="select-none text-muted-foreground/30">{connector}</span>
        <span className="w-3 shrink-0 text-center">{dot}</span>
        <span className={`font-semibold ${isRunning ? "text-foreground" : "text-muted-foreground"}`}>
          {call.tool_name}
        </span>
        {call.input_summary ? (
          <span className="truncate text-muted-foreground/50">
            ({call.input_summary}{call.input_summary.length >= 60 ? "…" : ""})
          </span>
        ) : null}
        {!isRunning && call.artifact_keys?.length ? (
          <span className="ml-1 shrink-0 text-emerald-500/70">
            → {call.artifact_keys.join(", ")}
          </span>
        ) : null}
        {hasDetail ? (
          <span className="ml-auto shrink-0 select-none text-[10px] text-muted-foreground/25">
            {expanded ? "▲" : "▼"}
          </span>
        ) : null}
      </div>

      {/* Elapsed timer while running */}
      {isRunning && elapsed >= 2 ? (
        <div className="ml-[3.25rem] font-mono text-[11px] leading-4 text-muted-foreground/40">
          ⎿&nbsp;&nbsp;{elapsed}s
        </div>
      ) : null}

      {/* Expanded: input + output sections */}
      {expanded ? (
        <div className="ml-[3.25rem] mt-1 flex flex-col gap-2">
          {inputCode ? (
            <div className="flex flex-col gap-0.5">
              <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground/35 select-none">
                Input
              </span>
              <div className="overflow-x-auto rounded border border-border/25 bg-muted/20 px-3 py-2 font-mono text-[11px] leading-relaxed text-muted-foreground/65 whitespace-pre">
                {inputCode}
              </div>
            </div>
          ) : null}

          {call.output_preview ? (
            <div className="flex flex-col gap-0.5">
              <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground/35 select-none">
                Output
              </span>
              <div className="overflow-x-auto rounded border border-border/25 bg-muted/20 px-3 py-2 font-mono text-[11px] leading-relaxed text-muted-foreground/65 whitespace-pre-wrap">
                {call.output_preview}
              </div>
            </div>
          ) : call.artifact_keys?.length ? (
            <div className="flex flex-col gap-0.5">
              <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground/35 select-none">
                Output
              </span>
              <div className="rounded border border-border/25 bg-muted/20 px-3 py-2 font-mono text-[11px] text-emerald-500/70">
                {call.artifact_keys.join(", ")}
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

// ─── Activity list — shared for streaming and completed messages ──────────────

export function ToolCallList({
  tools,
  reasoning,
  isLive,
}: {
  tools: StreamToolCall[];
  reasoning?: string;
  isLive?: boolean;
}) {
  if (!tools.length && !reasoning?.trim()) return null;

  const liveReasoning = reasoning?.trim() ?? "";

  return (
    <div className="flex flex-col gap-0.5 py-0.5">
      {/* Pure-thinking phase before any tool has fired */}
      {isLive && !tools.length && liveReasoning ? (
        <LiveReasoningText text={liveReasoning} />
      ) : null}

      {tools.map((call, idx) => {
        const isLast = idx === tools.length - 1;
        return (
          <React.Fragment key={call.id}>
            {call.pre_reasoning ? <ReasoningText text={call.pre_reasoning} /> : null}
            <ToolRow call={call} isLast={isLast && !liveReasoning} />
          </React.Fragment>
        );
      })}

      {/* Live reasoning streaming after last tool */}
      {isLive && tools.length > 0 && liveReasoning ? (
        <LiveReasoningText text={liveReasoning} />
      ) : null}
    </div>
  );
}

// ─── Live streaming feed ──────────────────────────────────────────────────────

type Props = {
  tools: StreamToolCall[];
  /** Live reasoning delta since last tool_start (reset on each tool_start). */
  reasoning: string;
  draft: string;
};

export function AgentActivityFeed({ tools, reasoning, draft }: Props) {
  const spinner = useSpinnerFrame();

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex gap-3 lg:gap-4"
    >
      {/* Spinner avatar */}
      <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-primary/10 font-mono text-sm text-primary select-none">
        {spinner}
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-2">
        <ToolCallList tools={tools} reasoning={reasoning} isLive />

        {/* Streaming answer text */}
        <div className="rounded-2xl rounded-tl-none border border-border/40 bg-card px-4 py-3 text-[13px] leading-relaxed lg:px-5 lg:py-4 lg:text-[14px]">
          {draft || <span className="text-muted-foreground/50">…</span>}
        </div>
      </div>
    </motion.div>
  );
}
