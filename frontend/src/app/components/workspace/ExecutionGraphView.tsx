import { useMemo } from "react";
import { AnimatePresence, motion } from "motion/react";
import type { ExecutionGraph, GraphNode } from "../../lib/backend-types";
import { formatDurationMs } from "../../lib/format";

const STATUS_COLORS: Record<GraphNode["status"], string> = {
  pending: "bg-muted text-muted-foreground border-border/40",
  running: "bg-primary/15 text-primary border-primary/40",
  done: "bg-emerald-500/15 text-emerald-600 border-emerald-500/40 dark:text-emerald-400",
  error: "bg-destructive/15 text-destructive border-destructive/40",
};

const PHASE_ICONS: Record<string, string> = {
  think: "\u{1F9E0}",
  act: "\u26A1",
  evaluate: "\u{1F50D}",
  finalize: "\u2705",
  decide: "\u{1F4CB}",
};

function nodeIcon(node: GraphNode): string {
  if (node.type === "tool") return "\u{1F6E0}";
  const base = node.label.toLowerCase().split("-")[0].replace(/[^a-z]/g, "");
  return PHASE_ICONS[base] ?? "\u25CF";
}

function nodeLabel(node: GraphNode): string {
  if (node.type === "tool") {
    return node.tool_name ?? node.label;
  }
  const base = node.label.split("-")[0];
  const map: Record<string, string> = {
    think: "Think",
    act: "Act",
    evaluate: "Eval",
    finalize: "Final",
    decide: "Decide",
  };
  return map[base.toLowerCase()] ?? base;
}

function PhaseNode({ node }: { node: GraphNode }) {
  const isRunning = node.status === "running";
  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.85 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.85 }}
      transition={{ duration: 0.2 }}
      className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-[11px] font-semibold leading-none ${STATUS_COLORS[node.status]}`}
    >
      <span className="text-xs">{nodeIcon(node)}</span>
      <span>{nodeLabel(node)}</span>
      {isRunning ? (
        <motion.span
          animate={{ opacity: [1, 0.3, 1] }}
          transition={{ repeat: Infinity, duration: 1.2 }}
          className="ml-0.5 inline-block h-1.5 w-1.5 rounded-full bg-current"
        />
      ) : null}
      {node.duration_ms != null && node.status === "done" ? (
        <span className="ml-0.5 text-[10px] opacity-60">{formatDurationMs(node.duration_ms)}</span>
      ) : null}
    </motion.div>
  );
}

function ToolBadge({ node }: { node: GraphNode }) {
  const isRunning = node.status === "running";
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 4 }}
      transition={{ duration: 0.15 }}
      className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[10px] font-medium leading-none ${STATUS_COLORS[node.status]}`}
    >
      <span className="text-[10px]">{nodeIcon(node)}</span>
      <span className="max-w-[120px] truncate">{nodeLabel(node)}</span>
      {isRunning ? (
        <motion.span
          animate={{ opacity: [1, 0.3, 1] }}
          transition={{ repeat: Infinity, duration: 1.2 }}
          className="ml-0.5 inline-block h-1 w-1 rounded-full bg-current"
        />
      ) : null}
      {node.artifact_keys?.length ? (
        <span className="ml-0.5 opacity-60">({node.artifact_keys.length})</span>
      ) : null}
    </motion.div>
  );
}

function Arrow() {
  return (
    <svg
      width="28"
      height="20"
      viewBox="0 0 28 20"
      fill="none"
      className="mx-0.5 flex-shrink-0 self-center"
      aria-hidden
    >
      <line
        x1="2"
        y1="10"
        x2="22"
        y2="10"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        className="text-muted-foreground/30"
      />
      <path
        d="M20 6.5L25 10L20 13.5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-muted-foreground/40"
      />
    </svg>
  );
}

export function ExecutionGraphView({
  graph,
  isLive = false,
}: {
  graph: ExecutionGraph;
  isLive?: boolean;
}) {
  if (!graph.nodes.length) return null;

  const phaseNodes = graph.nodes.filter((n) => n.type === "phase");
  const toolNodes = graph.nodes.filter((n) => n.type === "tool");

  const toolsByParent = useMemo(() => {
    const map = new Map<string, GraphNode[]>();
    for (const t of toolNodes) {
      const pid = t.parent_id ?? "";
      const list = map.get(pid) ?? [];
      list.push(t);
      map.set(pid, list);
    }
    return map;
  }, [toolNodes]);

  return (
    <div className="flex flex-wrap items-start gap-y-2">
      <AnimatePresence mode="popLayout">
        {phaseNodes.map((phase, idx) => {
          const children = toolsByParent.get(phase.id) ?? [];
          return (
            <motion.div
              key={phase.id}
              layout
              className="flex flex-col items-start"
            >
              <div className="flex items-center">
                {idx > 0 ? <Arrow /> : null}
                <PhaseNode node={phase} />
              </div>
              {children.length > 0 ? (
                <div className={`flex flex-col items-start gap-1 mt-1 ${idx > 0 ? "ml-8" : ""}`}>
                  {children.map((tool) => (
                    <ToolBadge key={tool.id} node={tool} />
                  ))}
                </div>
              ) : null}
            </motion.div>
          );
        })}
      </AnimatePresence>
      {isLive ? (
        <motion.span
          animate={{ opacity: [0.3, 1, 0.3] }}
          transition={{ repeat: Infinity, duration: 1.5 }}
          className="ml-1 self-center text-[10px] text-muted-foreground"
        >
          ...
        </motion.span>
      ) : null}
    </div>
  );
}
