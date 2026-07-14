import { useEffect, useState } from "react";
import {
  Check,
  Circle,
  Compass,
  LineChart,
  Loader2,
  Microscope,
  Sparkles,
  Table2,
  type LucideIcon,
} from "lucide-react";
import { motion } from "motion/react";
import type { AgentStageId, AgentStageItem } from "../../lib/agent-stages";
import { pickStageRunningMessage } from "../../lib/agent-stages";

const STAGE_ICONS: Record<AgentStageId, LucideIcon> = {
  1: Compass,
  2: Table2,
  3: LineChart,
  4: Microscope,
  5: Sparkles,
};

function StageIcon({ status }: { status: AgentStageItem["status"] }) {
  if (status === "running") {
    return <Loader2 className="h-4 w-4 animate-spin text-primary" />;
  }
  if (status === "done") {
    return <Check className="h-4 w-4 text-emerald-500" strokeWidth={2.5} />;
  }
  return <Circle className="h-3.5 w-3.5 text-muted-foreground/30" strokeWidth={1.5} />;
}

function StageRow({
  stage,
  isLast,
  index,
}: {
  stage: AgentStageItem;
  isLast: boolean;
  index: number;
}) {
  const isActive = stage.status === "running";
  const isDone = stage.status === "done";
  const isPending = stage.status === "pending";
  const StageGlyph = STAGE_ICONS[stage.id];

  const [liveMessage, setLiveMessage] = useState(() =>
    isActive ? pickStageRunningMessage(stage) : "",
  );

  useEffect(() => {
    if (!isActive) {
      setLiveMessage("");
      return;
    }
    setLiveMessage(pickStageRunningMessage(stage));
    const timer = setInterval(() => {
      setLiveMessage(pickStageRunningMessage(stage, Date.now()));
    }, 2800);
    return () => clearInterval(timer);
  }, [isActive, stage.id, stage.started_at, stage.status]);

  return (
    <motion.div
      initial={{ opacity: 0, x: -6 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.25, delay: index * 0.05, ease: "easeOut" }}
      className="flex gap-3"
    >
      <div className="flex flex-col items-center">
        <div
          className={`relative flex h-9 w-9 items-center justify-center rounded-xl border transition-all duration-300 ${
            isActive
              ? "border-primary/45 bg-primary/12 shadow-[0_0_20px_-6px] shadow-primary/40"
              : isDone
                ? "border-emerald-500/35 bg-emerald-500/10"
                : "border-border/40 bg-muted/15"
          }`}
        >
          {isActive ? (
            <span className="absolute inset-0 animate-ping rounded-xl bg-primary/10" />
          ) : null}
          <StageGlyph
            className={`relative h-4 w-4 ${
              isActive
                ? "text-primary"
                : isDone
                  ? "text-emerald-600 dark:text-emerald-400"
                  : "text-muted-foreground/35"
            }`}
          />
        </div>
        {!isLast ? (
          <div
            className={`mt-1.5 w-0.5 flex-1 min-h-[22px] rounded-full transition-colors ${
              isDone ? "bg-emerald-500/40" : isActive ? "bg-primary/25" : "bg-border/35"
            }`}
          />
        ) : null}
      </div>

      <div className={`min-w-0 flex-1 ${isLast ? "pb-0" : "pb-4"}`}>
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p
              className={`text-[13px] font-semibold leading-5 ${
                isActive
                  ? "text-foreground"
                  : isDone
                    ? "text-foreground/90"
                    : "text-muted-foreground/50"
              }`}
            >
              <span className="text-[11px] font-bold uppercase tracking-wide text-muted-foreground/45">
                {stage.id}
              </span>
              <span className="mx-1.5 text-muted-foreground/25">·</span>
              {stage.title}
              <span className="font-normal text-muted-foreground/60"> — </span>
              <span className={isActive ? "text-primary" : ""}>{stage.subtitle}</span>
            </p>
            <p
              className={`mt-1 text-[12px] leading-relaxed ${
                isActive
                  ? "text-muted-foreground"
                  : isDone
                    ? "text-muted-foreground/65"
                    : "text-muted-foreground/40"
              }`}
            >
              {isActive && liveMessage ? liveMessage : stage.hint}
            </p>
          </div>
          <div className="mt-0.5 shrink-0">
            <StageIcon status={stage.status} />
          </div>
        </div>

        {isPending ? (
          <p className="mt-1.5 text-[11px] text-muted-foreground/35">Скоро</p>
        ) : null}
      </div>
    </motion.div>
  );
}

type Props = {
  stages: AgentStageItem[];
};

export function AgentStageTimeline({ stages }: Props) {
  if (!stages.length) {
    return null;
  }

  const doneCount = stages.filter((stage) => stage.status === "done").length;
  const activeStage = stages.find((stage) => stage.status === "running");

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22, ease: "easeOut" }}
      className="overflow-hidden rounded-2xl border border-border/40 bg-gradient-to-br from-card/80 via-card/50 to-muted/20 px-4 py-4 backdrop-blur-sm"
    >
      <div className="mb-3 flex items-center justify-between gap-2 border-b border-border/25 pb-3">
        <p className="text-[11px] font-bold uppercase tracking-[0.14em] text-muted-foreground/55">
          Ход анализа
        </p>
        <p className="text-[11px] tabular-nums text-muted-foreground/45">
          {doneCount}/{stages.length}
        </p>
      </div>

      {activeStage ? (
        <p className="mb-3 text-[12px] leading-relaxed text-primary/85">
          Сейчас: {activeStage.title.toLowerCase()} — {activeStage.subtitle.toLowerCase()}
        </p>
      ) : null}

      {stages.map((stage, index) => (
        <StageRow
          key={stage.id}
          stage={stage}
          index={index}
          isLast={index === stages.length - 1}
        />
      ))}
    </motion.div>
  );
}
