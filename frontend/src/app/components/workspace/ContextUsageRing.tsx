import type { ContextUsageSnapshot } from "../../lib/backend-types";
import {
  buildContextUsageTooltipDetails,
  formatContextUsageLabel,
  getContextUsagePercent,
} from "../../lib/context-usage";

type ContextUsageRingProps = {
  usage: ContextUsageSnapshot | null;
  isLoading: boolean;
};

export function ContextUsageRing({ usage, isLoading }: ContextUsageRingProps) {
  const percent = getContextUsagePercent(usage);
  const details = buildContextUsageTooltipDetails(usage, isLoading);
  const label = isLoading ? details.percentLine : formatContextUsageLabel(usage);
  const radius = 7;
  const circumference = 2 * Math.PI * radius;
  const progressOffset = circumference - (circumference * percent) / 100;

  return (
    <div className="group/context relative inline-flex">
      <div
        className="grid h-7 w-7 place-items-center rounded-full text-primary transition-colors hover:text-primary/90 focus-visible:outline focus-visible:outline-1 focus-visible:outline-primary/50"
        aria-label={label}
        role="status"
        tabIndex={0}
      >
        <svg
          className={`h-5 w-5 -rotate-90 overflow-visible ${isLoading ? "animate-pulse" : ""}`}
          viewBox="0 0 20 20"
          aria-hidden="true"
        >
          <circle
            cx="10"
            cy="10"
            r={radius}
            fill="none"
            className="stroke-muted-foreground/20"
            strokeWidth="2.5"
          />
          <circle
            cx="10"
            cy="10"
            r={radius}
            fill="none"
            className="stroke-primary"
            strokeLinecap="round"
            strokeWidth="2.5"
            strokeDasharray={circumference}
            strokeDashoffset={progressOffset}
          />
        </svg>
      </div>
      <div className="pointer-events-none absolute bottom-full right-0 z-50 mb-2 w-[210px] rounded-2xl border border-border/70 bg-popover px-4 py-3 text-center text-[11px] leading-5 text-popover-foreground opacity-0 shadow-md transition-opacity duration-150 group-hover/context:opacity-100 group-focus-within/context:opacity-100">
        <div className="font-semibold text-foreground">{details.title}</div>
        <div>{details.percentLine}</div>
        <div>{details.usedLine}</div>
      </div>
    </div>
  );
}
