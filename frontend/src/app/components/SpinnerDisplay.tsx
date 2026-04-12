import { useEffect, useState } from "react";
import {
  getStoredSpinner,
  SPINNER_CHANGED_EVENT,
  type SpinnerId,
} from "../lib/spinner";

function useSpinnerId(): SpinnerId {
  const [id, setId] = useState<SpinnerId>(() => getStoredSpinner());
  useEffect(() => {
    const handler = () => setId(getStoredSpinner());
    window.addEventListener(SPINNER_CHANGED_EVENT, handler);
    return () => window.removeEventListener(SPINNER_CHANGED_EVENT, handler);
  }, []);
  return id;
}

// ── Individual spinner variants ──────────────────────────────────────────────

function RingSpinner({ size }: { size: "dot" | "icon" }) {
  if (size === "dot") {
    return (
      <span className="inline-flex h-3 w-3 animate-spin rounded-full border-[1.5px] border-primary/25 border-t-primary" />
    );
  }
  return (
    <span className="inline-flex h-4 w-4 animate-spin rounded-full border-2 border-primary/20 border-t-primary" />
  );
}

function DotsSpinner({ size }: { size: "dot" | "icon" }) {
  const dot = size === "dot" ? "h-1 w-1" : "h-1.5 w-1.5";
  return (
    <span className="inline-flex items-end gap-0.5">
      {([0, 160, 320] as const).map((delay) => (
        <span
          key={delay}
          className={`${dot} animate-bounce rounded-full bg-primary`}
          style={{ animationDelay: `${delay}ms` }}
        />
      ))}
    </span>
  );
}

function BarsSpinner({ size }: { size: "dot" | "icon" }) {
  const totalH = size === "dot" ? 12 : 16;
  return (
    <span
      className="inline-flex items-end gap-px"
      style={{ height: totalH }}
    >
      {([0, 140, 280] as const).map((delay) => (
        <span
          key={delay}
          className="animate-bar-eq rounded-sm bg-primary"
          style={{ width: 3, height: totalH, animationDelay: `${delay}ms` }}
        />
      ))}
    </span>
  );
}

function PulseSpinner({ size }: { size: "dot" | "icon" }) {
  const s = size === "dot" ? "h-2.5 w-2.5" : "h-3.5 w-3.5";
  return (
    <span className="relative inline-flex">
      <span
        className={`${s} absolute inline-flex animate-ping rounded-full bg-primary opacity-60`}
      />
      <span className={`${s} relative inline-flex rounded-full bg-primary`} />
    </span>
  );
}

function ScanSpinner({ size }: { size: "dot" | "icon" }) {
  const w = size === "dot" ? "w-5" : "w-6";
  return (
    <span
      className={`relative inline-flex ${w} overflow-hidden rounded-full bg-primary/15`}
      style={{ height: "2px" }}
    >
      <span className="animate-scan-line absolute inset-y-0 w-1/3 rounded-full bg-primary" />
    </span>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

function SpinnerById({
  id,
  size,
}: {
  id: SpinnerId;
  size: "dot" | "icon";
}) {
  switch (id) {
    case "ring":  return <RingSpinner size={size} />;
    case "dots":  return <DotsSpinner size={size} />;
    case "bars":  return <BarsSpinner size={size} />;
    case "pulse": return <PulseSpinner size={size} />;
    case "scan":  return <ScanSpinner size={size} />;
  }
}

/**
 * Renders the user-selected animated spinner.
 * - size="icon"  — fits in a 28×28 avatar box
 * - size="dot"   — fits inline in monospace tool rows
 * Pass `id` to force a specific variant (e.g. for previews in settings).
 */
export function SpinnerDisplay({
  id: propId,
  size = "icon",
}: {
  id?: SpinnerId;
  size?: "dot" | "icon";
}) {
  const storedId = useSpinnerId();
  return <SpinnerById id={propId ?? storedId} size={size} />;
}
