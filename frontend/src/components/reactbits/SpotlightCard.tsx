import { useRef } from "react";
import type { MouseEventHandler, ReactNode } from "react";

type SpotlightCardProps = {
  children: ReactNode;
  className?: string;
  spotlightColor?: string;
};

export function SpotlightCard({
  children,
  className = "",
  spotlightColor = "rgba(76, 113, 255, 0.35)"
}: SpotlightCardProps): JSX.Element {
  const cardRef = useRef<HTMLDivElement | null>(null);

  const handleMouseMove: MouseEventHandler<HTMLDivElement> = (event) => {
    const card = cardRef.current;
    if (!card) {
      return;
    }

    const rect = card.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    card.style.setProperty("--mouse-x", `${x}px`);
    card.style.setProperty("--mouse-y", `${y}px`);
    card.style.setProperty("--spotlight-color", spotlightColor);
  };

  return (
    <div ref={cardRef} onMouseMove={handleMouseMove} className={`rb-spotlight-card ${className}`.trim()}>
      {children}
    </div>
  );
}
