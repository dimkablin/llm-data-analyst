import { ReactNode, useEffect, useRef, useState } from "react";

type RevealOnScrollProps = {
  className?: string;
  children: ReactNode;
  delayMs?: number;
};

export function RevealOnScroll({ className = "", children, delayMs = 0 }: RevealOnScrollProps): JSX.Element {
  const [visible, setVisible] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) {
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setVisible(true);
            observer.disconnect();
          }
        });
      },
      {
        threshold: 0.16,
        rootMargin: "0px 0px -8% 0px"
      }
    );
    observer.observe(node);
    return () => {
      observer.disconnect();
    };
  }, []);

  return (
    <div
      ref={ref}
      className={`${className} reveal-up ${visible ? "is-visible" : ""}`.trim()}
      style={{ transitionDelay: `${delayMs}ms` }}
    >
      {children}
    </div>
  );
}
