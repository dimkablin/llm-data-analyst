import type { ComponentPropsWithoutRef, CSSProperties, ElementType, ReactNode } from "react";

type StarBorderProps<T extends ElementType> = {
  as?: T;
  children: ReactNode;
  className?: string;
  color?: string;
  speed?: string;
  thickness?: number;
  style?: CSSProperties;
} & Omit<ComponentPropsWithoutRef<T>, "as" | "children" | "className" | "style">;

export function StarBorder<T extends ElementType = "button">({
  as,
  children,
  className = "",
  color = "#6d8dff",
  speed = "6s",
  thickness = 1,
  style,
  ...rest
}: StarBorderProps<T>): JSX.Element {
  const Component = (as ?? "button") as ElementType;

  return (
    <Component
      className={`rb-star-border ${className}`.trim()}
      style={{
        padding: `${thickness}px 0`,
        ...style
      }}
      {...rest}
    >
      <span
        className="rb-star-border-bottom"
        style={{
          background: `radial-gradient(circle, ${color}, transparent 10%)`,
          animationDuration: speed
        }}
      />
      <span
        className="rb-star-border-top"
        style={{
          background: `radial-gradient(circle, ${color}, transparent 10%)`,
          animationDuration: speed
        }}
      />
      <span className="rb-star-border-inner">{children}</span>
    </Component>
  );
}
