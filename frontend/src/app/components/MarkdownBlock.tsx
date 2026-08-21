import { Children, memo, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";

import { downloadReportFile } from "../lib/backend-api";
import { splitCheckedNumbers } from "../lib/anomaly-check";
import type { AnomalyCheck, ArtifactPayload } from "../lib/backend-types";
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip";

const REMARK_PLUGINS = [remarkGfm];

function isReportDownloadLink(href: unknown): href is string {
  if (typeof href !== "string") {
    return false;
  }

  return (
    href.includes("/reports/download/") ||
    href.startsWith("reports/download/")
  );
}

type NumberAnnotations = {
  check: AnomalyCheck;
  artifacts: ArtifactPayload[];
  onOpenArtifact?: (artifact: ArtifactPayload) => void;
};

function CheckedNumber({
  text,
  item,
  annotations,
}: {
  text: string;
  item: AnomalyCheck["items"][number];
  annotations: NumberAnnotations;
}) {
  const matched = item.status === "matched";
  const firstArtifact = item.sources
    .map((source) => annotations.artifacts.find((artifact) => artifact.id === source.artifact_id))
    .find((artifact) => artifact !== undefined);

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          tabIndex={0}
          onClick={() => firstArtifact && annotations.onOpenArtifact?.(firstArtifact)}
          className={`rounded-sm border-b border-dotted px-0.5 font-semibold outline-none transition-colors focus:ring-2 focus:ring-offset-1 ${
            matched && firstArtifact
              ? "cursor-pointer border-emerald-500 bg-emerald-500/10 text-emerald-700 focus:ring-emerald-500/40 dark:text-emerald-300"
              : matched
                ? "cursor-help border-emerald-500 bg-emerald-500/10 text-emerald-700 focus:ring-emerald-500/40 dark:text-emerald-300"
                : "cursor-help border-amber-500 bg-amber-500/10 text-amber-700 focus:ring-amber-500/40 dark:text-amber-300"
          }`}
        >
          {text}
        </span>
      </TooltipTrigger>
      <TooltipContent sideOffset={6} className="max-w-sm space-y-1.5 text-left">
        <div className="font-semibold">{matched ? "✓ Число подтверждено" : "⚠ Источник не найден"}</div>
        {item.sources.slice(0, 3).map((source, index) => (
          <div key={`${source.artifact_id}-${index}`} className="text-primary-foreground/80">
            {source.artifact_title}
            {source.row ? ` · строка: ${source.row}` : ""}
            {source.column ? ` · столбец: ${source.column}` : ""}
          </div>
        ))}
        {firstArtifact ? <div className="text-primary-foreground/70">Нажмите, чтобы открыть артефакт</div> : null}
      </TooltipContent>
    </Tooltip>
  );
}

function annotateChildren(children: ReactNode, annotations?: NumberAnnotations): ReactNode {
  if (!annotations) return children;
  return Children.map(children, (child) => {
    if (typeof child !== "string") return child;
    return splitCheckedNumbers(child, annotations.check.items).map((part, index) =>
      part.item ? (
        <CheckedNumber
          key={`${part.item.id}-${index}`}
          text={part.text}
          item={part.item}
          annotations={annotations}
        />
      ) : part.text,
    );
  });
}

function buildMarkdownComponents(inverted: boolean, annotations?: NumberAnnotations): Components {
  const textClass = inverted ? "text-inherit" : "text-foreground";
  const mutedTextClass = inverted ? "text-inherit/90" : "text-foreground/90";

  return {
  h2({ children, ...props }) {
    return (
      <h2 className={`mb-3 mt-1 text-base font-bold tracking-tight ${textClass}`} {...props}>
        {annotateChildren(children, annotations)}
      </h2>
    );
  },

  strong({ children, ...props }) {
    return (
      <strong className={`font-bold ${textClass}`} {...props}>
        {annotateChildren(children, annotations)}
      </strong>
    );
  },

  li({ children, ...props }) {
    return (
      <li className="my-1.5 leading-relaxed" {...props}>
        {annotateChildren(children, annotations)}
      </li>
    );
  },

  p({ children, ...props }) {
    return (
      <p className={`my-2 leading-relaxed ${mutedTextClass}`} {...props}>
        {annotateChildren(children, annotations)}
      </p>
    );
  },

  pre(props) {
    return (
      <div className="markdown-scroll-block">
        <pre {...props} />
      </div>
    );
  },

  table(props) {
    return (
      <div className="markdown-scroll-block">
        <table {...props} />
      </div>
    );
  },

  td({ children, ...props }) {
    return <td {...props}>{annotateChildren(children, annotations)}</td>;
  },

  th({ children, ...props }) {
    return <th {...props}>{annotateChildren(children, annotations)}</th>;
  },

  a({ href, children, ...props }) {
    if (isReportDownloadLink(href)) {
      return (
        <button
          type="button"
          className="font-semibold text-primary underline underline-offset-4 transition-colors hover:text-primary/80"
          onClick={async (event) => {
            event.preventDefault();

            try {
              await downloadReportFile(href);
            } catch (error) {
              console.error("Failed to download report", error);
              window.alert("Не удалось скачать отчет. Возможно, сессия авторизации истекла.");
            }
          }}
        >
          {children}
        </button>
      );
    }

    return (
      <a
        href={href}
        target="_blank"
        rel="noreferrer"
        {...props}
      >
        {children}
      </a>
    );
  },
};
}

type MarkdownBlockProps = {
  content: string;
  className?: string;
  /** Light text on dark/colored bubbles (user messages). */
  inverted?: boolean;
  anomalyCheck?: AnomalyCheck | null;
  artifacts?: ArtifactPayload[];
  onOpenArtifact?: (artifact: ArtifactPayload) => void;
};

export const MarkdownBlock = memo(function MarkdownBlock({
  content,
  className,
  inverted = false,
  anomalyCheck,
  artifacts = [],
  onOpenArtifact,
}: MarkdownBlockProps): JSX.Element {
  const rootClass = [
    "markdown-body",
    inverted ? "markdown-invert text-inherit" : "",
    className ?? "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={rootClass}>
      <ReactMarkdown
        remarkPlugins={REMARK_PLUGINS}
        components={buildMarkdownComponents(
          inverted,
          anomalyCheck ? { check: anomalyCheck, artifacts, onOpenArtifact } : undefined,
        )}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
});
