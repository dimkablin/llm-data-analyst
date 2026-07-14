import { memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";

import { downloadReportFile } from "../lib/backend-api";

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

function buildMarkdownComponents(inverted: boolean): Components {
  const textClass = inverted ? "text-inherit" : "text-foreground";
  const mutedTextClass = inverted ? "text-inherit/90" : "text-foreground/90";

  return {
  h2({ children, ...props }) {
    return (
      <h2 className={`mb-3 mt-1 text-base font-bold tracking-tight ${textClass}`} {...props}>
        {children}
      </h2>
    );
  },

  strong({ children, ...props }) {
    return (
      <strong className={`font-bold ${textClass}`} {...props}>
        {children}
      </strong>
    );
  },

  li({ children, ...props }) {
    return (
      <li className="my-1.5 leading-relaxed" {...props}>
        {children}
      </li>
    );
  },

  p({ children, ...props }) {
    return (
      <p className={`my-2 leading-relaxed ${mutedTextClass}`} {...props}>
        {children}
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
};

export const MarkdownBlock = memo(function MarkdownBlock({
  content,
  className,
  inverted = false,
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
        components={buildMarkdownComponents(inverted)}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
});
