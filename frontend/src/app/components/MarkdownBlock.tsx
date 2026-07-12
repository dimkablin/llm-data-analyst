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

const MARKDOWN_COMPONENTS: Components = {
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

type MarkdownBlockProps = {
  content: string;
  className?: string;
};

export const MarkdownBlock = memo(function MarkdownBlock({
  content,
  className,
}: MarkdownBlockProps): JSX.Element {
  return (
    <div className={className ? `markdown-body ${className}` : "markdown-body"}>
      <ReactMarkdown remarkPlugins={REMARK_PLUGINS} components={MARKDOWN_COMPONENTS}>
        {content}
      </ReactMarkdown>
    </div>
  );
});
