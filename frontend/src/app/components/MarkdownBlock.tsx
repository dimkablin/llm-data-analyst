import { memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";

const REMARK_PLUGINS = [remarkGfm];

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
