import { memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const REMARK_PLUGINS = [remarkGfm];

type MarkdownBlockProps = {
  content: string;
};

export const MarkdownBlock = memo(function MarkdownBlock({ content }: MarkdownBlockProps): JSX.Element {
  return (
    <div className="markdown-body">
      <ReactMarkdown remarkPlugins={REMARK_PLUGINS}>{content}</ReactMarkdown>
    </div>
  );
});
