import React from "react";

import { MarkdownBlock } from "../../MarkdownBlock";

type Props = {
  content: string;
};

/**
 * Краткий текстовый блок для промежуточных сообщений агента перед инструментами и между шагами.
 * NOT used for the final answer — that still uses MarkdownBlock in the message bubble.
 */
export function IntentTextBlock({ content }: Props) {
  const trimmed = content.trim();
  if (!trimmed) return null;

  return (
    <MarkdownBlock
      content={trimmed}
      className="max-w-[75ch] px-2 py-2 text-[13px] leading-5 text-foreground/90 [&>p]:my-0"
    />
  );
}
