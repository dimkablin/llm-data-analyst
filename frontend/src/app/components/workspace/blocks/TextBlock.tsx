import React from "react";

/** Встроенная разметка: рендерит `code` внутри обычного текста. */
function InlineMarkdown({ text }: { text: string }) {
  const parts = text.split(/`([^`]+)`/);
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <code
            key={i}
            className="rounded bg-muted/40 px-[3px] py-[1px] font-mono text-[11px] not-italic text-foreground/70"
          >
            {part}
          </code>
        ) : (
          part
        ),
      )}
    </>
  );
}

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
    <p className="px-2 py-0.5 text-[13px] leading-5 text-muted-foreground/70">
      <InlineMarkdown text={trimmed} />
    </p>
  );
}
