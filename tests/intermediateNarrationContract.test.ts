import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { filterBlocks } from "../frontend/src/app/lib/think-filter.ts";
import type { AssistantBlock } from "../frontend/src/app/lib/backend-types.ts";

test("intermediate narration uses shared Markdown without heading-style prompts", () => {
  const textBlock = readFileSync(
    "frontend/src/app/components/workspace/blocks/TextBlock.tsx",
    "utf8",
  );
  const prompt = readFileSync("backend/agent/prompts.py", "utf8");

  assert.match(textBlock, /import \{ MarkdownBlock \}/);
  assert.match(textBlock, /<MarkdownBlock[\s\S]*content=\{trimmed\}/);
  assert.doesNotMatch(textBlock, /function InlineMarkdown/);
  assert.match(prompt, /Не используй заголовки и подзаголовки/);
  assert.match(prompt, /Единственная Markdown-разметка здесь — одиночные обратные кавычки/);
});

test("pre_text remains visible when reasoning is hidden", () => {
  const blocks: AssistantBlock[] = [
    { type: "text", id: "text-1", content: "Проверяю `termination_date`." },
    { type: "thinking", id: "think-1", content: "Скрытое рассуждение." },
  ];

  const filtered = filterBlocks(blocks, {
    show_thinking: false,
    show_think_planning: false,
    show_think_tool: false,
    show_think_final: false,
  });

  assert.deepEqual(filtered, [blocks[0]]);
});
