import React, { useMemo } from "react";
import type { AssistantBlock, ToolResultBlock as ToolResultBlockType } from "../../../lib/backend-types";
import { ThinkingBlock, LiveThinkingBlock } from "./ThinkingBlock";
import { ToolUseBlock } from "./ToolUseBlock";
import { IntentTextBlock } from "./TextBlock";

type Props = {
  blocks: AssistantBlock[];
  /** Live thinking text being streamed right now. */
  liveThinking?: string;
  /** Whether the stream is still active. */
  isLive?: boolean;
};

/**
 * Renders an ordered sequence of AssistantBlock items as a Claude-like activity timeline.
 * Each block type gets its own visual treatment.
 *
 * tool_result blocks are merged into their parent tool_use block for a clean display:
 * the tool row shows status, summary, and output — no separate result row.
 */
export function BlockTimeline({ blocks, liveThinking, isLive }: Props) {
  // Build a lookup: tool_use_id → tool_result block
  const resultByToolUseId = useMemo(() => {
    const map = new Map<string, ToolResultBlockType>();
    for (const b of blocks) {
      if (b.type === "tool_result") {
        map.set(b.tool_use_id, b);
      }
    }
    return map;
  }, [blocks]);

  if (!blocks.length && !liveThinking) return null;

  return (
    <div className="flex flex-col gap-0.5 py-0.5">
      {blocks.map((block) => {
        switch (block.type) {
          case "thinking":
            return (
              <ThinkingBlock
                key={block.id}
                content={block.content}
                defaultCollapsed
              />
            );
          case "text":
            return (
              <IntentTextBlock key={block.id} content={block.content} />
            );
          case "tool_use": {
            // Merge result data into the tool use block for unified display
            const result = resultByToolUseId.get(block.id);
            return (
              <ToolUseBlock
                key={block.id}
                tool_name={block.tool_name}
                input_summary={block.input_summary}
                input_code={block.input_code}
                input_preview={block.input_preview}
                status={block.status}
                started_at={block.started_at}
                result_summary={result?.result_summary}
                output_preview={result?.output_preview}
                artifact_keys={result?.artifact_keys}
              />
            );
          }
          case "tool_result":
            // Rendered inline with tool_use — skip standalone
            return null;
          default:
            return null;
        }
      })}

      {/* Live thinking indicator during streaming */}
      {isLive && liveThinking ? (
        <LiveThinkingBlock content={liveThinking} />
      ) : null}
    </div>
  );
}
