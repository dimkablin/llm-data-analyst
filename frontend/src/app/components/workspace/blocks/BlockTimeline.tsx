import React, { useMemo } from "react";
import { buildAgentStagesFromBlocks } from "../../../lib/agent-stages";
import type { AssistantBlock, ToolResultBlock as ToolResultBlockType } from "../../../lib/backend-types";
import { AgentStageTimeline } from "../AgentStageTimeline";
import { ThinkingBlock, LiveThinkingBlock } from "./ThinkingBlock";
import { ToolUseBlock } from "./ToolUseBlock";


type Props = {
  blocks: AssistantBlock[];
  /** Live thinking text being streamed right now. */
  liveThinking?: string;
  /** Активен ли поток сейчас. */
  isLive?: boolean;
  showDetailedTools?: boolean;
  isSummarizing?: boolean;
};

/**
 * Renders an ordered sequence of AssistantBlock items as a Claude-like activity timeline.
 * Each block type gets its own visual treatment.
 *
 * Блоки tool_result объединяются с родительским tool_use для компактного показа:
 * строка инструмента показывает статус, сводку и выход без отдельной строки результата.
 */
export function BlockTimeline({
  blocks,
  liveThinking,
  isLive,
  showDetailedTools = false,
  isSummarizing = false,
}: Props) {
  const simplifiedStages = useMemo(
    () =>
      showDetailedTools
        ? []
        : buildAgentStagesFromBlocks(blocks, { isLive, isSummarizing }),
    [blocks, isLive, isSummarizing, showDetailedTools],
  );

  const resultByToolUseId = useMemo(() => {
    const map = new Map<string, ToolResultBlockType>();
    for (const b of blocks) {
      if (b.type === "tool_result") {
        map.set(b.tool_use_id, b);
      }
    }
    return map;
  }, [blocks]);

  if (!showDetailedTools && (simplifiedStages.length > 0 || (isLive && liveThinking))) {
    return (
      <div className="flex flex-col gap-2 py-0.5">
        {simplifiedStages.length ? <AgentStageTimeline stages={simplifiedStages} /> : null}
        {isLive && liveThinking ? <LiveThinkingBlock content={liveThinking} /> : null}
      </div>
    );
  }

  if (!blocks.length && !liveThinking) return null;

  return (
    <div className="flex flex-col gap-0.5 py-0.5">
      {blocks.map((block, idx) => {
        switch (block.type) {
          case "thinking": {
            // Если рассуждение идет перед вызовом инструмента, используем имя инструмента как подпись.
            // Otherwise derive the label from the block's semantic kind.
            const next = blocks[idx + 1];
            let sourceLabel: string | undefined;
            if (next?.type === "tool_use") {
              sourceLabel = next.tool_name;
            } else {
              switch (block.kind) {
                case "planning":
                  sourceLabel = "планировщик";
                  break;
                case "final_synthesis":
                  sourceLabel = "агент";
                  break;
                default:
                  sourceLabel = "агент";
              }
            }
            return (
              <ThinkingBlock
                key={block.id}
                content={block.content}
                defaultCollapsed
                sourceLabel={sourceLabel}
              />
            );
          }
          case "text":
            return null;
          case "tool_use": {
            // Объединяем данные результата с блоком вызова инструмента для единого отображения.
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
            // Отображается внутри tool_use, отдельно не рендерится.
            return null;
          default:
            return null;
        }
      })}

      {/* Индикатор живого рассуждения во время потока. */}
      {isLive && liveThinking ? (
        <LiveThinkingBlock content={liveThinking} />
      ) : null}
    </div>
  );
}
