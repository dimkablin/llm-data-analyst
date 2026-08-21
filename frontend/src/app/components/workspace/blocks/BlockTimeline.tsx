import React, { useMemo } from "react";
import { ChevronDown } from "lucide-react";
import { buildAgentStagesFromBlocks } from "../../../lib/agent-stages";
import type { AssistantBlock, ToolResultBlock as ToolResultBlockType } from "../../../lib/backend-types";
import {
  analysisPlanDisplayState,
  latestAnalysisPlan,
  type AnalysisPlanItem,
} from "../../../lib/analysis-plan";
import { AgentStageTimeline } from "../AgentStageTimeline";
import { IntentTextBlock } from "./TextBlock";
import { ThinkingBlock, LiveThinkingBlock } from "./ThinkingBlock";
import { ToolUseBlock } from "./ToolUseBlock";

function AnalysisPlanCard({ plan, isLive }: { plan: AnalysisPlanItem[]; isLive?: boolean }) {
  const completed = plan.filter((item) => item.status === "completed").length;
  const active = plan.find((item) => item.status === "in_progress");
  const displayState = analysisPlanDisplayState(plan, Boolean(isLive));
  const allDone = displayState === "completed";
  const dotClass =
    displayState === "completed"
      ? "bg-emerald-500"
      : displayState === "running"
        ? "animate-pulse bg-primary motion-reduce:animate-none"
        : "bg-amber-500";
  return (
    <details
      className="group rounded-xl border border-border/40 bg-card/70 shadow-sm"
      open={isLive && !allDone ? true : undefined}
      aria-live={isLive ? "polite" : undefined}
    >
      <summary className="flex cursor-pointer list-none items-center gap-3 px-3 py-2.5 select-none [&::-webkit-details-marker]:hidden">
        <span
          className={`h-2 w-2 shrink-0 rounded-full ${dotClass}`}
          aria-hidden="true"
        />
        <span className="min-w-0 flex-1">
          <span className="block text-[12px] font-medium text-foreground/85">План анализа</span>
          {displayState === "stopped" ? (
            <span className="block text-[11px] text-amber-600 dark:text-amber-400">Не завершён</span>
          ) : active ? (
            <span className="block truncate text-[11px] text-muted-foreground">{active.step}</span>
          ) : null}
        </span>
        <span className="text-[11px] tabular-nums text-muted-foreground">
          {completed}/{plan.length}
        </span>
        <ChevronDown className="h-3.5 w-3.5 text-muted-foreground transition-transform group-open:rotate-180" />
      </summary>
      <ul className="border-t border-border/30 px-3 py-2 text-[12px] leading-5">
        {plan.map((item) => (
          <li
            key={item.step}
            className={`flex gap-2 ${
              item.status === "completed" ? "text-muted-foreground/60" : "text-foreground/80"
            }`}
          >
            <span className="w-3 shrink-0 text-center" aria-hidden="true">
              {item.status === "completed" ? "✓" : item.status === "in_progress" ? "●" : "○"}
            </span>
            <span className={item.status === "completed" ? "line-through decoration-border" : ""}>
              {item.step}
            </span>
          </li>
        ))}
      </ul>
    </details>
  );
}

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
  const planState = useMemo(() => latestAnalysisPlan(blocks), [blocks]);

  const visibleBlocks = useMemo(
    () =>
      blocks.filter((block) => {
        if (block.type === "tool_use") return !planState.successfulToolUseIds.has(block.id);
        if (block.type === "tool_result") {
          return !planState.successfulToolUseIds.has(block.tool_use_id);
        }
        return true;
      }),
    [blocks, planState.successfulToolUseIds],
  );

  const simplifiedStages = useMemo(
    () =>
      showDetailedTools
        ? []
        : buildAgentStagesFromBlocks(visibleBlocks, { isLive, isSummarizing }),
    [visibleBlocks, isLive, isSummarizing, showDetailedTools],
  );

  const resultByToolUseId = useMemo(() => {
    const map = new Map<string, ToolResultBlockType>();
    for (const b of visibleBlocks) {
      if (b.type === "tool_result") {
        map.set(b.tool_use_id, b);
      }
    }
    return map;
  }, [visibleBlocks]);

  if (
    !showDetailedTools &&
    (planState.plan || simplifiedStages.length > 0 || (isLive && liveThinking))
  ) {
    return (
      <div className="flex flex-col gap-2 py-0.5">
        {planState.plan ? (
          <AnalysisPlanCard plan={planState.plan} isLive={isLive} />
        ) : null}
        {simplifiedStages.length ? <AgentStageTimeline stages={simplifiedStages} /> : null}
        {isLive && liveThinking ? <LiveThinkingBlock content={liveThinking} /> : null}
      </div>
    );
  }

  if (!visibleBlocks.length && !liveThinking && !planState.plan) return null;

  return (
    <div className="flex flex-col gap-0.5 py-0.5">
      {planState.plan ? (
        <AnalysisPlanCard plan={planState.plan} isLive={isLive} />
      ) : null}
      {visibleBlocks.map((block, idx) => {
        switch (block.type) {
          case "thinking": {
            // Если рассуждение идет перед вызовом инструмента, используем имя инструмента как подпись.
            // Otherwise derive the label from the block's semantic kind.
            const next = visibleBlocks[idx + 1];
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
            return <IntentTextBlock key={block.id} content={block.content} />;
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
