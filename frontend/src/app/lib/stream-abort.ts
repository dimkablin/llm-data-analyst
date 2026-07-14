import type {
  AssistantBlock,
  PhaseEvent,
  StreamToolCall,
  ToolResultBlock,
  ToolUseBlock,
} from "./backend-types";

export const INTERRUPTED_TOOL_OUTPUT = "Остановлено пользователем.";
export const INTERRUPTED_ASSISTANT_CONTENT =
  "_Генерация остановлена пользователем до появления итогового текста._";

type InterruptedAssistantInput = {
  text: string;
  reasoning: string | null;
  phases: PhaseEvent[];
  tools: StreamToolCall[];
  blocks: AssistantBlock[];
};

type InterruptedAssistantState = {
  shouldAppend: boolean;
  content: string;
  reasoning?: string | null;
  phases?: PhaseEvent[];
  tools?: StreamToolCall[];
  blocks?: AssistantBlock[];
};

function finalizeTool(call: StreamToolCall): StreamToolCall {
  if (call.status !== "running") return call;
  return {
    ...call,
    status: "error",
    output_preview: INTERRUPTED_TOOL_OUTPUT,
  };
}

function interruptedResultFor(block: ToolUseBlock): ToolResultBlock {
  return {
    type: "tool_result",
    id: `${block.id}-interrupted-result`,
    tool_use_id: block.id,
    tool_name: block.tool_name,
    status: "error",
    result_summary: INTERRUPTED_TOOL_OUTPUT,
    output_preview: INTERRUPTED_TOOL_OUTPUT,
    artifact_keys: block.artifact_keys,
  };
}

function finalizeBlocks(blocks: AssistantBlock[]): AssistantBlock[] {
  const existingResults = new Set(
    blocks
      .filter((block): block is ToolResultBlock => block.type === "tool_result")
      .map((block) => block.tool_use_id),
  );
  const finalized: AssistantBlock[] = [];
  for (const block of blocks) {
    if (block.type !== "tool_use" || block.status !== "running") {
      finalized.push(block);
      continue;
    }
    finalized.push({
      ...block,
      status: "error",
      result_summary: block.result_summary || INTERRUPTED_TOOL_OUTPUT,
      output_preview: block.output_preview || INTERRUPTED_TOOL_OUTPUT,
    });
    if (!existingResults.has(block.id)) {
      finalized.push(interruptedResultFor(block));
    }
  }
  return finalized;
}

export function finalizeInterruptedAssistantState({
  text,
  reasoning,
  phases,
  tools,
  blocks,
}: InterruptedAssistantInput): InterruptedAssistantState {
  const cleanText = text.trim();
  const cleanReasoning = reasoning?.trim() || null;
  const finalizedTools = tools.map(finalizeTool);
  const finalizedBlocks = finalizeBlocks(blocks);
  const shouldAppend = Boolean(
    cleanText ||
      cleanReasoning ||
      phases.length > 0 ||
      finalizedTools.length > 0 ||
      finalizedBlocks.length > 0,
  );

  return {
    shouldAppend,
    content: cleanText || INTERRUPTED_ASSISTANT_CONTENT,
    reasoning: cleanReasoning,
    phases: phases.length > 0 ? [...phases] : undefined,
    tools: finalizedTools.length > 0 ? finalizedTools : undefined,
    blocks: finalizedBlocks.length > 0 ? finalizedBlocks : undefined,
  };
}
