import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { useChatAgent } from "../hooks/useChatAgent";

type BindChatAgentArgs = {
  sessionId: string;
  includeReasoning: boolean;
  useHistory: boolean;
  analysisDepth?: string;
};

type ChatAgentContextValue = ReturnType<typeof useChatAgent> & {
  boundSessionId: string;
  bindChatAgent: (args: BindChatAgentArgs) => void;
};

const ChatAgentContext = createContext<ChatAgentContextValue | null>(null);

const DEFAULT_ARGS: BindChatAgentArgs = {
  sessionId: "",
  includeReasoning: true,
  useHistory: true,
  analysisDepth: "light",
};

export function ChatAgentProvider({ children }: { children: ReactNode }) {
  const [args, setArgs] = useState<BindChatAgentArgs>(DEFAULT_ARGS);
  const agent = useChatAgent(args);

  const bindChatAgent = useCallback((nextArgs: BindChatAgentArgs) => {
    setArgs((prev) => {
      if (
        prev.sessionId === nextArgs.sessionId &&
        prev.includeReasoning === nextArgs.includeReasoning &&
        prev.useHistory === nextArgs.useHistory &&
        prev.analysisDepth === nextArgs.analysisDepth
      ) {
        return prev;
      }
      return nextArgs;
    });
  }, []);

  const value = useMemo<ChatAgentContextValue>(
    () => ({
      ...agent,
      boundSessionId: args.sessionId,
      bindChatAgent,
    }),
    [agent, args.sessionId, bindChatAgent],
  );

  return (
    <ChatAgentContext.Provider value={value}>
      {children}
    </ChatAgentContext.Provider>
  );
}

export function useChatAgentContext(): ChatAgentContextValue {
  const value = useContext(ChatAgentContext);
  if (!value) {
    throw new Error("useChatAgentContext must be used within ChatAgentProvider");
  }
  return value;
}
