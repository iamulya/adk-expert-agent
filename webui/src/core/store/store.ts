import { nanoid } from "nanoid";
import { toast } from "sonner";
import { create } from "zustand";
import { useShallow } from "zustand/react/shallow";
import { chatStream } from "../api/chat";
import type { Message } from "../messages";
import { mergeMessage } from "../messages";

const SESSION_ID = nanoid(); 

export const useStore = create<{
  responding: boolean;
  sessionId: string;
  messageIds: string[];
  messages: Map<string, Message>;
  appendMessage: (message: Message) => void;
  updateMessage: (message: Message) => void;
}>((set, get) => ({
  responding: false,
  sessionId: SESSION_ID,
  messageIds: [],
  messages: new Map<string, Message>(),

  appendMessage(message: Message) {
    set((state) => ({
      messageIds: [...state.messageIds, message.id],
      messages: new Map(state.messages).set(message.id, message),
    }));
  },
  updateMessage(message: Message) {
    set((state) => ({
      messages: new Map(state.messages).set(message.id, message),
    }));
  },
}));

export async function sendMessage(
  content: string,
  params: Record<string, unknown> = {},
  options: { abortSignal?: AbortSignal } = {},
) {
  const userMessageId = nanoid();
  useStore.getState().appendMessage({
    id: userMessageId,
    threadId: useStore.getState().sessionId,
    role: "user",
    content: content,
    contentChunks: [content],
    agent: "user",
  });

  useStore.setState({ responding: true });
  const assistantMessageId = nanoid();
  let assistantMessage: Message = {
    id: assistantMessageId,
    threadId: useStore.getState().sessionId,
    role: "assistant",
    agent: "adk_expert_agent",
    content: "",
    contentChunks: [],
    isStreaming: true,
  };
  useStore.getState().appendMessage(assistantMessage);

  try {
    const stream = chatStream(
      content,
      { session_id: useStore.getState().sessionId, ...params },
      options,
    );

    for await (const event of stream) {
      const currentAssistantMessage = useStore.getState().messages.get(assistantMessageId);
      if (currentAssistantMessage) {
        assistantMessage = mergeMessage(currentAssistantMessage, event);
        useStore.getState().updateMessage(assistantMessage);
      }
    }
  } catch (error) {
    console.error("Error during chat stream:", error);
    toast.error(`An error occurred: ${(error as Error).message}`);
    const finalAssistantMessage = useStore.getState().messages.get(assistantMessageId);
    if (finalAssistantMessage?.isStreaming) {
      finalAssistantMessage.isStreaming = false;
      finalAssistantMessage.finishReason = "error";
      useStore.getState().updateMessage(finalAssistantMessage);
    }
  } finally {
    useStore.setState({ responding: false });
  }
}

export function useMessage(messageId: string | null | undefined) {
  return useStore(
    useShallow((state) => (messageId ? state.messages.get(messageId) : undefined)),
  );
}

export function useMessageIds() {
  return useStore(useShallow((state) => state.messageIds));
}
