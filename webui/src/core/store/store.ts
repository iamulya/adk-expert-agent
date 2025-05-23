import { nanoid } from "nanoid";
import { toast } from "sonner";
import { create } from "zustand";
import { useShallow } from "zustand/react/shallow";
import { sendAdkMessage, createAdkEventSource } from "../api/chat"; // Updated import
import type { ChatEvent } from "../api/types"; // Keep ChatEvent for UI consistency
import type { Message } from "../messages";
import { mergeMessage } from "../messages";

const SESSION_ID = nanoid(); 

export const useStore = create<{
  responding: boolean;
  sessionId: string;
  messageIds: string[];
  messages: Map<string, Message>;
  currentEventSource: EventSource | null; // To manage the EventSource instance
  appendMessage: (message: Message) => void;
  updateMessage: (message: Message) => void;
  setEventSource: (es: EventSource | null) => void;
}>((set, get) => ({
  responding: false,
  sessionId: SESSION_ID,
  messageIds: [],
  messages: new Map<string, Message>(),
  currentEventSource: null,

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
  setEventSource(es: EventSource | null) {
    set({ currentEventSource: es });
  }
}));

export async function sendMessage(
  userInput: string,
  params: Record<string, unknown> = {}, // Kept for potential future use
  options: { abortSignal?: AbortSignal } = {}, // AbortSignal for the initial POST
) {
  const currentSessionId = useStore.getState().sessionId;

  // 1. Append User Message to UI
  const userMessageId = nanoid();
  useStore.getState().appendMessage({
    id: userMessageId,
    threadId: currentSessionId,
    role: "user",
    content: userInput,
    contentChunks: [userInput],
    agent: "user",
  });

  useStore.setState({ responding: true });

  // 2. Create a placeholder for Assistant's message
  const assistantMessageId = nanoid();
  let assistantMessage: Message = {
    id: assistantMessageId,
    threadId: currentSessionId,
    role: "assistant",
    agent: "adk_expert_agent",
    content: "",
    contentChunks: [],
    isStreaming: true,
  };
  useStore.getState().appendMessage(assistantMessage);

  try {
    // 3. Send the message to ADK via POST
    const postResponse = await sendAdkMessage(userInput, currentSessionId, options);
    if (!postResponse.ok) {
      const errorText = await postResponse.text();
      throw new Error(`Failed to send message to ADK: ${postResponse.status} ${postResponse.statusText} - ${errorText}`);
    }
    // console.log("ADK POST Response:", await postResponse.json()); // Or .text() if not JSON

    // 4. Setup EventSource to receive stream
    const eventSource = createAdkEventSource(
      currentSessionId,
      (chunkData) => { // onMessage
        const currentAssistantMsg = useStore.getState().messages.get(assistantMessageId);
        if (currentAssistantMsg) {
          // Adapt chunkData to ChatEvent structure for mergeMessage
          const chatEvent: ChatEvent = {
            type: "message_chunk",
            data: {
              id: assistantMessageId, // id of the message being updated
              thread_id: currentSessionId,
              role: "assistant",
              agent: "adk_expert_agent",
              content: chunkData.text,
              finish_reason: chunkData.done ? "stop" : undefined,
            },
          };
          const updatedMsg = mergeMessage(currentAssistantMsg, chatEvent);
          useStore.getState().updateMessage(updatedMsg);
        }
      },
      (error) => { // onError
        console.error("SSE Error in store:", error);
        toast.error(`Streaming error: ${(error as Error).message || "Unknown SSE error"}`);
        const finalMsg = useStore.getState().messages.get(assistantMessageId);
        if (finalMsg?.isStreaming) {
          finalMsg.isStreaming = false;
          finalMsg.finishReason = "error";
          useStore.getState().updateMessage(finalMsg);
        }
        useStore.setState({ responding: false, currentEventSource: null });
      },
      () => { // onOpen
         useStore.getState().setEventSource(eventSource);
      },
      () => { // onClose (called by server 'close' event or EventSource.close())
        const finalMsg = useStore.getState().messages.get(assistantMessageId);
        if (finalMsg?.isStreaming) { // Ensure it's marked as not streaming
          finalMsg.isStreaming = false;
          if (!finalMsg.finishReason) finalMsg.finishReason = "stop"; // If not already set by 'done'
          useStore.getState().updateMessage(finalMsg);
        }
        useStore.setState({ responding: false, currentEventSource: null });
      }
    );
    if (eventSource) { // Only set if not mock
        useStore.getState().setEventSource(eventSource);
    }


  } catch (error) {
    console.error("Error in sendMessage:", error);
    toast.error(`Failed to send message: ${(error as Error).message}`);
    const finalAssistantMsg = useStore.getState().messages.get(assistantMessageId);
    if (finalAssistantMsg) { // Check if it exists before trying to update
        finalAssistantMsg.isStreaming = false;
        finalAssistantMsg.finishReason = "error";
        useStore.getState().updateMessage(finalAssistantMsg);
    }
    useStore.setState({ responding: false, currentEventSource: null });
  }
  // Note: `responding` is set to false by the EventSource onClose/onError handlers now.
}


// Function to cancel the stream
export function cancelStream() {
    const es = useStore.getState().currentEventSource;
    if (es) {
        es.close(); // This will trigger the onclose handler in createAdkEventSource
        console.log("Manually closed EventSource.");
    }
    // Update responding state if it wasn't updated by onclose
    if (useStore.getState().responding) {
        useStore.setState({ responding: false });
        // Also update the last assistant message to not be streaming
        const { messageIds, messages, updateMessage } = useStore.getState();
        if (messageIds.length > 0) {
            const lastMessageId = messageIds[messageIds.length - 1];
            const lastMessage = messages.get(lastMessageId!);
            if (lastMessage && lastMessage.role === 'assistant' && lastMessage.isStreaming) {
                const updatedMessage = { ...lastMessage, isStreaming: false, finishReason: "stop" as const };
                updateMessage(updatedMessage);
            }
        }
    }
}


// Hooks for easy component consumption
export function useMessage(messageId: string | null | undefined) {
  return useStore(
    useShallow((state) => (messageId ? state.messages.get(messageId) : undefined)),
  );
}

export function useMessageIds() {
  return useStore(useShallow((state) => state.messageIds));
}