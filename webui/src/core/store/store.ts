import { nanoid } from "nanoid";
import { toast } from "sonner";
import { create } from "zustand";
import { useShallow } from "zustand/react/shallow";
import { 
  createAdkSession,        
  postToRunSseAndStream 
} from "../api/chat";
import type { AdkSession, ChatEvent } from "../api/types"; 
import type { Message } from "../messages";
import { mergeMessage } from "../messages";

interface AppState {
  responding: boolean;
  adkSession: AdkSession | null; 
  messageIds: string[];
  messages: Map<string, Message>;
  currentAbortController: AbortController | null;
  
  setAdkSession: (session: AdkSession | null) => void;
  appendMessage: (message: Message) => void;
  updateMessage: (message: Message) => void;
  setAbortController: (ac: AbortController | null) => void;
  clearChatAndCreateNewSession: () => Promise<void>; 
  initializeSessionIfNeeded: () => Promise<AdkSession | null>;
}

export const useStore = create<AppState>((set, get) => ({
  responding: false,
  adkSession: null, 
  messageIds: [],
  messages: new Map<string, Message>(),
  currentAbortController: null,

  setAdkSession: (session) => {
     console.log("[STORE] setAdkSession called. Session:", session);
     set({ adkSession: session, currentAbortController: null });
  },
  appendMessage(message: Message) {
    console.log("[STORE] appendMessage called. Message ID:", message.id, "Role:", message.role);
    set((state) => ({
      messageIds: [...state.messageIds, message.id],
      messages: new Map(state.messages).set(message.id, message),
    }));
  },
  updateMessage(message: Message) {
    // console.log("[STORE] updateMessage called. Message ID:", message.id, "New content length:", message.content.length, "Is Streaming:", message.isStreaming, "Finish Reason:", message.finishReason);
    set((state) => ({
      messages: new Map(state.messages).set(message.id, message),
    }));
  },
  setAbortController(ac: AbortController | null) {
    console.log("[STORE] setAbortController called. New AC:", !!ac);
    const oldAc = get().currentAbortController;
    if (oldAc && oldAc !== ac) {
      console.log("[STORE] Aborting previous AbortController.");
      oldAc.abort("New request started or session cleared");
    }
    set({ currentAbortController: ac });
  },
  async clearChatAndCreateNewSession() {
    console.log("[STORE] clearChatAndCreateNewSession called.");
    const { currentAbortController, setAbortController, setAdkSession } = get();
    if (currentAbortController) {
      currentAbortController.abort("Chat cleared by user");
      setAbortController(null);
    }
    set({ messageIds: [], messages: new Map(), responding: false, adkSession: null }); 
    try {
      console.log("[STORE] Attempting to create new ADK session after clearing chat...");
      const newSession = await createAdkSession(); 
      setAdkSession(newSession);
      console.log("[STORE] New ADK session created and set:", newSession);
    } catch (error) {
        console.error("[STORE] Failed to create new ADK session after clearing chat:", error);
        toast.error("Could not start a new chat session with the agent.");
    }
  },
  async initializeSessionIfNeeded() {
     let session = get().adkSession;
     if (!session || !session.session_id) {
         console.log("[STORE] initializeSessionIfNeeded: No valid ADK session found, creating one...");
         try {
             session = await createAdkSession();
             get().setAdkSession(session);
         } catch (error) {
             console.error("[STORE] initializeSessionIfNeeded: Failed to initialize ADK session:", error);
             toast.error(`Error connecting to agent: ${(error as Error).message}`);
             return null; 
         }
     } else {
        // console.log("[STORE] initializeSessionIfNeeded: Existing ADK session found:", session);
     }
     return session;
  }
}));
   
export async function sendMessage(
  userInput: string,
  _params: Record<string, unknown> = {},
) {
  console.log("[STORE] sendMessage called. User input:", userInput);
  console.log("[STORE] Current responding state (before set):", useStore.getState().responding);
  useStore.setState({ responding: true });
  console.log("[STORE] Responding state set to true.");

  const currentAdkSession = await useStore.getState().initializeSessionIfNeeded();
  
  if (!currentAdkSession || !currentAdkSession.session_id) { 
    console.error("[STORE] sendMessage: Could not ensure ADK session. Aborting send.");
    useStore.setState({ responding: false }); 
    return; 
  }
  
  const adkSessionIdToUse = currentAdkSession.session_id;
  console.log("[STORE] sendMessage: Using ADK Session ID:", adkSessionIdToUse);

  const userMessageId = nanoid(); 
  useStore.getState().appendMessage({
    id: userMessageId,
    threadId: adkSessionIdToUse, 
    role: "user",
    content: userInput,
    contentChunks: [userInput],
    agent: "user",
  });

  const assistantMessageId = nanoid(); 
  let assistantMessage: Message = {
    id: assistantMessageId,
    threadId: adkSessionIdToUse,
    role: "assistant",
    agent: process.env.NEXT_PUBLIC_ADK_APP_NAME || 'expert-agents',
    content: "",
    contentChunks: [],
    isStreaming: true,
  };
  useStore.getState().appendMessage(assistantMessage);
  console.log("[STORE] Appended placeholder assistant message. ID:", assistantMessageId);

  const abortController = new AbortController();
  useStore.getState().setAbortController(abortController);

  try {
    console.log("[STORE] Calling postToRootRunSseAndStream for session:", adkSessionIdToUse);
    const stream = postToRunSseAndStream(adkSessionIdToUse, userInput, { abortSignal: abortController.signal });

    for await (const event of stream) {
      if (abortController.signal.aborted) {
          console.log("[STORE] Stream processing aborted in sendMessage loop.");
          break; 
      }
      const eventForUIMerge: ChatEvent = {
        ...event,
        data: { ...event.data, id: assistantMessageId }
      };
      // console.log("[STORE] Received event from stream for UI merge:", JSON.stringify(eventForUIMerge, null, 2).substring(0, 300) + "...");

      const currentAssistantMsg = useStore.getState().messages.get(assistantMessageId);
      if (currentAssistantMsg) {
        const updatedMsg = mergeMessage(currentAssistantMsg, eventForUIMerge);
        useStore.getState().updateMessage(updatedMsg);
      } else {
        console.warn("[STORE] Could not find assistant message to update. ID:", assistantMessageId);
      }
      if (eventForUIMerge.data.finish_reason === "stop") {
        console.log("[STORE] Finish reason 'stop' received in event from stream, breaking loop for message ID:", assistantMessageId);
        break;
      }
    }
    console.log("[STORE] Finished iterating stream for message ID:", assistantMessageId);

  } catch (error) {
    if (abortController.signal.aborted && (error as Error).name === 'AbortError') {
        console.log("[STORE] Fetch stream for session-specific /run_sse aborted by AbortController in sendMessage.");
    } else {
        console.error("[STORE] Error in sendMessage (streaming from session-specific /run_sse):", error);
        toast.error(`Message processing error: ${(error as Error).message}`);
        const finalAssistantMsg = useStore.getState().messages.get(assistantMessageId);
        if (finalAssistantMsg) { 
            finalAssistantMsg.isStreaming = false;
            finalAssistantMsg.finishReason = "error";
            useStore.getState().updateMessage(finalAssistantMsg);
        }
    }
  } finally {
    console.log("[STORE] sendMessage finally block. Message ID:", assistantMessageId, "Aborted:", abortController.signal.aborted, "Current AC in store matches:", useStore.getState().currentAbortController === abortController);
    if (!abortController.signal.aborted || useStore.getState().currentAbortController === abortController) {
        useStore.setState({ responding: false, currentAbortController: null });
        console.log("[STORE] Set responding to false in finally block.");
    }
  }
}

export function cancelStream() {
    console.log("[STORE] cancelStream called.");
    const ac = useStore.getState().currentAbortController;
    if (ac) {
        console.log("[STORE] Aborting current AbortController.");
        ac.abort("User cancelled request from UI"); 
    }
    if (useStore.getState().responding) {
        console.log("[STORE] cancelStream: Responding was true, setting to false.");
        useStore.setState({ responding: false });
        const { messageIds, messages, updateMessage, adkSession } = useStore.getState();
         if (messageIds.length > 0 && adkSession) { 
            const currentSessionId = adkSession.session_id;
            for (let i = messageIds.length - 1; i >= 0; i--) {
                const msgId = messageIds[i];
                const msg = messages.get(msgId!);
                if (msg && msg.threadId === currentSessionId && msg.role === 'assistant' && msg.isStreaming) {
                    console.log("[STORE] cancelStream: Updating last assistant message to not streaming. ID:", msg.id);
                    const updatedMessage = { ...msg, isStreaming: false, finishReason: "stop" as const };
                    updateMessage(updatedMessage);
                    break; 
                }
            }
        }
    } else {
        console.log("[STORE] cancelStream: Responding was already false.");
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
   
export function useAdkSessionDetails() { 
    return useStore(useShallow((state) => state.adkSession));
}