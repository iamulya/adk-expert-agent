// webui/src/app/chat/main.tsx
"use client";
import { useCallback, useEffect, useRef } from "react";
import { ConversationStarter } from "./components/conversation-starter";
import { InputBox } from "./components/input-box";
import { MessageListView } from "./components/message-list-view";
import { Welcome } from "./components/welcome"; // Import the Welcome component
import { useMessageIds, useStore, sendMessage as sendApiMessage } from "~/core/store/store";
// import { toast } from "sonner"; // Not needed here if errors handled in store

export default function Main() {
  const messageIds = useMessageIds();
  const responding = useStore((state) => state.responding);
 
  useEffect(() => {
    useStore.getState().initializeSessionIfNeeded().catch(error => {
        console.error("Main.tsx: Failed to initialize session on mount:", error);
    });
  }, []);


  const handleSend = useCallback(
    async (message: string) => {
      try {
        await sendApiMessage(message);
      } catch (e) {
        console.error("Send message error in Main.tsx (from sendApiMessage):", e);
      }
    },
    [],
  );

  const showWelcome = !responding && messageIds.length === 0;

  return (
    <div className="flex h-full w-full max-w-2xl flex-col justify-center px-4 pt-16 pb-4">
      <MessageListView className="flex-grow" />
      <div className="relative mt-4 flex h-auto shrink-0 flex-col pb-4">
        {/* Container for Welcome and ConversationStarter, shown when chat is empty */}
        {showWelcome && (
          <div className="flex flex-col items-center justify-center flex-grow mb-6"> {/* Centers content */}
            <Welcome className="mb-8" /> {/* Added margin-bottom */}
            <ConversationStarter onSend={handleSend} />
          </div>
        )}
        <InputBox
          className="h-full w-full"
          responding={responding} 
          onSend={handleSend}
        />
      </div>
    </div>
  );
}