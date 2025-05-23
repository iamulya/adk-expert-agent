"use client";
import { useCallback, useRef } from "react";
import { ConversationStarter } from "./components/conversation-starter";
import { InputBox } from "./components/input-box";
import { MessageListView } from "./components/message-list-view";
import { useMessageIds, useStore, sendMessage as sendApiMessage } from "~/core/store/store";
import { cn } from "~/lib/utils";

export default function Main() {
  const messageIds = useMessageIds();
  const responding = useStore((state) => state.responding);
  const abortControllerRef = useRef<AbortController | null>(null);

  const handleSend = useCallback(
    async (message: string) => {
      const abortController = new AbortController();
      abortControllerRef.current = abortController;
      try {
        await sendApiMessage(message, {}, { abortSignal: abortController.signal });
      } catch (e) {
        console.error("Send message error in Main.tsx", e);
      }
    },
    [],
  );

  const handleCancel = useCallback(() => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    useStore.setState({ responding: false });
  }, []);

  return (
    <div className="flex h-full w-full max-w-2xl flex-col justify-center px-4 pt-16 pb-4">
      <MessageListView className="flex-grow" />
      <div className="relative mt-4 flex h-auto shrink-0 flex-col pb-4">
        {!responding && messageIds.length === 0 && (
          <ConversationStarter
            className="mb-4"
            onSend={handleSend}
          />
        )}
        <InputBox
          className="h-full w-full"
          responding={responding}
          onSend={handleSend}
          onCancel={handleCancel}
        />
      </div>
    </div>
  );
}
