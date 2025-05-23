// webui/src/app/chat/components/message-list-view.tsx
import { motion } from "framer-motion";
import { useRef } from "react";
import { Markdown } from "~/components/adk-chat-ui/markdown";
import { ScrollContainer, type ScrollContainerRef } from "~/components/adk-chat-ui/scroll-container";
import type { Message } from "~/core/messages";
import { useMessage, useMessageIds, useStore } from "~/core/store/store";
import { cn } from "~/lib/utils";
import { LoadingAnimation } from "~/components/adk-chat-ui/loading-animation";
import ImageComponent from "~/components/adk-chat-ui/image";
import { Card, CardContent } from "~/components/ui/card";


export function MessageListView({ className }: { className?: string }) {
  const scrollContainerRef = useRef<ScrollContainerRef>(null);
  const messageIds = useMessageIds();
  const responding = useStore((state) => state.responding);

  return (
    <ScrollContainer
      className={cn("flex h-full w-full flex-col overflow-hidden", className)}
      scrollShadowColor="var(--app-background)"
      autoScrollToBottom
      ref={scrollContainerRef}
    >
      <ul className="flex flex-col gap-3 px-2 py-4 md:px-0">
        {messageIds.map((messageId) => (
          <MessageListItem key={messageId} messageId={messageId} />
        ))}
        {/* This general loading animation appears if 'responding' is true AND the last message was from the user.
            It will typically be very brief as the assistant placeholder message gets added quickly.
            The per-message loading animation inside the assistant's bubble is usually more prominent.
        */}
        {responding && messageIds.length > 0 && getMessageRole(messageIds[messageIds.length - 1]) === 'user' && (
          <motion.li
            className={cn(
              "mt-1 flex w-full flex-col items-start" // Aligns to left like an assistant message
            )}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
          >
            <MessageBubble message={{ role: "assistant" } as Message}> {/* Simulates an assistant bubble */}
              <LoadingAnimation size="sm" />
            </MessageBubble>
          </motion.li>
        )}
      </ul>
    </ScrollContainer>
  );
}

function getMessageRole(messageId: string): Message['role'] | undefined {
  const msg = useStore.getState().messages.get(messageId);
  return msg?.role;
}

function MessageListItem({
  className,
  messageId,
}: {
  className?: string;
  messageId: string;
}) {
  const message = useMessage(messageId);

  if (!message) {
    return null;
  }

  const gcsImageRegex = /^(https?:\/\/storage\.googleapis\.com\/[^?]+\.(?:png|jpe?g|gif|webp))(?:\?.*)?$/i;
  const hasContent = message.content && message.content.trim() !== "";
  const isGcsImage = hasContent && gcsImageRegex.test(message.content);

  return (
    <motion.li
      className={cn(
        "flex w-full flex-col",
        message.role === "user" ? "items-end" : "items-start",
        className,
      )}
      // Key is already on the <MessageListItem> in the map in MessageListView
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      style={{ transition: "all 0.2s ease-out" }}
      transition={{
        duration: 0.2,
        ease: "easeOut",
      }}
    >
      <MessageBubble message={message}>
        <div className="flex w-full flex-col">
          {/* Render GCS Image if applicable */}
          {hasContent && isGcsImage && (
            <Card className="my-1 w-auto max-w-full self-start md:max-w-md">
              <CardContent className="p-1.5">
                <ImageComponent src={message.content} alt="Mermaid Diagram" className="max-h-[400px] w-auto rounded-md object-contain" />
              </CardContent>
            </Card>
          )}

          {/* Render Markdown if content is not a GCS image (or if it's user message) */}
          {hasContent && !isGcsImage && (
            <Markdown>{message.content}</Markdown>
          )}

          {/* Per-message loading animation for assistant while streaming */}
          {message.isStreaming && message.role === "assistant" && (
            // Adjust margin if there's no content yet to avoid double spacing
            <LoadingAnimation size="sm" className={cn("mt-1 self-start", { "mt-0": !hasContent })} />
          )}
        </div>
      </MessageBubble>
    </motion.li>
  );
}

// Standard MessageBubble component
function MessageBubble({
  className,
  message,
  children,
}: {
  className?: string;
  message: Message; // Can be a partial message for the general loader
  children: React.ReactNode;
}) {
  return (
    <div
      className={cn(
        "flex w-fit max-w-[85%] flex-col rounded-xl px-3.5 py-2.5 shadow-sm md:max-w-[75%]",
        message.role === "user"
          ? "bg-brand text-primary-foreground rounded-br-sm"
          : "bg-card text-card-foreground rounded-bl-sm",
        className,
      )}
    >
      {children}
    </div>
  );
}