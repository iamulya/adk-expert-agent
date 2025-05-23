import { ArrowUp } from "lucide-react";
import {
  type KeyboardEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { Tooltip } from "~/components/adk-chat-ui/tooltip";
import { Button } from "~/components/ui/button";
import { cn } from "~/lib/utils";
import { cancelStream, useStore } from "~/core/store/store"; // Add cancelStream and useStore

export function InputBox({
  className,
  responding,
  onSend,
  onCancel,
}: {
  className?: string;
  responding?: boolean;
  onSend?: (message: string) => void;
  onCancel?: () => void;
}) {
  const [message, setMessage] = useState("");
  const [imeStatus, setImeStatus] = useState<"active" | "inactive">("inactive");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isActuallyResponding = useStore((state) => state.responding); // Get actual responding state

  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  const handleSendOrCancel = useCallback(() => {
    if (isActuallyResponding) {
      cancelStream(); // Call the store's cancel function
    } else {
      if (message.trim() === "") {
        return;
      }
      if (onSend) {
        onSend(message);
        setMessage("");
        textareaRef.current?.focus();
      }
    }
  }, [isActuallyResponding, message, onSend]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (
        event.key === "Enter" &&
        !event.shiftKey &&
        !event.metaKey &&
        !event.ctrlKey &&
        imeStatus === "inactive"
      ) {
        event.preventDefault();
        handleSendOrCancel(); // Use the combined handler
      }
    },
    [imeStatus, handleSendOrCancel],
  );

  return (
    <div className={cn("bg-card relative rounded-xl border shadow-sm", className)}>
      <textarea
        ref={textareaRef}
        className="m-0 w-full resize-none border-none bg-transparent p-3 pr-12 text-sm placeholder:text-muted-foreground focus:outline-none"
        placeholder="Ask about Google ADK or a GitHub issue..."
        value={message}
        rows={1} 
        onCompositionStart={() => setImeStatus("active")}
        onCompositionEnd={() => setImeStatus("inactive")}
        onKeyDown={handleKeyDown}
        onChange={(event) => {
          setMessage(event.target.value);
          if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
            textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
          }
        }}
        style={{ maxHeight: "150px", overflowY: "auto" }} 
      />
      <div className="absolute right-2 bottom-2 flex shrink-0 items-center">
        <Tooltip title={responding ? "Stop" : "Send message"}>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 rounded-full"
            onClick={handleSendOrCancel}
            disabled={!message.trim() && !responding}
          >
            {responding ? (
              <div className="h-3.5 w-3.5 animate-pulse rounded-sm bg-foreground opacity-80" />
            ) : (
              <ArrowUp className="h-4 w-4 text-muted-foreground group-hover:text-foreground" />
            )}
          </Button>
        </Tooltip>
      </div>
    </div>
  );
}
