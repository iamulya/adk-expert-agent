import { motion } from "framer-motion";
import { cn } from "~/lib/utils";

const questions = [
  "What is Google ADK?",
  "How do I set up a new ADK agent?",
  "Explain the AgentTool in ADK.",
  "Can you help with GitHub issue #123 in google/adk-python?",
  "Show in a sequence diagram how a user request is handled by ADK",
  "Create a PDF document about ADK tools."
];

export function ConversationStarter({
  className,
  onSend,
}: {
  className?: string;
  onSend?: (message: string) => void;
}) {
  return (
    <div className={cn("flex flex-col items-center", className)}>
      <p className="text-muted-foreground mb-3 text-sm">Or try one of these common questions:</p>
      <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-3">
        {questions.map((question, index) => (
          <motion.li
            key={question}
            className="flex w-full shrink-0"
            style={{ transition: "all 0.2s ease-out" }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -15 }}
            transition={{
              duration: 0.2,
              delay: index * 0.05 + 0.3,
              ease: "easeOut",
            }}
          >
            <button
              type="button"
              className="bg-card text-card-foreground hover:bg-accent w-full cursor-pointer rounded-lg border px-3 py-2.5 text-left text-xs shadow-sm transition-all duration-200 hover:shadow-md active:scale-[0.98]"
              onClick={() => {
                onSend?.(question);
              }}
            >
              {question}
            </button>
          </motion.li>
        ))}
      </ul>
    </div>
  );
}
