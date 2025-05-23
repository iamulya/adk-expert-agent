export type MessageRole = "user" | "assistant";

export interface Message {
  id: string;
  threadId: string; 
  agent?: string;
  role: MessageRole;
  isStreaming?: boolean;
  content: string;
  contentChunks: string[];
  finishReason?: "stop" | "error";
}
