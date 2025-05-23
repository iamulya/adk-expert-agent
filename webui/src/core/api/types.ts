export interface ChatResponseChunk {
  text: string;
  done: boolean;
  // ADK might also send 'type' (TEXT, FUNCTION_CALL, etc.) and 'content' for structured data
  // For simplicity, focusing on 'text' and 'done' from 'delta' events.
}

// Represents the response from ADK when a session is created
export interface AdkSession {
  name: string; // Full resource name like apps/app_name/users/user_id/sessions/session_id
  session_id: string; // The actual session ID part
  user_id: string;
  // createTime?: string;
  // updateTime?: string;
  // sessionState?: Record<string, any>; 
}

interface GenericEvent<T extends string, D extends object> {
  type: T;
  data: {
    id: string; // Message ID in UI
    thread_id: string; // Corresponds to ADK session_id
    role: "user" | "assistant";
    agent: string;
    finish_reason?: "stop" | "error";
  } & D;
}

export interface MessageChunkEvent
  extends GenericEvent<
    "message_chunk",
    {
      content?: string;
    }
  > {}

export type ChatEvent = MessageChunkEvent;