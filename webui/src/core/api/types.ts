export interface ChatResponseChunk {
  text: string;
  done: boolean;
}

interface GenericEvent<T extends string, D extends object> {
  type: T;
  data: {
    id: string;
    thread_id: string;
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
