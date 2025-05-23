import type { ChatEvent, MessageChunkEvent } from "../api/types";
import { deepClone } from "../utils/deep-clone";
import type { Message } from "./types";

export function mergeMessage(message: Message, event: ChatEvent): Message {
  const newMessage = deepClone(message); 

  if (event.type === "message_chunk") {
    mergeTextMessage(newMessage, event);
  }

  if (event.data.finish_reason) {
    newMessage.finishReason = event.data.finish_reason;
    newMessage.isStreaming = false;
  }
  return newMessage;
}

function mergeTextMessage(message: Message, event: MessageChunkEvent) {
  if (event.data.content) {
    message.content += event.data.content;
    message.contentChunks.push(event.data.content);
  }
}
