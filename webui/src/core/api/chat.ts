import { env } from "~/env";
import { fetchStream } from "../sse";
import { resolveServiceURL } from "./resolve-service-url";
import type { ChatEvent, ChatResponseChunk } from "./types";
import { nanoid } from "nanoid";
import { sleep } from "../utils";

export async function* chatStream(
  userMessage: string,
  params: {
    session_id: string;
  },
  options: { abortSignal?: AbortSignal } = {},
): AsyncIterable<ChatEvent> {
  if (env.NEXT_PUBLIC_MOCK_API) {
    yield* mockChatStream(userMessage, params, options);
    return;
  }

  const stream = fetchStream(resolveServiceURL(""), { 
    method: "POST",
    body: JSON.stringify({
      session_id: params.session_id,
      user_input: userMessage,
    }),
    headers: {
      "Content-Type": "application/json",
      "Accept": "text/event-stream",
    },
    signal: options.abortSignal,
  });

  for await (const event of stream) {
    if (event.event === "delta") {
      try {
        const chunkData = JSON.parse(event.data) as ChatResponseChunk;
        yield {
          type: "message_chunk",
          data: {
            id: params.session_id, 
            role: "assistant",
            agent: "adk_expert_agent",
            thread_id: params.session_id,
            content: chunkData.text,
            finish_reason: chunkData.done ? "stop" : undefined,
          },
        } as ChatEvent;
        if (chunkData.done) break;
      } catch (e) {
        console.error("Failed to parse ADK delta event data:", event.data, e);
      }
    } else if (event.event === "close") {
      yield {
        type: "message_chunk",
        data: {
          id: params.session_id,
          role: "assistant",
          agent: "adk_expert_agent",
          thread_id: params.session_id,
          content: "",
          finish_reason: "stop",
        },
      } as ChatEvent;
      break;
    }
  }
}

async function* mockChatStream(
  userMessage: string,
  params: { session_id: string },
  _options: { abortSignal?: AbortSignal } = {},
): AsyncIterable<ChatEvent> {
  const mockResponses = [
    "Hello! I am your ADK Expert Agent. ",
    "I can help you with questions about Google's Agent Development Kit. ",
    `You asked about: "${userMessage}". `,
    "Let me check that for you... ",
    "The Agent Development Kit (ADK) is designed to simplify building complex AI agents. ",
    "It offers tools for state management, multi-agent orchestration, and easy integration with Google Cloud services. ",
    "What else can I assist you with regarding ADK?",
  ];
  const messageId = params.session_id + "_" + Date.now();

  for (const textChunk of mockResponses) {
    await sleep(150 + Math.random() * 250);
    yield {
      type: "message_chunk",
      data: {
        id: messageId,
        role: "assistant",
        agent: "adk_expert_agent",
        thread_id: params.session_id,
        content: textChunk,
      },
    } as ChatEvent;
  }
  await sleep(100);
  yield {
    type: "message_chunk",
    data: {
      id: messageId,
      role: "assistant",
      agent: "adk_expert_agent",
      thread_id: params.session_id,
      content: "",
      finish_reason: "stop",
    },
  } as ChatEvent;
}
