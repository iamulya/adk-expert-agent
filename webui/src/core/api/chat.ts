import { env } from "~/env";
import { resolveServiceURL } from "./resolve-service-url";
import type { ChatEvent, ChatResponseChunk, AdkSession } from "./types";
import { nanoid } from "nanoid";
import { sleep } from "../utils";
import { fetchStream } from "../sse";

const ADK_APP_NAME = env.NEXT_PUBLIC_ADK_APP_NAME;
const ADK_USER_ID = 'user';

export async function createAdkSession(
  initialState: Record<string, any> = {},
  options: { abortSignal?: AbortSignal } = {},
): Promise<AdkSession> {
  if (env.NEXT_PUBLIC_MOCK_API) {
    const mockSessionId = `mock-session-${nanoid()}`;
    console.log("[MOCK API CHAT] Creating session:", mockSessionId);
    return { name: `apps/${ADK_APP_NAME}/users/${ADK_USER_ID}/sessions/${mockSessionId}`, session_id: mockSessionId, user_id: ADK_USER_ID };
  }

  const url = resolveServiceURL("create_session");
  console.log(`[API CHAT] Attempting to create session at URL: ${url}`);
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_state: initialState }), 
    signal: options.abortSignal,
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error(`[API CHAT] Failed to create ADK session. Status: ${response.status}. Body: ${errorText}`);
    throw new Error(`Failed to create ADK session: ${response.status} ${response.statusText} - Detail: ${errorText}`);
  }
  const responseData = await response.json();
  const sessionIdFromServer = responseData.id;

  console.log("[API CHAT] ADK Session created successfully:", responseData, "Parsed Session ID:", sessionIdFromServer);
  return { 
    name: responseData.name, 
    session_id: sessionIdFromServer!,
    user_id: ADK_USER_ID 
  };
}

export async function* postToRunSseAndStream(
  adkSessionId: string,
  userMessage: string,
  options: { abortSignal?: AbortSignal } = {},
): AsyncIterable<ChatEvent> {

  const newMessagePayloadForRunSse = {
    role: "user",
    parts: [{ text: userMessage }],
  };

  const requestPayloadForRunSse = { 
    appName: ADK_APP_NAME,
    userId: ADK_USER_ID,
    sessionId: adkSessionId,
    newMessage: newMessagePayloadForRunSse,
    streaming: false,
  };

  if (env.NEXT_PUBLIC_MOCK_API) {
    console.log("[MOCK API CHAT] postToRunSseAndStream called for session:", adkSessionId);
    const mockResponses = [
      `Mock response for: "${userMessage}" in session ${adkSessionId}. `,
      "Thinking... ",
      "This is a streamed chunk. ",
      "Another chunk for you.",
    ];
    const assistantMessageId = adkSessionId + "_mock_assistant_" + Date.now();
    for (const textChunk of mockResponses) {
        await sleep(300);
        yield {
            type: "message_chunk",
            data: {
            id: assistantMessageId, thread_id: adkSessionId, role: "assistant",
            agent: ADK_APP_NAME, content: textChunk,
            },
        } as ChatEvent;
    }
    await sleep(100);
    yield {
        type: "message_chunk",
        data: {
        id: assistantMessageId, thread_id: adkSessionId, role: "assistant",
        agent: ADK_APP_NAME, content: "", finish_reason: "stop",
        },
    } as ChatEvent;
    return;
  }

  const url = resolveServiceURL("run_sse_root");
  console.log(`[API CHAT] Calling session-specific /run_sse at URL: ${url}`);
  console.log(`[API CHAT] Payload for session-specific /run_sse:`, JSON.stringify(requestPayloadForRunSse, null, 2));
  
  const sseStream = fetchStream(url, {
    method: "POST",
    body: JSON.stringify(requestPayloadForRunSse),
    headers: {
      "Content-Type": "application/json",
      "Accept": "text/event-stream",
    },
    signal: options.abortSignal,
  });

  const assistantMessageIdForUI = "assistant_" + nanoid();
  let streamHasYieldedData = false;

  try {
    for await (const event of sseStream) {
      console.log("[API CHAT] Received SSE Raw Event from /run_sse:", JSON.stringify(event));
      streamHasYieldedData = true;

      if (event.event === "delta") {
        try {
          const chunkData = JSON.parse(event.data) as ChatResponseChunk;
          console.log("[API CHAT] Parsed SSE Delta Chunk from /run_sse:", chunkData);
          yield {
            type: "message_chunk",
            data: {
              id: assistantMessageIdForUI,
              thread_id: adkSessionId, 
              role: "assistant",
              agent: ADK_APP_NAME, 
              content: chunkData.text,
              finish_reason: chunkData.done ? "stop" : undefined,
            },
          } as ChatEvent;
          if (chunkData.done && (chunkData.text === null || chunkData.text === "")) { 
             console.log("[API CHAT] /run_sse Delta 'done' with null/empty text, signaling stop.");
             yield { 
              type: "message_chunk",
              data: {
                id: assistantMessageIdForUI, thread_id: adkSessionId, role: "assistant",
                agent: ADK_APP_NAME, content: "", finish_reason: "stop",
              },
            } as ChatEvent;
             break;
          }
        } catch (e) {
          console.error("[API CHAT] Failed to parse ADK delta event from /run_sse:", event.data, e);
        }
      } else if (event.event === "close") {
        console.log("[API CHAT] Received SSE 'close' event from /run_sse.");
        yield {
          type: "message_chunk",
          data: {
            id: assistantMessageIdForUI, thread_id: adkSessionId, role: "assistant",
            agent: ADK_APP_NAME, content: "", finish_reason: "stop",
          },
        } as ChatEvent;
        break;
      } else {
        console.warn("[API CHAT] Received unknown SSE event type from /run_sse:", event.event, "Data:", event.data);
      }
    }
  } catch (error) {
      if (options.abortSignal?.aborted) {
          console.log("[API CHAT] SSE stream fetch for /run_sse aborted.");
      } else {
          console.error("[API CHAT] Error iterating SSE stream for /run_sse:", error);
          throw error; 
      }
  } finally {
      console.log("[API CHAT] Exiting /run_sse SSE stream processing loop. streamHasYieldedData:", streamHasYieldedData);
      if (!streamHasYieldedData && !options.abortSignal?.aborted) {
          console.log("[API CHAT] No data yielded from /run_sse stream, ensuring stop signal for UI.");
          yield {
            type: "message_chunk",
            data: {
              id: assistantMessageIdForUI, thread_id: adkSessionId, role: "assistant",
              agent: ADK_APP_NAME, content: "", finish_reason: "stop",
            },
          } as ChatEvent;
      }
  }
}