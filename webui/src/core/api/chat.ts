// webui/src/core/api/chat.ts
import { env } from "~/env";
import { resolveServiceURL } from "./resolve-service-url";
import type { ChatEvent, ChatResponseChunk, AdkSession } from "./types";
import { nanoid } from "nanoid";
import { sleep } from "../utils";
import { fetchStream } from "../sse";
import { useStore } from "../store/store";

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

  // Ensure this reflects your desired backend behavior.
  // If ADK backend is configured for streaming only with "streaming": true,
  // and "streaming": false means "send full response in one 'message' event",
  // the logic below is now designed to handle that.
  const requestPayloadForRunSse = {
    appName: ADK_APP_NAME,
    userId: ADK_USER_ID,
    sessionId: adkSessionId,
    newMessage: newMessagePayloadForRunSse,
    streaming: false, // Set this to true if you want the backend to attempt to stream via "delta" events
  };

  if (env.NEXT_PUBLIC_MOCK_API) {
    console.log("[MOCK API CHAT] postToRunSseAndStream called for session:", adkSessionId);
    const mockResponses = [
      `Mock response for: "${userMessage}" in session ${adkSessionId}. `,
      "Thinking... ",
      "This is a streamed chunk. ",
      "Another chunk for you.",
    ];
    const assistantMessageId = adkSessionId + "_mock_assistant_" + Date.now(); // Use a different ID for mock to avoid clash if UI uses nanoid()
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

  // This assistantMessageIdForUI is what the UI store will use for its message.id
  // It's generated on the client to have an immediate ID for the placeholder.
  const assistantMessageIdForUI = useStore.getState().messageIds.find(id => {
    const msg = useStore.getState().messages.get(id);
    return msg?.role === 'assistant' && msg?.isStreaming === true && msg?.threadId === adkSessionId;
  }) || "assistant_" + nanoid(); // Fallback, but should ideally get from store if placeholder was added.


  let streamHasYieldedData = false;

  try {
    for await (const event of sseStream) {
      console.log("[API CHAT] Received SSE Raw Event from /run_sse:", JSON.stringify(event));
      if (options.abortSignal?.aborted) {
        console.log("[API CHAT] Aborting stream processing due to signal.");
        break;
      }
      streamHasYieldedData = true;

      if (event.event === "delta" || event.event === "message") {
        try {
          const parsedEventData = JSON.parse(event.data);
          let contentText = "";
          let isDone = false;

          if (event.event === "delta") {
            const chunkData = parsedEventData as ChatResponseChunk;
            console.log("[API CHAT] Parsed SSE Delta Chunk from /run_sse:", chunkData);
            contentText = chunkData.text ?? ""; // Ensure contentText is always a string
            isDone = chunkData.done;
          } else if (event.event === "message") {
            console.log("[API CHAT] Processing 'message' event from /run_sse:", parsedEventData);
            if (parsedEventData.content && parsedEventData.content.parts && parsedEventData.content.parts[0] && typeof parsedEventData.content.parts[0].text === 'string') {
              contentText = parsedEventData.content.parts[0].text;
            } else {
              console.warn("[API CHAT] Could not extract text from 'message' event structure. Raw data:", parsedEventData);
              contentText = "[Error: Could not parse backend message event content]";
            }
            isDone = true; // A single "message" event implies the full response is delivered.
          }

          yield {
            type: "message_chunk",
            data: {
              id: assistantMessageIdForUI,
              thread_id: adkSessionId,
              role: "assistant",
              agent: ADK_APP_NAME,
              content: contentText,
              finish_reason: isDone ? "stop" : undefined,
            },
          } as ChatEvent;

          if (isDone && event.event === "message") { // If full message event is processed, break
            console.log("[API CHAT] Full 'message' event processed, breaking SSE loop.");
            break;
          }
          if (isDone && event.event === "delta" && (contentText === null || contentText === "")) {
             console.log("[API CHAT] /run_sse Delta 'done' with null/empty text, also ensuring stop signal.");
             // This case might be redundant if the above yield already set finish_reason: "stop"
             // but kept for safety if a "done" delta has no text.
          }

        } catch (e) {
          console.error(`[API CHAT] Failed to parse or process ADK ${event.event} event from /run_sse:`, event.data, e);
          yield {
            type: "message_chunk",
            data: {
              id: assistantMessageIdForUI, thread_id: adkSessionId, role: "assistant",
              agent: ADK_APP_NAME, content: "[Error processing backend response stream]", finish_reason: "error",
            },
          } as ChatEvent;
        }
      } else if (event.event === "close") {
        console.log("[API CHAT] Received SSE 'close' event from /run_sse.");
        yield { // Ensure UI knows the stream is finished
          type: "message_chunk",
          data: {
            id: assistantMessageIdForUI, thread_id: adkSessionId, role: "assistant",
            agent: ADK_APP_NAME, content: "", finish_reason: "stop",
          },
        } as ChatEvent;
        break;
      } else {
        console.warn("[API CHAT] Received unhandled SSE event type from /run_sse:", event.event, "Data:", event.data);
      }
    }
  } catch (error) {
      if (options.abortSignal?.aborted) {
          console.log("[API CHAT] SSE stream fetch for /run_sse aborted by signal.");
      } else {
          console.error("[API CHAT] Error iterating SSE stream for /run_sse:", error);
          yield {
            type: "message_chunk",
            data: {
              id: assistantMessageIdForUI, thread_id: adkSessionId, role: "assistant",
              agent: ADK_APP_NAME, content: "[Network or stream iteration error]", finish_reason: "error",
            },
          } as ChatEvent;
      }
  } finally {
      console.log("[API CHAT] Exiting /run_sse SSE stream processing loop. streamHasYieldedData:", streamHasYieldedData, "Aborted:", options.abortSignal?.aborted);
      // If the stream was aborted, or if it finished naturally (e.g. "close" or "message" event break),
      // the finish_reason should have already been set.
      // This ensures a stop signal if the loop terminates unexpectedly without yielding a "done" state AND wasn't aborted.
      if (!streamHasYieldedData && !options.abortSignal?.aborted) {
          console.log("[API CHAT] No data yielded from /run_sse stream and not aborted, ensuring stop signal for UI.");
          yield {
            type: "message_chunk",
            data: {
              id: assistantMessageIdForUI, thread_id: adkSessionId, role: "assistant",
              agent: ADK_APP_NAME, content: "", finish_reason: "stop", // Or "error" if appropriate
            },
          } as ChatEvent;
      }
      // Check if the message is still marked as streaming in the store, if so, mark it as stopped.
      // This is a final safeguard.
      const finalStoreMessage = useStore.getState().messages.get(assistantMessageIdForUI);
      if (finalStoreMessage?.isStreaming && !options.abortSignal?.aborted) {
        console.warn(`[API CHAT] SSE loop finished, but message ${assistantMessageIdForUI} still marked as streaming in store. Forcing stop.`);
         yield {
            type: "message_chunk",
            data: {
              id: assistantMessageIdForUI, thread_id: adkSessionId, role: "assistant",
              agent: ADK_APP_NAME, content: finalStoreMessage.content, finish_reason: "stop",
            },
          } as ChatEvent;
      }
  }
}