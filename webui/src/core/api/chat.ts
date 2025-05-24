// webui/src/core/api/chat.ts
import { env } from "~/env";
import { resolveServiceURL } from "./resolve-service-url";
import type { ChatEvent, AdkSession } from "./types"; // ChatResponseChunk might be implicitly covered by parsedEventData
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

  const requestPayloadForRunSse = {
    appName: ADK_APP_NAME,
    userId: ADK_USER_ID,
    sessionId: adkSessionId,
    newMessage: newMessagePayloadForRunSse,
    streaming: false, // Keep streaming true for delta events
  };

  const requestPayloadForRunSsev05 = {
    app_name: ADK_APP_NAME,
    user_id: ADK_USER_ID,
    session_id: adkSessionId,
    new_message: newMessagePayloadForRunSse,
    streaming: false, // Keep streaming true for delta events
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
    body: JSON.stringify(requestPayloadForRunSsev05),
    headers: {
      "Content-Type": "application/json",
      "Accept": "text/event-stream",
    },
    signal: options.abortSignal,
  });

  const assistantMessageIdForUI = useStore.getState().messageIds.find(id => {
    const msg = useStore.getState().messages.get(id);
    return msg?.role === 'assistant' && msg?.isStreaming === true && msg?.threadId === adkSessionId;
  }) || "assistant_" + nanoid();


  let streamHasYieldedData = false;

  try {
    for await (const event of sseStream) {
      console.log("[API CHAT] Received SSE Raw Event from /run_sse:", JSON.stringify(event));
      if (options.abortSignal?.aborted) {
        console.log("[API CHAT] Aborting stream processing due to signal.");
        break;
      }
      streamHasYieldedData = true;

      // `parsedEventData` is the raw JSON payload from ADK for this specific SSE event.
      // It might contain `text` for simple deltas, or `content.parts` for structured messages.
      const parsedEventData = JSON.parse(event.data); 
      
      let uiMessageText = ""; // Text to be sent to the UI message store for this chunk/event
      let isStreamSegmentDone = false; // True if (event is 'message') or (delta event has 'done: true')
      let isAgentTurnCompletelyFinal = false; // True if this event signals the end of the agent's entire turn

      if (event.event === "delta" || event.event === "message") {
          isStreamSegmentDone = (event.event === "message") || (parsedEventData.done === true);

          let hasFunctionCall = false;
          let hasFunctionResponse = false; // Tracks if a function_response is present in the current event part

          if (parsedEventData.content && parsedEventData.content.parts && parsedEventData.content.parts.length > 0) {
              // For simplicity, assuming the primary information (text, function_call, function_response)
              // is in the first part of the content. ADK might send multiple parts.
              const adkPart = parsedEventData.content.parts[0]; 

              if (adkPart.functionCall) {
                  hasFunctionCall = true;
                  // If agent "thinks" (text) before calling a tool, capture it.
                  if (adkPart.text) {
                      uiMessageText = adkPart.text;
                  }
                  // A function call means the agent's turn is NOT final yet.
                  isAgentTurnCompletelyFinal = false;
              } else if (adkPart.functionResponse) {
                  hasFunctionResponse = true;
                  // Check the ADK-specific 'is_final_response' flag as per prompt requirement
                  // This flag should ideally be sent by ADK on the part containing the function_response
                  // when that function_response is part of the agent's final utterance to the user.
                  if ((parsedEventData.actions && (parsedEventData.actions.skipSummarization || parsedEventData.actions.skip_summarization)) || parsedEventData.isFinalResponse) {
                      isAgentTurnCompletelyFinal = true; // This is the key condition from the prompt
                      if (adkPart.functionResponse.response && adkPart.functionResponse.response.result && adkPart.functionResponse.response.result.parts.length > 0) {
                          const adkFuncResponsePart = adkPart.functionResponse.response.result.parts[0]; 
                          uiMessageText = adkFuncResponsePart.text; // Use this text as the agent's final message
                      } else {
                          uiMessageText = ""; // Final response, but no accompanying text
                      }
                  } else {
                      // This is an intermediate function_response. The agent's turn is not final.
                      // UI typically doesn't show these directly as user-facing messages
                      // unless the agent explicitly formats them into its own 'text' response later.
                      isAgentTurnCompletelyFinal = false;
                  }
              } else if (adkPart.text) { // Plain text part
                  uiMessageText = adkPart.text;
              }
          } else if (parsedEventData.text) { 
              // Handles simpler event structures, e.g., a delta containing only text
              uiMessageText = parsedEventData.text;
          }

          // Determine overall turn finality if not already set by a final function_response
          if (!isAgentTurnCompletelyFinal) { 
              if (hasFunctionCall || hasFunctionResponse /* implies !adkPart.is_final_response */) {
                  // If a functionCall is made, or a non-final functionResponse is received,
                  // the agent's turn is not over.
                  isAgentTurnCompletelyFinal = false; 
              } else if (isStreamSegmentDone) {
                  // If a delta stream ends (parsedEventData.done) or it's a complete 'message' event,
                  // AND there are no pending tool interactions (functionCall or non-final functionResponse from this event),
                  // then the agent's turn is considered final.
                  isAgentTurnCompletelyFinal = true;
              }
          }

          if(parsedEventData.error && parsedEventData.error.indexOf("500 INTERNAL") !== -1) {
            uiMessageText = "The model encountered an internal error. Please try again later.";
            console.error("[API CHAT] Internal Server Error in ADK response:", parsedEventData.error);
          }
          
          // Yield to UI store if there's text to display or if it's a final signal (even if empty)
          if (uiMessageText || isAgentTurnCompletelyFinal) {
              yield {
                  type: "message_chunk",
                  data: {
                      id: assistantMessageIdForUI,
                      thread_id: adkSessionId,
                      role: "assistant",
                      agent: parsedEventData.agent || ADK_APP_NAME, 
                      content: uiMessageText, 
                      finish_reason: isAgentTurnCompletelyFinal ? "stop" : undefined,
                  },
              } as ChatEvent;
          }

          // Break loop if the agent's turn is completely final
          if (isAgentTurnCompletelyFinal) {
              console.log(`[API CHAT] Agent turn is final. Event: ${event.event}. Breaking SSE loop.`);
              break;
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
      const finalStoreMessage = useStore.getState().messages.get(assistantMessageIdForUI);
      if (finalStoreMessage?.isStreaming && !options.abortSignal?.aborted) {
        // This ensures that if the loop exited for reasons other than a clean 'isAgentTurnCompletelyFinal' or abortion,
        // the UI still gets a final "stop" signal for the message.
        console.warn(`[API CHAT] SSE loop finished, but message ${assistantMessageIdForUI} still marked as streaming. Forcing stop.`);
         yield {
            type: "message_chunk",
            data: {
              id: assistantMessageIdForUI, thread_id: adkSessionId, role: "assistant",
              agent: finalStoreMessage.agent || ADK_APP_NAME, content: finalStoreMessage.content, finish_reason: "stop",
            },
          } as ChatEvent;
      }
  }
}