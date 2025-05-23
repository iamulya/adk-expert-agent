import { env } from "~/env";
import { resolveServiceURL } from "./resolve-service-url";
import type { ChatEvent, ChatResponseChunk } from "./types";
import { nanoid } from "nanoid";
import { sleep } from "../utils";

/**
 * Sends the user's message to the ADK agent.
 * This is a standard POST request.
 */
export async function sendAdkMessage(
  userMessage: string,
  sessionId: string,
  options: { abortSignal?: AbortSignal } = {},
): Promise<Response> {
  if (env.NEXT_PUBLIC_MOCK_API) {
    console.log("[MOCK] Sending message:", userMessage, "Session:", sessionId);
    // Simulate a successful send for mock environment
    return new Response(JSON.stringify({ status: "ok", messageId: nanoid() }), { status: 200 });
  }

  const url = resolveServiceURL(""); // POST to the root agent, proxied to /api/adk
  return fetch(url, {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      user_input: userMessage,
    }),
    headers: {
      "Content-Type": "application/json",
    },
    signal: options.abortSignal,
  });
}

/**
 * Creates an EventSource to listen for streaming events from the ADK agent.
 * This is for the GET request to the /stream endpoint.
 */
export function createAdkEventSource(
  sessionId: string,
  onMessage: (eventData: ChatResponseChunk) => void,
  onError: (error: Event | Error) => void,
  onOpen?: () => void,
  onClose?: () => void,
): EventSource | null {
  if (env.NEXT_PUBLIC_MOCK_API) {
    // Simulate mock stream after a delay
    const mockMessageId = sessionId + "_" + Date.now();
    const mockResponses = [
      "This is a mocked ADK agent response. ",
      `You said: "${"mocked original message"}" for session ${sessionId}. `,
      "I am processing your request... ",
      "Here are some details about ADK... ",
      "The stream is now closing.",
    ];
    let i = 0;
    const intervalId = setInterval(() => {
      if (i < mockResponses.length) {
        onMessage({ text: mockResponses[i]!, done: false });
        i++;
      } else {
        onMessage({ text: "", done: true }); // Simulate done
        if (onClose) onClose();
        clearInterval(intervalId);
      }
    }, 500);
    // For mock, we don't return a real EventSource, just simulate callbacks
    if (onOpen) onOpen();
    return null; // Or a mock EventSource object if needed for type checking
  }

  // ADK `/stream` endpoint expects session_id as a query parameter
  const url = resolveServiceURL("stream", true) + `?session_id=${sessionId}`;
  const eventSource = new EventSource(url);

  eventSource.onopen = () => {
    console.log("SSE connection opened to ADK agent for session:", sessionId);
    if (onOpen) onOpen();
  };

  eventSource.addEventListener("delta", (event) => {
    try {
      const chunkData = JSON.parse(event.data) as ChatResponseChunk;
      onMessage(chunkData);
    } catch (e) {
      console.error("Failed to parse ADK delta event:", event.data, e);
      onError(e instanceof Error ? e : new Error("Error parsing delta event"));
    }
  });

  eventSource.addEventListener("close", () => {
    console.log("SSE connection closed by ADK agent for session:", sessionId);
    if (onClose) onClose();
    eventSource.close(); // Ensure it's fully closed client-side too
  });
  
  eventSource.onerror = (errorEvent) => {
    console.error("SSE error for session:", sessionId, errorEvent);
    onError(new Error("SSE connection error. Check console."));
    // EventSource will attempt to reconnect by default on network errors.
    // If it's a fatal error (like 404 on /stream), it might stop.
    // The 'close' event from the server is the definitive end.
  };

  return eventSource;
}