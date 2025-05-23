import { type StreamEvent } from "./StreamEvent";

export async function* fetchStream(
  url: string,
  init: RequestInit, 
): AsyncIterable<StreamEvent> {
  console.log(`[SSE FETCH] Initiating fetch to ${url} with method ${init.method}`);
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json", 
      "Accept": "text/event-stream",
    },
    ...init, 
  });
  console.log(`[SSE FETCH] Response status from ${url}: ${response.status}`);

  if (!response.ok) {
    const errorBody = await response.text();
    console.error(`[SSE FETCH] Error body from ${url}: ${errorBody}`);
    throw new Error(`Failed to fetch from ${url}: ${response.status} ${response.statusText}. Body: ${errorBody}`);
  }
  if (response.headers.get("content-type")?.toLowerCase().indexOf("text/event-stream") === -1) {
    const responseBody = await response.text();
    console.warn(`[SSE FETCH] Expected text/event-stream from ${url}, but got ${response.headers.get("content-type")}. Body:`, responseBody);
    throw new Error(`Expected text/event-stream from ${url} but received ${response.headers.get("content-type")}`);
  }

  const reader = response.body
    ?.pipeThrough(new TextDecoderStream())
    .getReader();
  if (!reader) {
    console.error("[SSE FETCH] Response body is not readable for SSE.");
    throw new Error("Response body is not readable for SSE");
  }

  let buffer = "";
  console.log("[SSE FETCH] Starting to read SSE stream...");
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      console.log("[SSE FETCH] Stream reader marked as done.");
      if (buffer.trim()) { 
        console.log("[SSE FETCH] Processing remaining buffer before close:", buffer.trim());
        const event = parseEvent(buffer.trim());
        if (event) {
            console.log("[SSE FETCH] Yielding event from remaining buffer:", event);
            yield event;
        }
      }
      break;
    }
    buffer += value;
    // console.log("[SSE FETCH] Buffer updated with new value (length):", value?.length, "Total buffer length:", buffer.length);
    let eventEndIndex;
    while ((eventEndIndex = buffer.indexOf("\n\n")) !== -1) {
      const chunk = buffer.slice(0, eventEndIndex);
      buffer = buffer.slice(eventEndIndex + 2);
      // console.log("[SSE FETCH] Processing event chunk:", chunk);
      const event = parseEvent(chunk);
      if (event) {
        // console.log("[SSE FETCH] Parsed and yielding event:", event);
        yield event;
      }
    }
  }
  console.log("[SSE FETCH] Finished reading SSE stream.");
}

function parseEvent(chunk: string): StreamEvent | undefined {
  let eventType = "message"; 
  let eventData: string | null = null;
  let eventId: string | undefined;
  // console.log("[SSE PARSE] Attempting to parse chunk:\n---\n", chunk, "\n---");

  for (const line of chunk.split("\n")) {
    const trimmedLine = line.trim();
    if (trimmedLine === "") continue; // Skip empty lines after splitting

    if (trimmedLine.startsWith("event:")) {
      eventType = trimmedLine.substring("event:".length).trim();
    } else if (trimmedLine.startsWith("data:")) {
      const currentDataLine = trimmedLine.substring("data:".length).trim();
      eventData = eventData === null ? currentDataLine : eventData + "\n" + currentDataLine;
    } else if (trimmedLine.startsWith("id:")) {
      eventId = trimmedLine.substring("id:".length).trim();
    } else if (trimmedLine.startsWith(":")) {
      // This is an SSE comment, log and ignore
      console.log("[SSE PARSE] Ignoring SSE comment:", trimmedLine);
    } else {
      console.warn("[SSE PARSE] Ignoring unexpected line in SSE chunk:", trimmedLine);
    }
  }

  if (eventData !== null) {
    // console.log(`[SSE PARSE] Successfully parsed event: type=${eventType}, id=${eventId}, data=${eventData.substring(0,100)}...`);
    return { event: eventType, data: eventData, id: eventId };
  }
  // console.log("[SSE PARSE] No data field found in chunk, or data was empty after processing lines.");
  return undefined;
}