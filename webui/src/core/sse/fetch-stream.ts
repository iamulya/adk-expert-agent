import { type StreamEvent } from "./StreamEvent";

export async function* fetchStream(
  url: string,
  init: RequestInit,
): AsyncIterable<StreamEvent> {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "text/event-stream",
    },
    ...init,
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`Failed to fetch from ${url}: ${response.status} ${response.statusText}. Body: ${errorBody}`);
  }

  const reader = response.body
    ?.pipeThrough(new TextDecoderStream())
    .getReader();
  if (!reader) {
    throw new Error("Response body is not readable");
  }

  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      if (buffer.trim()) { 
        const event = parseEvent(buffer.trim());
        if (event) yield event;
      }
      break;
    }
    buffer += value;
    let eventEndIndex;
    while ((eventEndIndex = buffer.indexOf("\n\n")) !== -1) {
      const chunk = buffer.slice(0, eventEndIndex);
      buffer = buffer.slice(eventEndIndex + 2);
      const event = parseEvent(chunk);
      if (event) {
        yield event;
      }
    }
  }
}

function parseEvent(chunk: string): StreamEvent | undefined {
  let eventType = "message";
  let eventData: string | null = null;
  let eventId: string | undefined;

  for (const line of chunk.split("\n")) {
    if (line.startsWith("event:")) {
      eventType = line.substring("event:".length).trim();
    } else if (line.startsWith("data:")) {
      const currentDataLine = line.substring("data:".length).trim();
      eventData = eventData === null ? currentDataLine : eventData + "\n" + currentDataLine;
    } else if (line.startsWith("id:")) {
      eventId = line.substring("id:".length).trim();
    }
  }

  if (eventData !== null) {
    return { event: eventType, data: eventData, id: eventId };
  }
  return undefined;
}
