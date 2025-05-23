import { parse } from "best-effort-json-parser";

export function parseJSON<T>(json: string | null | undefined, fallback: T): T {
  if (!json) {
    return fallback;
  }
  try {
    const raw = json
      .trim()
      .replace(/^```json\s*/, "")
      .replace(/^```\s*/, "")
      .replace(/\s*```$/, "");
    return parse(raw) as T;
  } catch (e) {
    console.warn("Failed to parse JSON, returning fallback:", e, "Original JSON:", json);
    return fallback;
  }
}
