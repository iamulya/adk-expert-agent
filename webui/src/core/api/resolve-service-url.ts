import { env } from "~/env";

export function resolveServiceURL(path: string, forSse: boolean = false) {
  // This function now constructs the path that the Next.js UI will call.
  // The Next.js server will then proxy these requests based on next.config.mjs rewrites.

  // If NEXT_PUBLIC_API_URL is set, it implies the UI is calling an external proxy/gateway
  // or directly if deployed appropriately. For local dev with Next.js proxy, this might not be set.
  if (env.NEXT_PUBLIC_API_URL) {
    let BASE_URL = env.NEXT_PUBLIC_API_URL.replace(/\/$/, ''); // Remove trailing slash if present
    if (path === "stream") {
      return `${BASE_URL}/stream`; // Or whatever the configured stream path is
    }
    return `${BASE_URL}/${path === "" ? "" : path}`;
  }

  // Default to using the Next.js proxy paths
  if (forSse && path === "stream") {
    return "/api/adk/stream"; // Path for EventSource
  }
  if (path === "") { // For POSTing to the root agent
    return "/api/adk";
  }
  return `/api/adk/${path}`; // For any other specific paths (less common for basic chat)
}