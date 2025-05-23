import { env } from "~/env";

export function resolveServiceURL(path: string) {
  let BASE_URL = env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/"; 
  if (!BASE_URL.endsWith("/")) {
    BASE_URL += "/";
  }
  
  const uiHost = typeof window !== 'undefined' ? window.location.host : '';
  // Determine if running in dev mode (Next.js default port 3000 or 3001) vs directly or proxied
  const isLikelyNextJsDevServer = uiHost.includes('localhost:3000') || uiHost.includes('localhost:3001');
  
  // If the BASE_URL is the same as the UI host (meaning it's proxied) OR if it's a relative path (starts with /)
  // OR if it's the default ADK agent URL and UI is on a typical Next.js dev port, assume proxy.
  const shouldUseProxy = BASE_URL.startsWith('/') || 
                         (typeof window !== 'undefined' && BASE_URL.startsWith(window.location.origin)) ||
                         (BASE_URL.includes('localhost:8000') && isLikelyNextJsDevServer);


  if (shouldUseProxy) {
      if (path === "stream" || path === "") { 
          return `/api/adk${path === "stream" ? '/stream' : ''}`; 
      }
      return `/api/adk/${path.startsWith('/') ? path.substring(1) : path}`;
  }
  return new URL(path, BASE_URL).toString();
}
