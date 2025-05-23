import { env } from "~/env";

const appName = env.NEXT_PUBLIC_ADK_APP_NAME; 
const userId = 'user';

export function resolveServiceURL(
  pathTemplate: "create_session" | "run_sse_root", // Changed template name
  // sessionId is no longer needed for run_sse_root URL path
) {
  let specificPath = "";

  switch (pathTemplate) {
    case "create_session":
      specificPath = `/api/adk/apps/${appName}/users/${userId}/sessions`;
      break;
    case "run_sse_root": // For POSTing to the root /run_sse
      specificPath = `/api/adk/run_sse`;
      break;
    default:
      const _exhaustiveCheck: never = pathTemplate;
      throw new Error(`Unknown path template: ${_exhaustiveCheck}`);
  }
  
  if (env.NEXT_PUBLIC_API_URL && !env.NEXT_PUBLIC_API_URL.includes('localhost')) {
      const BASE_URL = env.NEXT_PUBLIC_API_URL.replace(/\/$/, '');
      return `${BASE_URL}${specificPath}`;
  }
  return specificPath;
}