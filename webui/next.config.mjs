/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ["geist"],

  async rewrites() {
    const adkAgentTarget = (process.env.ADK_AGENT_URL || 'http://localhost:8000').replace(/\/$/, '');
    const appName = process.env.NEXT_PUBLIC_ADK_APP_NAME || 'expert-agents';
    const userId = 'user'; 

    return [
      // For the session-specific /run_sse endpoint. No, this is WRONG based on screenshot.
      // The /run_sse is at the ROOT.

      // Corrected: For sending user input AND receiving SSE stream via POST to root /run_sse
      {
        source: '/api/adk/run_sse', // UI will call this (e.g., POST /api/adk/run_sse)
        destination: `${adkAgentTarget}/run_sse`, // Proxies to ADK's root /run_sse
      },
      // Create a new session for the specific app
      {
        source: `/api/adk/apps/${appName}/users/${userId}/sessions`,
        destination: `${adkAgentTarget}/apps/${appName}/users/${userId}/sessions`,
      },
      // Fallback for any other /api/adk/... paths
      {
        source: '/api/adk/:path*',
        destination: `${adkAgentTarget}/:path*`,
      },
    ];
  },
};

export default nextConfig;