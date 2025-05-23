/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ["geist"],
  // Optional: Add a base path if you plan to deploy to a subdirectory
  // basePath: '/my-adk-expert-agent-ui',

  async rewrites() {
    const adkAgentBaseUrl = (process.env.ADK_AGENT_URL || 'http://localhost:8000').replace(/\/$/, '');
    return [
      {
        source: '/api/adk/stream', // For SSE GET requests
        destination: `${adkAgentBaseUrl}/stream`, // Note: query params will be forwarded
      },
      {
        source: '/api/adk', // For POST requests to the root agent
        destination: `${adkAgentBaseUrl}/`, // Target the root
      },
      // If you have other specific paths for ADK, add them above the general one
      {
        source: '/api/adk/:path*', 
        destination: `${adkAgentBaseUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;