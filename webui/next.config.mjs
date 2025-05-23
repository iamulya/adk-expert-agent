/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ["geist"],
  // Optional: Add a base path if you plan to deploy to a subdirectory
  // basePath: '/my-adk-expert-agent-ui',

  async rewrites() {
    return [
      {
        source: '/api/adk/:path*', 
        destination: process.env.ADK_AGENT_URL || 'http://localhost:8000/:path*',
      },
    ];
  },
};

export default nextConfig;
