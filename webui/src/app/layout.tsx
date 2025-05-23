import "~/styles/globals.css";
import { type Metadata } from "next";
import { GeistSans } from "geist/font/sans"; // Corrected import
// If you also want the mono version: import { GeistMono } from "geist/font/mono";
import Script from "next/script";

import { ThemeProviderWrapper } from "~/components/adk-chat-ui/theme-provider-wrapper";
import { Toaster } from "~/components/adk-chat-ui/toaster";

export const metadata: Metadata = {
  title: "ADK Expert Agent",
  description:
    "An expert agent for Google's Agent Development Kit (ADK).",
  icons: [{ rel: "icon", url: "/favicon.ico" }],
};

// No need to call GeistSans like a function here if using its variable directly.
// const geist = Geist({ ... }) // This line is removed

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    // Apply the font variable directly to the html tag
    // If you also imported GeistMono, you would add ${GeistMono.variable} here too.
    <html lang="en" className={`${GeistSans.variable}`} suppressHydrationWarning>
      <head>
        <Script id="markdown-it-fix" strategy="beforeInteractive">
          {`
            if (typeof window !== 'undefined' && typeof window.isSpace === 'undefined') {
              window.isSpace = function(code) {
                return code === 0x20 || code === 0x09 || code === 0x0A || code === 0x0B || code === 0x0C || code === 0x0D;
              };
            }
          `}
        </Script>
      </head>
      <body className="bg-app-background text-foreground">
        <ThemeProviderWrapper>{children}</ThemeProviderWrapper>
        <Toaster />
      </body>
    </html>
  );
}