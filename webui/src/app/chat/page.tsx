"use client";
import dynamic from "next/dynamic";
import { Suspense } from "react";
import { SiteHeader } from "./components/site-header";

const Main = dynamic(() => import("./main"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center">
      Loading ADK Expert Agent...
    </div>
  ),
});

export default function ChatPage() {
  return (
    <div className="flex h-screen w-screen flex-col items-center overscroll-none">
      <SiteHeader />
      <Suspense fallback={<div>Loading...</div>}>
        <Main />
      </Suspense>
    </div>
  );
}
