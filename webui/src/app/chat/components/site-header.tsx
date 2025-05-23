import { Github } from "lucide-react";
import Link from "next/link";
import { Button } from "~/components/ui/button";
import { ThemeToggle } from "~/components/adk-chat-ui/theme-toggle";
import { Logo } from "~/components/adk-chat-ui/logo";
import { Tooltip } from "~/components/adk-chat-ui/tooltip";

export async function SiteHeader() {
  return (
    <header className="fixed top-0 left-0 z-40 flex h-12 w-full items-center justify-between border-b bg-background/80 px-4 backdrop-blur-lg">
      <Logo />
      <div className="flex items-center">
        <Tooltip title="View on GitHub">
          <Button variant="ghost" size="icon" asChild>
            <Link
              href="https://github.com/iamulya/iamulya-adk-expert-agent" 
              target="_blank"
            >
              <Github className="h-5 w-5" />
            </Link>
          </Button>
        </Tooltip>
        <ThemeToggle />
      </div>
    </header>
  );
}
