import { Check, Copy } from "lucide-react";
import { useMemo, useState } from "react";
import ReactMarkdown, { type Options as ReactMarkdownOptions } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import { Button } from "~/components/ui/button";
import { cn } from "~/lib/utils";
import ImageComponent from "./image"; 
import { Tooltip } from "./tooltip";
import { Link } from "./link";

export function Markdown({
  className,
  children,
  style,
  enableCopy = true,
  ...props
}: ReactMarkdownOptions & {
  className?: string;
  enableCopy?: boolean;
  style?: React.CSSProperties;
}) {
  const components: ReactMarkdownOptions["components"] = useMemo(() => {
    return {
      a: ({ href, children: linkChildren }) => (
        <Link href={href}>{linkChildren}</Link>
      ),
      img: ({ src, alt }) => (
        <a href={src as string} target="_blank" rel="noopener noreferrer">
          <ImageComponent className="my-2 rounded-md border" src={src as string} alt={alt ?? ""} />
        </a>
      ),
    };
  }, []);

  const rehypePlugins = useMemo(() => [rehypeKatex], []);

  return (
    <div className={cn(className, "prose prose-sm dark:prose-invert max-w-none prose-p:my-2 prose-li:my-0.5")} style={style}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={rehypePlugins}
        components={components}
        {...props}
      >
        {children?.toString() ?? ""}
      </ReactMarkdown>
      {enableCopy && typeof children === "string" && children.trim().length > 0 && (
        <div className="mt-1 flex">
          <CopyButton content={children} />
        </div>
      )}
    </div>
  );
}

function CopyButton({ content }: { content: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Tooltip title="Copy to clipboard">
      <Button
        variant="ghost"
        size="icon"
        className="text-muted-foreground hover:text-foreground h-6 w-6 p-1"
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(content);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          } catch (error) {
            console.error("Failed to copy:",error);
          }
        }}
      >
        {copied ? (
          <Check className="h-3.5 w-3.5" />
        ) : (
          <Copy className="h-3.5 w-3.5" />
        )}
      </Button>
    </Tooltip>
  );
}
