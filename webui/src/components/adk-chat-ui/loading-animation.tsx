// webui/src/components/adk-chat-ui/loading-animation.tsx
import { cn } from "~/lib/utils";
import styles from "./loading-animation.module.css"; // Re-enable custom styles

export function LoadingAnimation({
  className,
  size = "normal",
}: {
  className?: string;
  size?: "normal" | "sm";
}) {
  // console.log("[LoadingAnimation] Rendering ORIGINAL version. className:", className, "size:", size); // You can keep this log for a bit
  return (
    <div
      className={cn(
        styles.loadingAnimation,
        size === "sm" && styles.sm,
        className,
      )}
      data-testid="original-loading-animation" // Add a test ID if you want to find it easily
    >
      <div></div>
      <div></div>
      <div></div>
    </div>
  );
}