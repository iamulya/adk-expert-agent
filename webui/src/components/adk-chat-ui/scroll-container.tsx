"use client";

import React, { useEffect, useImperativeHandle, useRef, type ReactNode, type RefObject } from "react";
import { useStickToBottom } from "use-stick-to-bottom";

import { ScrollArea as ShadcnScrollArea, ScrollBar as ShadcnScrollBar } from "~/components/ui/scroll-area";
import { cn } from "~/lib/utils";

export interface ScrollContainerProps {
  className?: string;
  children?: ReactNode;
  scrollShadow?: boolean;
  scrollShadowColor?: string;
  autoScrollToBottom?: boolean;
}

export interface ScrollContainerRef {
  scrollToBottom(): void;
  scrollToTop(): void;
  getScrollPosition(): number | undefined;
  getViewport(): HTMLDivElement | null;
}

// Custom ScrollArea that accepts a viewportRef
const CustomScrollArea = React.forwardRef<
  HTMLDivElement, // This ref is for the root of ScrollAreaPrimitive.Root
  React.ComponentProps<typeof ShadcnScrollArea> & { viewportRef?: React.Ref<HTMLDivElement> }
>(({ children, viewportRef, ...props }, ref) => {
  return (
    <ShadcnScrollArea {...props} ref={ref}>
      {/* We need to get the actual scrollable viewport. Shadcn's ScrollArea uses a Viewport component internally.
          Ideally, Shadcn's ScrollArea would allow passing a ref to its internal Viewport.
          If not, we might need to query for it or adjust how useStickToBottom is used.
          For now, let's assume the main ScrollArea ref can be used for some scrolling,
          and we'll try to make  work with what we have.
      */}
      <div ref={viewportRef} className="h-full w-full"> {/* This div wraps children and gets viewportRef */}
        {children}
      </div>
      <ShadcnScrollBar orientation="vertical" />
    </ShadcnScrollArea>
  );
});
CustomScrollArea.displayName = "CustomScrollArea";


export const ScrollContainer = React.forwardRef<ScrollContainerRef, ScrollContainerProps>(
  (
    {
      className,
      children,
      scrollShadow = true,
      scrollShadowColor = "hsl(var(--background))", 
      autoScrollToBottom = false,
    },
    ref,
  ) => {
    const { scrollRef: stickToBottomScrollRef, contentRef: stickToBottomContentRef, scrollToBottom, isAtBottom } = useStickToBottom({ initial: "instant" });
    const internalViewportRef = useRef<HTMLDivElement>(null); // Ref for the div inside CustomScrollArea

    useImperativeHandle(ref, () => ({
      scrollToBottom() {
        if (internalViewportRef.current) {
          internalViewportRef.current.scrollTop = internalViewportRef.current.scrollHeight;
        }
      },
      scrollToTop() {
        if (internalViewportRef.current) {
          internalViewportRef.current.scrollTop = 0;
        }
      },
      getScrollPosition() {
        return internalViewportRef.current?.scrollTop;
      },
      getViewport() {
        return internalViewportRef.current;
      }
    }));

    useEffect(() => {
      if (autoScrollToBottom && isAtBottom && internalViewportRef.current) {
         // Use a slight delay to ensure content is rendered
        setTimeout(() => {
          if (internalViewportRef.current) {
            internalViewportRef.current.scrollTop = internalViewportRef.current.scrollHeight;
          }
        }, 0);
      }
    }, [autoScrollToBottom, children, isAtBottom]); // Rerun if children change to scroll down


    return (
      <div className={cn("relative h-full w-full", className)}>
        {scrollShadow && (
          <>
            <div
              className={cn(
                "pointer-events-none absolute top-0 right-0 left-0 z-10 h-8 bg-gradient-to-t",
                `from-transparent to-[${scrollShadowColor}]`, // Corrected template literal
              )}
            ></div>
            <div
              className={cn(
                "pointer-events-none absolute right-0 bottom-0 left-0 z-10 h-8 bg-gradient-to-b",
                `from-transparent to-[${scrollShadowColor}]`, // Corrected template literal
              )}
            ></div>
          </>
        )}
        <CustomScrollArea
          className="h-full w-full"
          ref={stickToBottomScrollRef as RefObject<HTMLDivElement>} // Ref for useStickToBottom
          viewportRef={internalViewportRef} // Pass our ref to the inner div
        >
          <div ref={stickToBottomContentRef as RefObject<HTMLDivElement>}> {/* Ref for useStickToBottom's content div */}
            {children}
          </div>
        </CustomScrollArea>
      </div>
    );
  },
);
ScrollContainer.displayName = "ScrollContainer";
