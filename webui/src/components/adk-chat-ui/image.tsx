import { memo, useCallback, useEffect, useState } from "react";
import { cn } from "~/lib/utils";
import { Tooltip } from "./tooltip";

function ImageComponent({ 
  className,
  src,
  alt,
  fallback = null,
}: {
  className?: string;
  src: string;
  alt: string;
  fallback?: React.ReactNode;
}) {
  const [, setIsLoading] = useState(true);
  const [isError, setIsError] = useState(false);

  useEffect(() => {
    setIsError(false);
    setIsLoading(true);
  }, [src]);

  const handleLoad = useCallback(() => {
    setIsError(false);
    setIsLoading(false);
  }, []);

  const handleError = useCallback(
    (e: React.SyntheticEvent<HTMLImageElement>) => {
      e.currentTarget.style.display = "none";
      console.warn(`Image "${e.currentTarget.src}" failed to load`);
      setIsError(true);
    },
    [],
  );
  return (
    <span className={cn("block w-fit overflow-hidden", className)}>
      {isError || !src ? (
        fallback
      ) : (
        <Tooltip title={alt ?? "Image"}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            className={cn("size-full object-contain", className)}
            src={src}
            alt={alt}
            onLoad={handleLoad}
            onError={handleError}
          />
        </Tooltip>
      )}
    </span>
  );
}
export default memo(ImageComponent);
