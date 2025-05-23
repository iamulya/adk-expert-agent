import { cn } from "~/lib/utils";

export const Link = ({
  href,
  children,
}: {
  href: string | undefined;
  children: React.ReactNode;
}) => {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={cn("text-brand hover:underline")}
    >
      {children}
    </a>
  );
};
