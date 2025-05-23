import Link from "next/link";

export function Logo() {
  return (
    <Link
      className="font-semibold text-lg opacity-90 transition-opacity duration-300 hover:opacity-100"
      href="/chat"
    >
      🤖 ADK Expert
    </Link>
  );
}
