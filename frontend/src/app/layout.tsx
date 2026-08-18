import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Axiom — Evidence-led claim analysis",
  description:
    "Test factual claims against retrieved sources, contradiction signals, and transparent AI reasoning.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
