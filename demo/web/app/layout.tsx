import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Personalized Knowledge Graph — Adobe × UpSync",
  description:
    "Live demo of the Graphiti-powered personalized knowledge graph: episodes are logged, entities are extracted and linked to a user profile, and personalized context is surfaced in the answer.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
