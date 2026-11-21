import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Geist } from "next/font/google";
import "./globals.css";

// One typeface for the whole console; the CSS names it first and falls back to Helvetica.
const geist = Geist({ subsets: ["latin"], variable: "--font-geist" });

export const metadata: Metadata = {
  title: "Orientyr · Mission control",
  description: "GNSS-denied drone navigation: the drone's position, found by matching its camera to the map",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={geist.variable}>
      {/* Browser extensions (Grammarly and the like) add attributes to <body> before React starts: not a mismatch of ours. */}
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
