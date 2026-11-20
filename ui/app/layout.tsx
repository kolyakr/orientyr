import type { Metadata } from "next";
import type { ReactNode } from "react";
import { B612, B612_Mono } from "next/font/google";
import "./globals.css";

// B612 is the family drawn for aircraft cockpit displays: the text face for words, the mono face for every number,
// so figures keep their place while they change. Next downloads both at build time and serves them from this app.
const text = B612({ subsets: ["latin"], weight: ["400", "700"], variable: "--font-text" });
const mono = B612_Mono({ subsets: ["latin"], weight: ["400", "700"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "Orientyr · Mission control",
  description: "GNSS-denied drone navigation: the drone's position, found by matching its camera to the map",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${text.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
