import type { Metadata } from "next";
import { Geist, Geist_Mono, Playfair_Display } from "next/font/google";
import { SuppressHydrationWarning } from "@/components/SuppressHydrationWarning";
import { ScrollProgress } from "@/components/ScrollProgress";
import { Footer } from "@/components/Footer";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const playfairDisplay = Playfair_Display({
  variable: "--font-playfair",
  subsets: ["vietnamese", "latin"],
  weight: ["400", "700", "900"],
});

export const metadata: Metadata = {
  title: "Geometric ML Foundations — Demo",
  description: "Demo tương tác 3 hướng nghiên cứu Geometric Machine Learning: bất biến xoay SO(2), phân loại đám mây điểm 3D với PointNet, và dự đoán năng lượng phân tử với NequIP.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="vi"
      suppressHydrationWarning
      className={`${geistSans.variable} ${geistMono.variable} ${playfairDisplay.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col" suppressHydrationWarning>
        <SuppressHydrationWarning />
        <ScrollProgress />
        {children}
        <Footer />
      </body>
    </html>
  );
}
