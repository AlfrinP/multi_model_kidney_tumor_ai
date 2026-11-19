import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Multi-Modal Kidney Tumor AI",
  description: "Accurate, automated kidney tumor detection, segmentation, and localization using multi-modal AI fusion of CT and MRI scans.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
