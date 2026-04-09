import "./globals.css";
import React from "react";

export const metadata = {
  title: "Truth GNN Analytics - Resgate",
  description: "Portal de Análise de Fake News via GNN",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pt">
      <body>{children}</body>
    </html>
  );
}
