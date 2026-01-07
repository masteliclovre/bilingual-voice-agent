import "./globals.css";
import { AuthProvider } from "@/lib/auth";

export const metadata = {
  title: "ENNA Next - Voice Agent Portal",
  description: "Dashboard za Vapi glasovne agente.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="hr">
      <body>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
