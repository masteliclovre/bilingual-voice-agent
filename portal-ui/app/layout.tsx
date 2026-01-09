import "./globals.css";
import { SessionProvider } from "@/lib/auth";
import AuthSessionProvider from "@/components/AuthSessionProvider";

export const metadata = {
  title: "ENNA Next - Voice Agent Portal",
  description: "Dashboard za Vapi glasovne agente.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="hr">
      <body>
        <AuthSessionProvider>{children}</AuthSessionProvider>
      </body>
    </html>
  );
}
