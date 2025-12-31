import "./globals.css";
import { UserProvider } from "@auth0/nextjs-auth0/client";

export const metadata = {
  title: "Voice Agent Portal",
  description: "Ops dashboard for Vapi-powered voice agents.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <UserProvider>
          <div className="layout">
            <aside className="sidebar">
              <h2>Voice Portal</h2>
              <nav>
                <a href="/overview">Overview</a>
                <a href="/outcomes">Outcomes</a>
                <a href="/sla">SLA & Reliability</a>
                <a href="/calls">Calls</a>
              </nav>
            </aside>
            <main className="content">{children}</main>
          </div>
        </UserProvider>
      </body>
    </html>
  );
}
