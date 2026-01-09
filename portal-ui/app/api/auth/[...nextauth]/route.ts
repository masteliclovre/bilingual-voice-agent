import NextAuth, { AuthOptions } from "next-auth";
import GoogleProvider from "next-auth/providers/google";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

export const authOptions: AuthOptions = {
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
  ],
  callbacks: {
    async signIn({ user, account, profile }) {
      // Sync user to backend database
      try {
        const response = await fetch(`${API_URL}/api/auth/sync-user`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            email: user.email,
            full_name: user.name,
            google_id: account?.providerAccountId,
            profile_picture: user.image,
          }),
        });

        if (!response.ok) {
          console.error("Failed to sync user to backend");
          return false;
        }

        const data = await response.json();

        // Attach user_id, approval status and tenant info to session
        if (user) {
          (user as any).user_id = data.user_id;
          (user as any).approval_status = data.approval_status;
          (user as any).tenants = data.tenants;
          (user as any).current_tenant = data.tenants[0]; // Default to first tenant
        }

        return true;
      } catch (error) {
        console.error("Error syncing user:", error);
        return false;
      }
    },
    async jwt({ token, user, trigger, session }) {
      // Initial sign in
      if (user) {
        token.user_id = (user as any).user_id;
        token.approval_status = (user as any).approval_status;
        token.tenants = (user as any).tenants;
        token.current_tenant = (user as any).current_tenant;
      }

      // Handle tenant switch
      if (trigger === "update" && session?.tenant_id) {
        const tenants = token.tenants as any[];
        const newTenant = tenants?.find((t: any) => t.tenant_id === session.tenant_id);
        if (newTenant) {
          token.current_tenant = newTenant;
        }
      }

      return token;
    },
    async session({ session, token }) {
      // Add custom fields to session
      if (session.user) {
        (session.user as any).user_id = token.user_id;
        (session.user as any).approval_status = token.approval_status;
        (session.user as any).tenants = token.tenants;
        (session.user as any).current_tenant = token.current_tenant;
      }
      return session;
    },
  },
  pages: {
    signIn: "/",
    error: "/auth/error",
  },
};

const handler = NextAuth(authOptions);

export { handler as GET, handler as POST };
