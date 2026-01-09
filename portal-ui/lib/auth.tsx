"use client";

import { useSession, signIn, signOut } from "next-auth/react";

// Re-export session hook for convenience
export { SessionProvider } from "next-auth/react";

// Extended user type with tenant info
export interface TenantInfo {
  tenant_id: string;
  tenant_name: string;
  role: string;
}

export interface ExtendedUser {
  user_id: string;
  email: string;
  name: string;
  image?: string;
  tenants: TenantInfo[];
  current_tenant: TenantInfo;
}

// Auth hook that wraps NextAuth useSession
export function useAuth() {
  const { data: session, status, update } = useSession();

  // Debug logging with proper expansion
  console.log("useAuth - Session Status:", status);
  console.log("useAuth - Session User Details:", {
    name: session?.user?.name,
    email: session?.user?.email,
    user_id: (session?.user as any)?.user_id,
    approval_status: (session?.user as any)?.approval_status,
    tenants: (session?.user as any)?.tenants,
    current_tenant: (session?.user as any)?.current_tenant,
  });

  const user = session?.user as ExtendedUser | undefined;

  return {
    user: user || null,
    isAuthenticated: status === "authenticated",
    isLoading: status === "loading",
    loginWithGoogle: () => signIn("google"),
    logout: () => signOut({ callbackUrl: "/" }),
    switchTenant: async (tenant_id: string) => {
      await update({ tenant_id });
    },
    currentTenant: user?.current_tenant || null,
    tenants: user?.tenants || [],
  };
}
