"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from "react";

// Auth types
interface User {
  id: string;
  email: string;
  name: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<boolean>;
  loginWithGoogle: () => Promise<void>; // Placeholder for Google OAuth
  logout: () => void;
  isLoading: boolean;
}

// Create context
const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Mock user for testing - replace with real backend later
const MOCK_USER: User = {
  id: "1",
  email: "admin@ennanext.com",
  name: "Admin User",
};

// Auth Provider Component
export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Check for existing session on mount
  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
    setIsLoading(false);
  }, []);

  // Mock login - replace with real API call later
  const login = async (email: string, password: string): Promise<boolean> => {
    setIsLoading(true);

    // Simulate API call delay
    await new Promise((resolve) => setTimeout(resolve, 500));

    // Mock authentication - REPLACE WITH REAL BACKEND
    // TODO: POST /api/auth/login with { email, password }
    if (email === "admin" && password === "admin") {
      setUser(MOCK_USER);
      localStorage.setItem("user", JSON.stringify(MOCK_USER));
      setIsLoading(false);
      return true;
    }

    setIsLoading(false);
    return false;
  };

  // Google OAuth login - placeholder
  const loginWithGoogle = async () => {
    // TODO: Implement Google OAuth flow
    // 1. Redirect to Google OAuth consent screen
    // 2. Handle callback with auth code
    // 3. Exchange code for tokens
    // 4. Fetch user profile
    // 5. Create/update user in backend
    console.log("Google OAuth not implemented yet");
    throw new Error("Google OAuth not implemented");
  };

  // Logout
  const logout = () => {
    setUser(null);
    localStorage.removeItem("user");
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        login,
        loginWithGoogle,
        logout,
        isLoading,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

// Custom hook to use auth context
export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}
