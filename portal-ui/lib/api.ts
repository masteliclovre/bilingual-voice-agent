import { getAccessToken } from "@auth0/nextjs-auth0";

const API_BASE = process.env.PORTAL_API_BASE ?? "http://localhost:8001";

export async function apiFetch<T>(path: string): Promise<T> {
  const { accessToken } = await getAccessToken();
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  return (await response.json()) as T;
}
