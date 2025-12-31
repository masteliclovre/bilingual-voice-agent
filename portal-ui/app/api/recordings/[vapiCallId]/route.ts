import { NextResponse } from "next/server";
import { apiFetch } from "@/lib/api";

export async function GET(
  _request: Request,
  { params }: { params: { vapiCallId: string } }
) {
  const data = await apiFetch<{ url: string | null }>(`/calls/${params.vapiCallId}/recording`);
  if (!data.url) {
    return NextResponse.json({ error: "Recording unavailable" }, { status: 404 });
  }
  return NextResponse.redirect(data.url);
}
