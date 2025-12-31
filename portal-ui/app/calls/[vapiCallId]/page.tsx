import { apiFetch } from "@/lib/api";

type CallDetail = {
  vapiCallId: string;
  startedAt: string;
  answeredAt: string;
  endedAt: string;
  durationSec: number;
  answerTimeSec: number;
  direction: string;
  callerMasked: string;
  calledMasked: string;
  outcome: string;
  escalated: boolean;
  reason: string;
  summary: string;
  transcriptText: string;
  recording: { available: boolean; url: string | null };
  structuredData: Record<string, unknown>;
  technical: { endedReason: string; errorCode: string | null };
};

export default async function CallDetailPage({
  params,
}: {
  params: { vapiCallId: string };
}) {
  const call = await apiFetch<CallDetail>(`/calls/${params.vapiCallId}`);

  return (
    <div>
      <h1>Call {call.vapiCallId}</h1>
      <div className="card">
        <p>
          <strong>Outcome:</strong> {call.outcome}
        </p>
        <p>
          <strong>Reason:</strong> {call.reason}
        </p>
        <p>
          <strong>Started:</strong> {call.startedAt}
        </p>
        <p>
          <strong>Duration:</strong> {call.durationSec}s
        </p>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Summary</h3>
        <p>{call.summary}</p>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Transcript</h3>
        <pre style={{ whiteSpace: "pre-wrap" }}>{call.transcriptText}</pre>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Recording</h3>
        {call.recording.available ? (
          <a href={`/api/recordings/${call.vapiCallId}`}>Open recording</a>
        ) : (
          <p>No recording available.</p>
        )}
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Structured Data</h3>
        <pre>{JSON.stringify(call.structuredData, null, 2)}</pre>
      </div>
    </div>
  );
}
