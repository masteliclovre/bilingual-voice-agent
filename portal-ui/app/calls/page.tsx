import { apiFetch } from "@/lib/api";

type CallsResponse = {
  page: number;
  pageSize: number;
  total: number;
  items: {
    vapiCallId: string;
    startedAt: string;
    durationSec: number;
    outcome: string;
    reason: string;
    summary: string;
    callerMasked: string;
  }[];
};

function buildRange() {
  const to = new Date();
  const from = new Date();
  from.setDate(to.getDate() - 7);
  return { from: from.toISOString(), to: to.toISOString() };
}

export default async function CallsPage() {
  const { from, to } = buildRange();
  const calls = await apiFetch<CallsResponse>(`/calls?from=${from}&to=${to}&page=1&pageSize=20`);

  return (
    <div>
      <h1>Calls</h1>
      <table className="table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Caller</th>
            <th>Duration</th>
            <th>Outcome</th>
            <th>Reason</th>
            <th>Summary</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          {calls.items.map((call) => (
            <tr key={call.vapiCallId}>
              <td>{call.startedAt}</td>
              <td>{call.callerMasked}</td>
              <td>{call.durationSec}s</td>
              <td>
                <span className={`badge ${call.outcome}`}>{call.outcome}</span>
              </td>
              <td>{call.reason}</td>
              <td>{call.summary}</td>
              <td>
                <a href={`/calls/${call.vapiCallId}`}>Open</a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
