import { apiFetch } from "@/lib/api";

type Reasons = {
  items: {
    reason: string;
    calls: number;
    resolvedPct: number;
    escalatedPct: number;
    avgDurationSec: number;
  }[];
};

type Impact = {
  humanMinutesSaved: number;
  estimatedCostSaved: number;
  assumptions: {
    humanMinutesPerResolvedCall: number;
    costPerMinute: number;
    currency: string;
  };
};

function buildRange() {
  const to = new Date();
  const from = new Date();
  from.setDate(to.getDate() - 7);
  return { from: from.toISOString(), to: to.toISOString() };
}

export default async function OutcomesPage() {
  const { from, to } = buildRange();
  const reasons = await apiFetch<Reasons>(`/kpi/reasons?from=${from}&to=${to}&limit=10`);
  const impact = await apiFetch<Impact>(`/kpi/impact?from=${from}&to=${to}`);

  return (
    <div>
      <h1>Outcomes & Reasons</h1>
      <table className="table">
        <thead>
          <tr>
            <th>Reason</th>
            <th>Calls</th>
            <th>Resolved %</th>
            <th>Escalated %</th>
            <th>Avg Duration</th>
          </tr>
        </thead>
        <tbody>
          {reasons.items.map((item) => (
            <tr key={item.reason}>
              <td>{item.reason}</td>
              <td>{item.calls}</td>
              <td>{item.resolvedPct}%</td>
              <td>{item.escalatedPct}%</td>
              <td>{item.avgDurationSec}s</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="card" style={{ marginTop: 24 }}>
        <h3>Automation Impact</h3>
        <p>Human minutes saved: {impact.humanMinutesSaved}</p>
        <p>
          Estimated cost saved: {impact.estimatedCostSaved} {impact.assumptions.currency}
        </p>
        <small>
          Assumptions: {impact.assumptions.humanMinutesPerResolvedCall} min/call @{" "}
          {impact.assumptions.costPerMinute} {impact.assumptions.currency}/min
        </small>
      </div>
    </div>
  );
}
