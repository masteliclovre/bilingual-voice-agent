import { apiFetch } from "@/lib/api";

type Overview = {
  totalCalls: number;
  aiAnswerRatePct: number;
  resolutionRatePct: number;
  escalationRatePct: number;
  failedCalls: number;
  avgAnswerTimeSec: number;
  p95AnswerTimeSec: number;
  slaCompliancePct: number;
};

type Trend = {
  points: { day: string; total: number; resolved: number; escalated: number; failed: number; abandoned: number }[];
};

function buildRange() {
  const to = new Date();
  const from = new Date();
  from.setDate(to.getDate() - 7);
  return { from: from.toISOString(), to: to.toISOString() };
}

export default async function OverviewPage() {
  const { from, to } = buildRange();
  const overview = await apiFetch<Overview>(`/kpi/overview?from=${from}&to=${to}`);
  const trend = await apiFetch<Trend>(`/kpi/overview/trend?from=${from}&to=${to}`);

  return (
    <div>
      <h1>Overview</h1>
      <div className="card-grid">
        <div className="card">
          <div>Total Calls</div>
          <strong>{overview.totalCalls}</strong>
        </div>
        <div className="card">
          <div>AI Answer Rate</div>
          <strong>{overview.aiAnswerRatePct}%</strong>
        </div>
        <div className="card">
          <div>Resolution Rate</div>
          <strong>{overview.resolutionRatePct}%</strong>
        </div>
        <div className="card">
          <div>Escalation Rate</div>
          <strong>{overview.escalationRatePct}%</strong>
        </div>
        <div className="card">
          <div>Failed Calls</div>
          <strong>{overview.failedCalls}</strong>
        </div>
        <div className="card">
          <div>Avg Answer Time</div>
          <strong>{overview.avgAnswerTimeSec}s</strong>
        </div>
        <div className="card">
          <div>P95 Answer Time</div>
          <strong>{overview.p95AnswerTimeSec}s</strong>
        </div>
        <div className="card">
          <div>SLA Compliance</div>
          <strong>{overview.slaCompliancePct}%</strong>
        </div>
      </div>

      <div className="card" style={{ marginTop: 24 }}>
        <h3>Calls per day (last 7 days)</h3>
        <ul>
          {trend.points.map((point) => (
            <li key={point.day}>
              {point.day}: {point.total} total ({point.resolved} resolved, {point.escalated} escalated,{" "}
              {point.failed} failed)
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
