import { apiFetch } from "@/lib/api";

type Sla = {
  slaTargetSec: number;
  slaCompliancePct: number;
  within5SecPct: number;
  within10SecPct: number;
  within20SecPct: number;
  sampleSizeAnswered: number;
};

type Reliability = {
  failedCalls: number;
  erroredCalls: number;
  escalations: number;
  transferFailures: number;
};

function buildRange() {
  const to = new Date();
  const from = new Date();
  from.setDate(to.getDate() - 7);
  return { from: from.toISOString(), to: to.toISOString() };
}

export default async function SlaPage() {
  const { from, to } = buildRange();
  const sla = await apiFetch<Sla>(`/kpi/sla?from=${from}&to=${to}`);
  const reliability = await apiFetch<Reliability>(`/kpi/reliability?from=${from}&to=${to}`);

  return (
    <div>
      <h1>SLA & Reliability</h1>
      <div className="card-grid">
        <div className="card">
          <div>SLA Compliance</div>
          <strong>{sla.slaCompliancePct}%</strong>
        </div>
        <div className="card">
          <div>Within 5s</div>
          <strong>{sla.within5SecPct}%</strong>
        </div>
        <div className="card">
          <div>Within 10s</div>
          <strong>{sla.within10SecPct}%</strong>
        </div>
        <div className="card">
          <div>Within 20s</div>
          <strong>{sla.within20SecPct}%</strong>
        </div>
      </div>

      <div className="card" style={{ marginTop: 24 }}>
        <h3>Reliability</h3>
        <p>Failed calls: {reliability.failedCalls}</p>
        <p>Errored calls: {reliability.erroredCalls}</p>
        <p>Escalations: {reliability.escalations}</p>
        <p>Transfer failures: {reliability.transferFailures}</p>
      </div>
    </div>
  );
}
