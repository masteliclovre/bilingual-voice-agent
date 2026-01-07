import { getAccessToken } from "@auth0/nextjs-auth0";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";
const USE_MOCK_DATA = process.env.NEXT_PUBLIC_USE_MOCK_DATA === "true";

// Mock data for local testing without backend
const mockData: Record<string, any> = {
  "/kpi/overview": {
    totalCalls: 247,
    aiAnswerRatePct: 94.3,
    resolutionRatePct: 78.5,
    escalationRatePct: 12.1,
    failedCalls: 8,
    avgAnswerTimeSec: 2.4,
    p95AnswerTimeSec: 4.2,
    slaCompliancePct: 96.8,
  },
  "/kpi/overview/trend": {
    points: [
      { day: "2026-01-01", total: 32, resolved: 25, escalated: 4, failed: 2, abandoned: 1 },
      { day: "2026-01-02", total: 38, resolved: 31, escalated: 5, failed: 1, abandoned: 1 },
      { day: "2026-01-03", total: 29, resolved: 23, escalated: 3, failed: 2, abandoned: 1 },
      { day: "2026-01-04", total: 41, resolved: 33, escalated: 5, failed: 2, abandoned: 1 },
      { day: "2026-01-05", total: 35, resolved: 28, escalated: 4, failed: 1, abandoned: 2 },
      { day: "2026-01-06", total: 37, resolved: 29, escalated: 6, failed: 1, abandoned: 1 },
      { day: "2026-01-07", total: 35, resolved: 27, escalated: 5, failed: 2, abandoned: 1 },
    ],
  },
  "/calls": {
    page: 1,
    pageSize: 20,
    total: 35,
    items: [
      {
        vapiCallId: "call-001",
        startedAt: "2026-01-07T14:32:15Z",
        durationSec: 127,
        outcome: "resolved",
        reason: "Pitanje o solarnim panelima",
        summary: "Klijent pitao o cijeni solarnih panela za obiteljsku kuću",
        callerMasked: "+385**345791",
      },
      {
        vapiCallId: "call-002",
        startedAt: "2026-01-07T13:18:42Z",
        durationSec: 89,
        outcome: "escalated",
        reason: "Tehnički zahtjev - detaljna ponuda",
        summary: "Proslijeđeno timu za pripremu personalizirane ponude",
        callerMasked: "+385**123456",
      },
      {
        vapiCallId: "call-003",
        startedAt: "2026-01-07T12:05:33Z",
        durationSec: 45,
        outcome: "resolved",
        reason: "Opće informacije o ENNA Next",
        summary: "Informacije o tvrtki i uslugama",
        callerMasked: "+385**987654",
      },
      {
        vapiCallId: "call-004",
        startedAt: "2026-01-07T11:22:11Z",
        durationSec: 156,
        outcome: "resolved",
        reason: "Pitanje o baterijskim sustavima",
        summary: "Razgovor o skladištenju energije i baterijama",
        callerMasked: "+385**555222",
      },
      {
        vapiCallId: "call-005",
        startedAt: "2026-01-07T10:47:58Z",
        durationSec: 12,
        outcome: "failed",
        reason: "Veza prekinuta",
        summary: "Tehnički problem - poziv prekinut",
        callerMasked: "+385**777888",
      },
    ],
  },
  "/kpi/reasons": {
    items: [
      {
        reason: "Pitanje o solarnim panelima",
        calls: 89,
        resolvedPct: 82.0,
        escalatedPct: 15.7,
        avgDurationSec: 142,
      },
      {
        reason: "Tehnički zahtjev - detaljna ponuda",
        calls: 52,
        resolvedPct: 23.1,
        escalatedPct: 73.1,
        avgDurationSec: 95,
      },
      {
        reason: "Pitanje o baterijskim sustavima",
        calls: 38,
        resolvedPct: 84.2,
        escalatedPct: 13.2,
        avgDurationSec: 158,
      },
      {
        reason: "Opće informacije o ENNA Next",
        calls: 34,
        resolvedPct: 97.1,
        escalatedPct: 2.9,
        avgDurationSec: 67,
      },
      {
        reason: "Pitanje o subvencijama",
        calls: 21,
        resolvedPct: 61.9,
        escalatedPct: 33.3,
        avgDurationSec: 112,
      },
      {
        reason: "Upit o instalaciji",
        calls: 13,
        resolvedPct: 69.2,
        escalatedPct: 23.1,
        avgDurationSec: 128,
      },
    ],
  },
  "/kpi/impact": {
    humanMinutesSaved: 1847,
    estimatedCostSaved: 9235,
    assumptions: {
      humanMinutesPerResolvedCall: 10,
      costPerMinute: 5,
      currency: "EUR",
    },
  },
  "/kpi/sla": {
    slaTargetSec: 5,
    slaCompliancePct: 96.8,
    within5SecPct: 96.8,
    within10SecPct: 99.2,
    within20SecPct: 100.0,
    sampleSizeAnswered: 239,
  },
  "/kpi/reliability": {
    failedCalls: 8,
    erroredCalls: 3,
    escalations: 30,
    transferFailures: 1,
  },
};

export async function apiFetch<T>(path: string): Promise<T> {
  // If mock mode is enabled, return mock data
  if (USE_MOCK_DATA) {
    await new Promise((resolve) => setTimeout(resolve, 300)); // Simulate network delay

    // Extract base path without query params for mock lookup
    const basePath = path.split("?")[0];

    // Handle dynamic call detail paths like /calls/call-001
    if (basePath.startsWith("/calls/call-")) {
      const callId = basePath.split("/calls/")[1];
      return {
        vapiCallId: callId,
        startedAt: "2026-01-07T14:32:15Z",
        answeredAt: "2026-01-07T14:32:17Z",
        endedAt: "2026-01-07T14:34:22Z",
        durationSec: 127,
        answerTimeSec: 2,
        direction: "inbound",
        callerMasked: "+385**345791",
        calledMasked: "+385**784641",
        outcome: "resolved",
        escalated: false,
        reason: "Pitanje o solarnim panelima",
        summary: "Klijent je pitao o cijeni i instalaciji solarnih panela za obiteljsku kuću. Agent je pružio osnovne informacije o ENNA Next uslugama i napomenuo da će detaljna ponuda biti proslijeđena prodajnom timu.",
        transcriptText: `Agent: Bok! ENNA Next, kako vam mogu pomoći?
Klijent: Dobar dan, zanima me koliko bi me koštali solarni paneli za kuću.
Agent: Razumijem. Prije svega, radi se o obiteljskoj kući?
Klijent: Da, da. Imam kuću od stotinjak kvadrata.
Agent: U redu. Cijena solarnih panela ovisi o više faktora - veličini sustava, vrsti panela i specifičnostima instalacije. Za točan odgovor i personaliziranu ponudu, vaš upit ću proslijediti našem timu.
Klijent: Okej, to mi je jasno. Koliko dugo traje instalacija obično?
Agent: Sama instalacija solarnih panela najčešće traje 1-3 dana, ovisno o kompleksnosti sustava. Ali za precizne informacije, najbolje da vas kontaktira naš tehnički tim.
Klijent: Super, hvala vam puno.
Agent: Nema na čemu! Naš tim će vas kontaktirati uskoro. Ugodan dan!`,
        recording: {
          available: false,
          url: null,
        },
        structuredData: {
          topic: "solar_panels",
          houseSize: "100sqm",
          customerIntent: "purchase_inquiry",
        },
        technical: {
          endedReason: "customer-ended-call",
          errorCode: null,
        },
      } as T;
    }

    const mockResponse = mockData[basePath];

    if (mockResponse) {
      return mockResponse as T;
    }

    throw new Error(`No mock data for path: ${path}`);
  }

  // Real API call
  // TEMP: Disable auth for testing
  // const { accessToken } = await getAccessToken();
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      // Authorization: `Bearer ${accessToken}`,
    },
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  return (await response.json()) as T;
}
