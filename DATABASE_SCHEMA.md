# Database Schema

## Overview
Ova baza podataka je dizajnirana za pohranu podataka o pozivima VAPI glasovnog agenta, transkripcija razgovora i metapodataka o namjerama korisnika.

## Tablice

### 1. `calls`
Glavna tablica za pohranu informacija o pozivima.

```sql
CREATE TABLE calls (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  vapi_call_id VARCHAR(255) UNIQUE NOT NULL,
  phone_number VARCHAR(50) NOT NULL,
  started_at TIMESTAMP WITH TIME ZONE NOT NULL,
  ended_at TIMESTAMP WITH TIME ZONE,
  duration_seconds INTEGER,
  status VARCHAR(50) NOT NULL, -- 'completed', 'failed', 'no-answer', 'busy'
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_calls_started_at ON calls(started_at DESC);
CREATE INDEX idx_calls_phone_number ON calls(phone_number);
CREATE INDEX idx_calls_vapi_call_id ON calls(vapi_call_id);
```

**Polja:**
- `id` - Interni UUID za svaki poziv
- `vapi_call_id` - Jedinstveni ID poziva iz VAPI platforme
- `phone_number` - Broj telefona korisnika
- `started_at` - Vrijeme početka poziva
- `ended_at` - Vrijeme završetka poziva
- `duration_seconds` - Trajanje poziva u sekundama
- `status` - Status poziva (completed, failed, no-answer, busy)
- `created_at` - Vrijeme kreiranja zapisa

### 2. `transcripts`
Tablica za pohranu transkripata razgovora po dijelovima (AI agent i korisnik).

```sql
CREATE TABLE transcripts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  call_id UUID NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
  speaker VARCHAR(50) NOT NULL, -- 'user' ili 'agent'
  text TEXT NOT NULL,
  timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_transcripts_call_id ON transcripts(call_id);
CREATE INDEX idx_transcripts_timestamp ON transcripts(timestamp);
```

**Polja:**
- `id` - Interni UUID
- `call_id` - Foreign key na `calls` tablicu
- `speaker` - Tko govori ('user' ili 'agent')
- `text` - Tekst transkripcije
- `timestamp` - Vrijeme kada je izgovoreno
- `created_at` - Vrijeme kreiranja zapisa

### 3. `call_metadata`
Tablica za pohranu analizirane metapodatke o pozivu, AI sažetke i informacije o prosljeđivanju.

```sql
CREATE TABLE call_metadata (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  call_id UUID NOT NULL UNIQUE REFERENCES calls(id) ON DELETE CASCADE,

  -- AI Summary
  ai_summary TEXT, -- Kratak AI generirani sažetak razgovora (2-3 rečenice)

  -- Classification
  customer_intent VARCHAR(255), -- 'pitanje_o_racunu', 'tehnicki_problem', 'ugovor', etc.
  topics TEXT[], -- Array of topics discussed (npr. ['Solarne ploče', 'Baterije', 'Subvencije'])
  sentiment VARCHAR(50), -- 'positive', 'neutral', 'negative'

  -- Escalation Info
  escalated BOOLEAN DEFAULT FALSE, -- Da li je poziv prosljeđen na ljudsku podršku
  escalation_reason TEXT, -- Razlog prosljeđivanja (npr. "Korisnik traži detaljnu ponudu")
  escalation_priority VARCHAR(50), -- 'high', 'medium', 'low'
  escalation_resolved BOOLEAN DEFAULT FALSE, -- Da li je prosljeđeni poziv riješen
  escalation_resolved_at TIMESTAMP WITH TIME ZONE, -- Kada je riješen
  escalation_notes TEXT, -- Bilješke korisničke službe

  -- Follow-up
  requires_followup BOOLEAN DEFAULT FALSE,
  notes TEXT,

  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_call_metadata_call_id ON call_metadata(call_id);
CREATE INDEX idx_call_metadata_customer_intent ON call_metadata(customer_intent);
CREATE INDEX idx_call_metadata_requires_followup ON call_metadata(requires_followup);
CREATE INDEX idx_call_metadata_escalated ON call_metadata(escalated);
CREATE INDEX idx_call_metadata_escalation_resolved ON call_metadata(escalation_resolved) WHERE escalated = TRUE;
CREATE INDEX idx_call_metadata_escalation_priority ON call_metadata(escalation_priority) WHERE escalated = TRUE;
```

**Polja:**
- `id` - Interni UUID
- `call_id` - Foreign key na `calls` tablicu (jedinstvena veza 1:1)
- `ai_summary` - **NOVO:** AI generirani sažetak cijelog razgovora (prikaže se odmah nakon poziva)
- `customer_intent` - Namjera korisnika (pitanje o računu, tehnički problem, ugovor, itd.)
- `topics` - Array tema o kojima se razgovaralo
- `sentiment` - Sentiment razgovora (pozitivan, neutralan, negativan)
- `escalated` - **NOVO:** Da li je AI prosljedil poziv na ljudsku podršku
- `escalation_reason` - **NOVO:** Detaljni razlog zašto AI nije mogao riješiti
- `escalation_priority` - **NOVO:** Prioritet (high/medium/low) za korisničku službu
- `escalation_resolved` - **NOVO:** Da li je korisnička služba riješila poziv
- `escalation_resolved_at` - **NOVO:** Timestamp kada je označeno kao riješeno
- `escalation_notes` - **NOVO:** Bilješke korisničke službe nakon rješavanja
- `requires_followup` - Zastavica da li poziv zahtijeva nastavak
- `notes` - Dodatne bilješke
- `created_at` - Vrijeme kreiranja zapisa
- `updated_at` - Vrijeme zadnje izmjene

## VAPI Webhook Integracija

### Endpoint: `POST /api/webhooks/vapi`

VAPI će slati webhooks na ovaj endpoint tijekom i nakon poziva.

**Eventi:**
1. `call.started` - Poziv je započeo → Kreiraj zapis u `calls` tablici sa statusom "active"
2. `call.transcript.update` - Nova transkripcija je dostupna → Dodaj u `transcripts` tablicu
3. `call.ended` - Poziv je završio → Update `calls` status na "completed", generiraj AI sažetak
4. `call.escalated` - **NOVO:** AI nije mogao riješiti, prosljeđivanje na ljudsku podršku

---

### 1️⃣ **Call Started Event**
Poziv je započeo, kreirati zapis u bazi.

**Payload:**
```json
{
  "type": "call.started",
  "call": {
    "id": "vapi_call_123456",
    "phoneNumber": "+385911234567",
    "startedAt": "2026-01-07T14:32:00Z"
  }
}
```

**Backend akcija:**
```sql
INSERT INTO calls (vapi_call_id, phone_number, started_at, status)
VALUES ('vapi_call_123456', '+385911234567', '2026-01-07 14:32:00+00', 'active');
```

**Dashboard prikaz:** Poziv se pojavljuje u sekciji **🟢 Aktivni Pozivi**

---

### 2️⃣ **Transcript Update Event**
Nova transkripcija (korisnik ili AI agent govori).

**Payload:**
```json
{
  "type": "call.transcript.update",
  "call": {
    "id": "vapi_call_123456"
  },
  "transcript": {
    "speaker": "user",
    "text": "Dobar dan, trebam detaljnu ponudu za solarni sustav 10kW.",
    "timestamp": "2026-01-07T14:32:15Z"
  }
}
```

**Backend akcija:**
```sql
INSERT INTO transcripts (call_id, speaker, text, timestamp)
SELECT id, 'user', 'Dobar dan, trebam detaljnu ponudu...', '2026-01-07 14:32:15+00'
FROM calls WHERE vapi_call_id = 'vapi_call_123456';
```

---

### 3️⃣ **Call Ended Event**
Poziv je završen - generiraj AI sažetak automatski.

**Payload:**
```json
{
  "type": "call.ended",
  "call": {
    "id": "vapi_call_123456",
    "phoneNumber": "+385911234567",
    "startedAt": "2026-01-07T14:32:00Z",
    "endedAt": "2026-01-07T14:36:23Z",
    "duration": 263,
    "status": "completed"
  }
}
```

**Backend akcija:**
1. Update `calls` tablicu:
```sql
UPDATE calls
SET ended_at = '2026-01-07 14:36:23+00',
    duration_seconds = 263,
    status = 'completed'
WHERE vapi_call_id = 'vapi_call_123456';
```

2. **Generiraj AI sažetak** pozivom GPT-4 API-ja sa svim transkriptima:
```javascript
// Dohvati sve transkripte za ovaj poziv
const transcripts = await getTranscripts('vapi_call_123456');

// GPT-4 prompt
const summary = await openai.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [{
    role: "system",
    content: "Ti si AI asistent koji sumira telefonske razgovore. Sažmi razgovor u 2-3 rečenice."
  }, {
    role: "user",
    content: `Sažmi ovaj razgovor:\n\n${transcripts.map(t => `${t.speaker}: ${t.text}`).join('\n')}`
  }]
});

// Pohrani sažetak i topics
await db.query(`
  INSERT INTO call_metadata (call_id, ai_summary, topics)
  VALUES ($1, $2, $3)
`, [callId, summary, extractedTopics]);
```

**Dashboard prikaz:** Poziv nestaje iz **🟢 Aktivni Pozivi** i pojavljuje se u **📋 Riješeni Pozivi (Danas)** sa AI sažetkom.

---

### 4️⃣ **Call Escalated Event** ⚠️ **KRITIČNO**
AI agent nije mogao riješiti zahtjev - prosljeđuje na ljudsku podršku.

**Payload:**
```json
{
  "type": "call.escalated",
  "call": {
    "id": "vapi_call_123456",
    "phoneNumber": "+385911234567"
  },
  "escalation": {
    "reason": "Korisnik traži detaljnu ponudu za solarni sustav 10kW s baterijama. Zanima ga točan iznos subvencije i rok instalacije. AI agent je dao opće informacije, ali korisnik inzistira na preciznom izračunu i terminu.",
    "priority": "high",
    "topics": ["Solarne ploče", "Baterije", "Subvencije", "Ponuda"]
  }
}
```

**Backend akcija:**
```sql
UPDATE call_metadata
SET escalated = TRUE,
    escalation_reason = 'Korisnik traži detaljnu ponudu...',
    escalation_priority = 'high',
    topics = ARRAY['Solarne ploče', 'Baterije', 'Subvencije', 'Ponuda']
WHERE call_id = (SELECT id FROM calls WHERE vapi_call_id = 'vapi_call_123456');
```

**Dashboard prikaz:** Poziv se pojavljuje u **🔴 Pozivi za Korisničku Službu** sa:
- Crvenim borderom (visoka prioriteta)
- AI sažetkom zašto nije mogao riješiti
- Topic tags
- Akcijskim gumbima ("Nazovi korisnika", "Pročitaj transkript", "Označi riješenim")

**Real-time notifikacija:** Poslati obavijest korisničkoj službi (email/SMS/browser notification)

---

### 5️⃣ **Dashboard Query Examples**

**Dohvati prosljeđene pozive (nerazriješene):**
```sql
SELECT c.*, cm.ai_summary, cm.escalation_reason, cm.escalation_priority, cm.topics
FROM calls c
JOIN call_metadata cm ON c.id = cm.call_id
WHERE cm.escalated = TRUE
  AND cm.escalation_resolved = FALSE
ORDER BY cm.escalation_priority DESC, c.started_at DESC;
```

**Dohvati aktivne pozive:**
```sql
SELECT vapi_call_id, phone_number, started_at
FROM calls
WHERE status = 'active'
ORDER BY started_at DESC;
```

**Dohvati riješene pozive (danas):**
```sql
SELECT c.*, cm.ai_summary
FROM calls c
LEFT JOIN call_metadata cm ON c.id = cm.call_id
WHERE c.status = 'completed'
  AND c.started_at >= CURRENT_DATE
  AND (cm.escalated = FALSE OR cm.escalated IS NULL)
ORDER BY c.started_at DESC
LIMIT 20;
```

## Backend API Endpoints (za implementaciju)

### 1. `GET /api/calls`
Dohvaća listu poziva.

**Query parametri:**
- `page` - Broj stranice (default: 1)
- `limit` - Broj rezultata po stranici (default: 20)
- `status` - Filter po statusu
- `phone_number` - Filter po broju telefona

**Response:**
```json
{
  "data": [
    {
      "id": "uuid",
      "vapi_call_id": "vapi_call_123456",
      "phone_number": "+385911234567",
      "started_at": "2026-01-07T14:32:00Z",
      "ended_at": "2026-01-07T14:36:23Z",
      "duration_seconds": 263,
      "status": "completed"
    }
  ],
  "total": 247,
  "page": 1,
  "limit": 20
}
```

### 2. `GET /api/calls/:id`
Dohvaća detalje pojedinog poziva uključujući transkripte i metadata.

**Response:**
```json
{
  "call": {
    "id": "uuid",
    "vapi_call_id": "vapi_call_123456",
    "phone_number": "+385911234567",
    "started_at": "2026-01-07T14:32:00Z",
    "ended_at": "2026-01-07T14:36:23Z",
    "duration_seconds": 263,
    "status": "completed"
  },
  "transcripts": [
    {
      "speaker": "agent",
      "text": "Dobar dan, ENNA Next ovdje. Kako vam mogu pomoći?",
      "timestamp": "2026-01-07T14:32:05Z"
    },
    {
      "speaker": "user",
      "text": "Dobar dan, imam pitanje o računu.",
      "timestamp": "2026-01-07T14:32:15Z"
    }
  ],
  "metadata": {
    "customer_intent": "pitanje_o_racunu",
    "topics": ["račun", "plaćanje"],
    "sentiment": "neutral",
    "requires_followup": false,
    "notes": ""
  }
}
```

### 3. `GET /api/stats`
Dohvaća statistiku poziva.

**Response:**
```json
{
  "total_calls": 247,
  "successful_calls": 231,
  "active_calls": 3,
  "failed_calls": 16,
  "avg_duration_seconds": 245
}
```

## Napomene

- Koristiti PostgreSQL bazu podataka
- Svi timestampovi su u UTC timezone
- Za produkciju dodati RLS (Row Level Security) za multi-tenant arhitekturu
- Razmotriti particioniranje `calls` tablice po datumu za bolje performanse
- Dodati periodic job za analizu sentiment i customer intent iz transkripata
