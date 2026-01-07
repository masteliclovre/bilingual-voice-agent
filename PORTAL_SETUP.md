# VAPI Call Portal - Setup Guide

Portal za praćenje poziva koje prima VAPI glasovni agent. Sastoji se od Flask API-ja i Next.js frontend aplikacije.

## 📁 Struktura

```
bilingual-voice-agent/
├── portal-api/          # Flask backend
│   ├── server.py
│   ├── requirements.txt
│   └── .env
└── portal-ui/           # Next.js frontend
    ├── app/
    ├── components/
    ├── lib/
    └── .env.local
```

---

## 🗄️ PostgreSQL Baza - Setup

### 1. Instalacija PostgreSQL

**Windows:** Preuzmi sa https://www.postgresql.org/download/windows/

**Linux/Mac:**
```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib

# Mac
brew install postgresql
```

### 2. Kreiranje Baze

```bash
# Uloguj se u PostgreSQL
psql -U postgres

# Kreiraj bazu
CREATE DATABASE voice_portal;

# Spoji se na bazu
\c voice_portal
```

### 3. Kreiraj Tablice

```sql
-- Tablica poziva
CREATE TABLE calls (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  vapi_call_id VARCHAR(255) UNIQUE NOT NULL,
  phone_number VARCHAR(50) NOT NULL,
  started_at TIMESTAMP WITH TIME ZONE NOT NULL,
  ended_at TIMESTAMP WITH TIME ZONE,
  duration_seconds INTEGER,
  status VARCHAR(50) NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_calls_started_at ON calls(started_at DESC);
CREATE INDEX idx_calls_vapi_call_id ON calls(vapi_call_id);

-- Tablica transkripcija
CREATE TABLE transcripts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  call_id UUID NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
  speaker VARCHAR(50) NOT NULL,
  text TEXT NOT NULL,
  timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_transcripts_call_id ON transcripts(call_id);

-- Tablica metapodataka
CREATE TABLE call_metadata (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  call_id UUID NOT NULL UNIQUE REFERENCES calls(id) ON DELETE CASCADE,
  ai_summary TEXT,
  customer_intent VARCHAR(255),
  topics TEXT[],
  sentiment VARCHAR(50),
  escalated BOOLEAN DEFAULT FALSE,
  escalation_reason TEXT,
  escalation_priority VARCHAR(50),
  escalation_resolved BOOLEAN DEFAULT FALSE,
  escalation_resolved_at TIMESTAMP WITH TIME ZONE,
  escalation_notes TEXT,
  requires_followup BOOLEAN DEFAULT FALSE,
  notes TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_call_metadata_call_id ON call_metadata(call_id);
CREATE INDEX idx_call_metadata_escalated ON call_metadata(escalated);
```

### 4. Podesi Port i Lozinku (ako treba)

**Promjena porta:**
```bash
# Otvori postgres config
sudo nano /etc/postgresql/15/main/postgresql.conf

# Promijeni liniju
port = 5433

# Restart servisa
sudo systemctl restart postgresql
```

**Promjena lozinke:**
```sql
ALTER USER postgres PASSWORD 'NovaJakaLozinka';
```

---

## 🔧 Flask API - Setup

### 1. Instaliraj Dependencies

```bash
cd portal-api
pip install -r requirements.txt
```

**requirements.txt:**
```
flask==3.0.0
flask-cors==4.0.0
psycopg[binary]==3.1.18
python-dotenv==1.0.0
```

### 2. Napravi `.env` fajl

```bash
# portal-api/.env
DATABASE_URL=postgresql://postgres:NovaJakaLozinka@localhost:5433/voice_portal
```

**Syntax:** `postgresql://korisnik:lozinka@host:port/ime_baze`

### 3. Pokreni API

```bash
cd portal-api
python server.py
```

API će biti dostupan na: `http://localhost:5000`

**Endpointi:**
- `GET /api/calls` - Svi pozivi (paginirano)
- `GET /api/calls/active` - Aktivni pozivi
- `GET /api/calls/completed` - Završeni pozivi
- `GET /api/calls/forwarded` - Prosljeđeni pozivi
- `GET /api/calls/<id>/transcript` - Transkript poziva
- `POST /api/webhooks/vapi` - VAPI webhook endpoint

---

## 🌐 Next.js Frontend - Setup

### 1. Instaliraj Dependencies

```bash
cd portal-ui
npm install
```

### 2. Napravi `.env.local` fajl

```bash
# portal-ui/.env.local
NEXT_PUBLIC_API_URL=http://localhost:5000
NEXT_PUBLIC_USE_MOCK_DATA=false
```

### 3. Pokreni Frontend

```bash
npm run dev
```

Frontend će biti dostupan na: `http://localhost:3000`

**Stranice:**
- `/dashboard` - Dashboard sa aktivnim/završenim pozivima
- `/calls` - Lista svih poziva

---

## 🔗 VAPI Webhook - Ngrok Setup

Ngrok omogućava da VAPI šalje webhooks na tvoj lokalni Flask API.

### 1. Instaliraj Ngrok

**Windows:** Preuzmi sa https://ngrok.com/download

**Linux/Mac:**
```bash
# Mac
brew install ngrok

# Linux
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list \
  && sudo apt update && sudo apt install ngrok
```

### 2. Authentifikacija

```bash
ngrok config add-authtoken <TVOJ_TOKEN>
```

Token dobiješ na: https://dashboard.ngrok.com/get-started/your-authtoken

### 3. Pokreni Ngrok Tunnel

```bash
ngrok http 5000
```

**Output:**
```
Forwarding  https://reformative-pseudobenevolent-darla.ngrok-free.dev -> http://localhost:5000
```

Kopiraj HTTPS URL (npr. `https://reformative-pseudobenevolent-darla.ngrok-free.dev`)

### 4. Konfiguriraj VAPI Webhook

**Pomoću curl-a:**
```bash
curl -X PATCH https://api.vapi.ai/assistant/b128ee37-46ef-4937-9f56-ff0695f90a4a \
  -H "Authorization: Bearer <VAPI_PRIVATE_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "serverUrl": "https://tvoj-ngrok-url.ngrok-free.dev/api/webhooks/vapi"
  }'
```

Zamijeni:
- `<VAPI_PRIVATE_KEY>` sa tvojim VAPI privatnim ključem
- `b128ee37-46ef-4937-9f56-ff0695f90a4a` sa tvojim Assistant ID-em
- `tvoj-ngrok-url.ngrok-free.dev` sa ngrok URL-om

---

## 📞 Test Poziva - Curl Command

Napravi testni outbound poziv preko VAPI-ja:

```bash
curl -X POST https://api.vapi.ai/call/phone \
  -H "Authorization: Bearer <VAPI_PRIVATE_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "phoneNumberId": "73c45fc7-7930-4437-af9b-230f3567e047",
    "customer": {
      "number": "+385919345791"
    },
    "assistantId": "b128ee37-46ef-4937-9f56-ff0695f90a4a",
    "name": "Test poziv"
  }'
```

Zamijeni:
- `<VAPI_PRIVATE_KEY>` - Tvoj VAPI private key
- `phoneNumberId` - ID tvog VAPI telefonskog broja
- `customer.number` - Broj koji želiš nazvati
- `assistantId` - ID tvog VAPI asistenta

---

## 🔍 Debugging

### Provjeri radi li PostgreSQL
```bash
# Linux/Mac
sudo systemctl status postgresql

# Windows - provjeri Task Manager
```

### Provjeri radi li Flask API
```bash
curl http://localhost:5000/api/calls
```

### Provjeri Flask logove
```bash
cd portal-api
python server.py
# Webhooks će se prikazivati u konzoli sa formatom:
# ========== WEBHOOK RECEIVED ==========
```

### Očisti testne podatke iz baze
```bash
psql -U postgres -d voice_portal -c "DELETE FROM transcripts; DELETE FROM call_metadata; DELETE FROM calls;"
```

---

## ⚠️ Production Deployment

**Za produkciju:**

1. **Webhook URL:** Umjesto ngrok, deploy Flask na Heroku/Railway/Render
2. **Auth:** Omogući Auth0 autentifikaciju (trenutno onemogućena za testiranje)
3. **Database:** Koristi managed PostgreSQL (Supabase, Neon, AWS RDS)
4. **Environment:** Stavi sve .env varijable u production environment
5. **HTTPS:** Uvijek koristi HTTPS za webhook endpointe

---

## 📚 Dodatne Informacije

- **Database Schema:** Vidi `DATABASE_SCHEMA.md` za detaljnu strukturu
- **VAPI Dokumentacija:** https://docs.vapi.ai
- **VAPI Webhook Eventi:** conversation-update, transcript, end-of-call-report

---

## 🆘 Troubleshooting

**Problem:** Pozivi se ne pojavljuju na dashboardu
- Provjeri radi li ngrok: `curl https://tvoj-ngrok-url.ngrok-free.dev/api/webhooks/vapi`
- Provjeri Flask logove za webhook evenimente
- Provjeri VAPI server URL konfiguraciju

**Problem:** Database connection error
- Provjeri radi li PostgreSQL
- Provjeri DATABASE_URL u `.env` fajlu
- Provjeri port (5432 ili 5433)

**Problem:** Frontend ne prikazuje podatke
- Provjeri `NEXT_PUBLIC_API_URL` u `.env.local`
- Provjeri radi li Flask API
- Otvori browser console za greške
