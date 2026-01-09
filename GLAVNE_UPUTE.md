# GLAVNE UPUTE - Pokretanje Projekta (Windows PowerShell)

Kompletne upute za pokretanje projekta. Sve komande su za Windows PowerShell.

---

## 📋 Sadržaj

1. [Preuzmi kod](#1-preuzmi-kod)
2. [Instaliraj što fali](#2-instaliraj-što-fali)
3. [PostgreSQL setup](#3-postgresql-setup)
4. [Backend (Flask API)](#4-backend-flask-api)
5. [Frontend (Next.js)](#5-frontend-nextjs)
6. [Testiranje](#6-testiranje)
7. [VAPI Webhook + ngrok](#7-vapi-webhook--ngrok)
8. [RunPod Server](#8-runpod-server)
9. [Groq API](#9-groq-api)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Preuzmi kod

Otvori PowerShell:

```powershell
git clone https://github.com/masteliclovre/bilingual-voice-agent.git
cd bilingual-voice-agent
git checkout feature/vapi-call-portal
```

---

## 2. Instaliraj što fali

**Node.js 18+**
- Download: https://nodejs.org/ (LTS verzija)

**PostgreSQL 14+**
- Download: https://www.postgresql.org/download/windows/
- Zapamti password za `postgres` korisnika
- Port: ostavi `5432`

**pgAdmin 4**
- Dolazi sa PostgreSQL instalacijom

**ngrok**
- Download: https://ngrok.com/download
- Registriraj se → dobij authtoken
- PowerShell: `ngrok config add-authtoken your_token_here`

---

## 3. PostgreSQL Setup

### 3.1 Provjeri da PostgreSQL radi

PowerShell:
```powershell
Get-Service | Where-Object {$_.Name -like "*postgresql*"}
```

Status mora biti "Running". Ako nije:
```powershell
Start-Service postgresql-x64-14
```

### 3.2 Otvori pgAdmin

1. Start menu → pgAdmin 4
2. Unesi master password (prvi put kreiraš)
3. Lijevi sidebar → Servers → PostgreSQL 14
4. Upiši password koji si postavio

### 3.3 Kreiraj bazu

Desni klik na "Databases" → Create → Database
- Database name: `voice_portal`
- Owner: `postgres`
- Save

### 3.4 Pokreni migracije

U pgAdmin:
1. Gore lijevo dropdown → odaberi `voice_portal`
2. Desni klik na "PostgreSQL 14" → Query Tool
3. Otvori `portal-api/migrations/001_multi_tenant_schema.sql`
4. Kopiraj sve → zalijepi u Query Tool → Execute (F5)
5. Otvori `portal-api/migrations/002_add_user_approval.sql`
6. Kopiraj sve → zalijepi → Execute (F5)

### 3.5 Provjeri tablice

Query Tool:
```sql
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
```

Trebao bi vidjeti: users, tenants, user_tenant_roles, calls, call_analytics

---

## 4. Backend (Flask API)

PowerShell u `portal-api` folderu:

```powershell
cd portal-api
```

### 4.1 Kreiraj Python virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Vidjet ćeš `(venv)` ispred prompta.

### 4.2 Instaliraj dependencies

```powershell
pip install -r requirements.txt
```

### 4.3 Kreiraj .env file

```powershell
Copy-Item .env.example .env
```

Otvori `.env` u editoru (Notepad++, VS Code):

```env
DATABASE_URL=postgresql://postgres:TVOJ_POSTGRES_PASSWORD@localhost:5432/voice_portal
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
NEXTAUTH_SECRET=...
NEXTAUTH_URL=http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:5000
```

**Promijeni samo `TVOJ_POSTGRES_PASSWORD`!** Ostalo NE DIRAJ.

### 4.4 Pokreni Flask server

```powershell
python server_multitenant.py
```

Output:
```
 * Running on http://127.0.0.1:5000
```

**Ostavi terminal otvoren!**

### 4.5 Testiraj backend

Novi PowerShell:

```powershell
Invoke-WebRequest http://localhost:5000/api/webhooks/vapi
```

Bilo koji odgovor = radi.

---

## 5. Frontend (Next.js)

Novi PowerShell u `portal-ui`:

```powershell
cd portal-ui
```

**Ne zatvaraj Flask terminal!**

### 5.1 Instaliraj dependencies

```powershell
npm install
```

Traje 1-2 min.

### 5.2 Kreiraj .env.local

```powershell
Copy-Item .env.local.example .env.local
```

Otvori `.env.local`:

```env
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
NEXTAUTH_SECRET=...
NEXTAUTH_URL=http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:5000
NEXT_PUBLIC_USE_MOCK_DATA=false
```

**NE MIJENJAJ** ništa - već podešeno.

### 5.3 Pokreni Next.js

```powershell
npm run dev
```

Output:
```
  ▲ Next.js 14.0.0
  - Local: http://localhost:3000
```

### 5.4 Otvori browser

http://localhost:3000

Vidjet ćeš "Sign in with Google".

---

## 6. Testiranje

1. http://localhost:3000
2. "Sign in with Google"
3. Odaberi account
4. **Automatski si admin** (za dev)
5. Dashboard → vidiš KPI metrike, grafove
6. "Admin" → tabla korisnika

Flask terminal:
```
INFO:__main__:========== SYNC USER REQUEST ==========
INFO:__main__:Email: tvoj_email@gmail.com
```

---

## 7. VAPI Webhook + ngrok

ngrok stvara public URL → tvoj localhost. VAPI šalje webhooks tamo.

### 7.1 Pokreni ngrok

Treći PowerShell:

```powershell
ngrok http 5000
```

Output:
```
Forwarding  https://abc123.ngrok-free.app -> http://localhost:5000
```

Kopiraj taj URL!

### 7.2 Postavi u VAPI

1. https://vapi.ai → login
2. Tvoj Assistant → "Server Settings"
3. Webhook URL: `https://abc123.ngrok-free.app/api/webhooks/vapi`
4. Save

### 7.3 Testiraj

PowerShell:
```powershell
$body = @{
    message = @{type="status-update"; status="ended"}
    call = @{id="test-123"; startedAt="2026-01-09T10:00:00Z"; customer=@{number="+385991234567"}}
} | ConvertTo-Json
Invoke-WebRequest -Method POST -Uri http://localhost:5000/api/webhooks/vapi -Body $body -ContentType "application/json"
```

Flask logs:
```
INFO:__main__:========== VAPI WEBHOOK ==========
```

Test sa VAPI:
1. Napravi test poziv u VAPI
2. Check Flask logs
3. pgAdmin: `SELECT * FROM calls;`

---

## 8. RunPod Server

RunPod = cloud GPU za voice agent server.

### 8.1 Setup

1. https://www.runpod.io/ → login
2. Deploy → GPU Instance
3. Template: PyTorch / Python 3.10
4. GPU: RTX 4000 / T4
5. Deploy

### 8.2 Upload kod

RunPod terminal:

```bash
cd /workspace
git clone https://github.com/masteliclovre/bilingual-voice-agent.git
cd bilingual-voice-agent/test2
pip install -r requirements.txt
```

### 8.3 Config

```bash
nano .env
```

Paste:
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile
WHISPER_MODEL=GoranS/whisper-base-1m.hr-ctranslate2
WHISPER_DEVICE=cuda
ELEVENLABS_API_KEY=...
PORT=8000
REMOTE_SERVER_AUTH_TOKEN=random_token_123
```

Save: Ctrl+X → Y → Enter

### 8.4 Pokreni

```bash
python server.py
```

ili:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 8.5 Kopiraj URL

RunPod dashboard → instanca → "TCP Port Mappings" → port 8000

URL: `https://abc123-8000.proxy.runpod.net/`

### 8.6 Testiraj

```bash
curl https://your-runpod-url.proxy.runpod.net/healthz
```

Output:
```json
{"status": "ok", "llm_provider": "groq"}
```

### 8.7 Poveži lokalni Voice Agent

Lokalno u `test2/.env`:

```env
REMOTE_AGENT_URL=https://your-runpod-url.proxy.runpod.net/
REMOTE_AGENT_TOKEN=random_token_123
REMOTE_AGENT_API_KEY=gsk_...
```

Test:
```powershell
cd test2
python voice_agent.py
```

Govori → dobiješ odgovor.

---

## 9. Groq API

Ultra-brzi LLM. Besplatan za dev.

### 9.1 Dobij API Key

1. https://console.groq.com/
2. Sign up (Google account)
3. API Keys → Create
4. Kopiraj key (`gsk_...`)

### 9.2 Postavi

RunPod `test2/.env`:
```env
GROQ_API_KEY=gsk_...
```

Lokalno `test2/.env`:
```env
REMOTE_AGENT_API_KEY=gsk_...
```

### 9.3 Modeli

```env
GROQ_MODEL=llama-3.1-8b-instant          # Najbrži
GROQ_MODEL=llama-3.3-70b-versatile       # Default
GROQ_MODEL=llama-3.1-70b-versatile       # Najkvalitetniji
```

---

## 10. Troubleshooting

### PostgreSQL connection error

PowerShell:
```powershell
Get-Service postgresql*
```

Provjeri password u `.env`:
```env
DATABASE_URL=postgresql://postgres:PRAVI_PASSWORD@localhost:5432/voice_portal
```

### "Failed to fetch" u Admin Panelu

1. Flask radi? (`python server_multitenant.py`)
2. F12 → Network tab
3. Check CORS u Flask logs

### "Module not found"

Python:
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Node.js:
```powershell
Remove-Item -Recurse -Force node_modules
npm install
```

### ngrok error

```powershell
ngrok config add-authtoken YOUR_TOKEN
```

Token: https://dashboard.ngrok.com/get-started/your-authtoken

### RunPod ne reagira

1. `server.py` radi?
2. URL točan? (završava sa `/`)
3. Test: `curl https://runpod-url.proxy.runpod.net/healthz`

### Groq API greška

- Key počinje sa `gsk_`?
- Novi key: https://console.groq.com/
- Rate limit → čekaj 1 min

### "Admin access required"

1. Logout → login ponovno
2. Flask logs: vidi "SYNC USER"?
3. pgAdmin:
```sql
SELECT u.email, utr.role FROM users u
JOIN user_tenant_roles utr ON u.id = utr.user_id
WHERE u.email = 'tvoj@email.com';
```

Mora biti `role = 'admin'`.

---

## 📝 Što treba biti pokrenuto?

**Lokalni dev** (3 PowerShell-a):

```powershell
# 1. Backend
cd portal-api
.\venv\Scripts\Activate.ps1
python server_multitenant.py

# 2. Frontend
cd portal-ui
npm run dev

# 3. Webhook (opciono)
ngrok http 5000
```

**Voice Agent + RunPod**:

```bash
# RunPod
cd /workspace/bilingual-voice-agent/test2
python server.py
```

```powershell
# Lokalno
cd test2
python voice_agent.py
```

---

## ✅ Checklist

- [ ] PostgreSQL → running
- [ ] Baza `voice_portal` → kreirana
- [ ] Migracije → izvršene
- [ ] Backend `.env` → password promjenjen
- [ ] Flask → pokrenut
- [ ] Frontend `npm install` → gotov
- [ ] Frontend `.env.local` → kopiran (NE mijenjaj)
- [ ] Next.js → pokrenut
- [ ] Login → radi
- [ ] Admin panel → vidiš
- [ ] ngrok + VAPI → podešeno (opciono)
- [ ] RunPod → setup (opciono)
- [ ] Groq API → key dobiven (opciono)
