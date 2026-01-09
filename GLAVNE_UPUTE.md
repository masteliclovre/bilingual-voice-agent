# GLAVNE UPUTE - Pokretanje Projekta

Kompletne upute za pokretanje projekta od nule. Sve što trebaš znati na jednom mjestu.

---

## 📋 Sadržaj

1. [Preuzmi kod](#1-preuzmi-kod)
2. [Instaliraj programe](#2-instaliraj-programe)
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

```bash
git clone https://github.com/masteliclovre/bilingual-voice-agent.git
cd bilingual-voice-agent
git checkout feature/vapi-call-portal
```

---

## 2. Instaliraj programe

### Windows

**Python 3.10+**
- Download: https://www.python.org/downloads/
- Važno: Označi "Add Python to PATH" prilikom instalacije

**Node.js 18+**
- Download: https://nodejs.org/ (preuzmi LTS verziju)

**PostgreSQL 14+**
- Download: https://www.postgresql.org/download/windows/
- Instalacija: Zapamti password za `postgres` korisnika!
- Port: ostavi default `5432`

**pgAdmin 4** (dolazi sa PostgreSQL)
- Automatski se instalira sa PostgreSQL-om
- Koristit ćeš ga za upravljanje bazom

**ngrok**
- Download: https://ngrok.com/download
- Registriraj se i dobij authtoken
- Setup: `ngrok config add-authtoken your_token_here`

**Git** (ako nemaš)
- Download: https://git-scm.com/

### Linux/Mac

```bash
# Python
sudo apt install python3.10 python3-pip python3-venv  # Ubuntu/Debian
brew install python@3.10                               # Mac

# Node.js
sudo apt install nodejs npm                            # Ubuntu/Debian
brew install node                                      # Mac

# PostgreSQL
sudo apt install postgresql postgresql-contrib         # Ubuntu/Debian
brew install postgresql@14                             # Mac

# ngrok
# Download sa https://ngrok.com/download
```

---

## 3. PostgreSQL Setup

### 3.1 Provjeri da PostgreSQL radi

**Windows:**
- Otvori Services (Win + R → `services.msc`)
- Potraži "postgresql-x64-14" (ili slična verzija)
- Status mora biti "Running"

**Linux/Mac:**
```bash
sudo service postgresql status
# Ako nije pokrenut:
sudo service postgresql start
```

### 3.2 Otvori pgAdmin

1. Pokreni pgAdmin 4 (iz Start menua ili Applications)
2. Unesi master password (prvi put će te pitati da kreiraš)
3. U lijevom sidebaru → Servers → PostgreSQL 14 (ili verzija koju imaš)
4. Upiši password koji si postavio prilikom instalacije

### 3.3 Kreiraj bazu

**Opcija 1: Putem pgAdmin GUI**
1. Desni klik na "Databases" → Create → Database
2. Database name: `voice_portal`
3. Owner: `postgres`
4. Klikni "Save"

**Opcija 2: Putem SQL konzole**
1. Desni klik na "PostgreSQL 14" → Query Tool
2. Kopiraj i execute:
```sql
CREATE DATABASE voice_portal;
```

### 3.4 Pokreni migracije

Još u pgAdmin Query Tool-u:

1. **Promijeni bazu**: Gore lijevo dropdown, odaberi `voice_portal` umjesto `postgres`

2. **Migracija 1**: Otvori file `portal-api/migrations/001_multi_tenant_schema.sql`
   - Kopiraj cijeli sadržaj
   - Zalijepi u Query Tool
   - Klikni "Execute" (F5)
   - Trebao bi vidjeti "Query returned successfully"

3. **Migracija 2**: Otvori file `portal-api/migrations/002_add_user_approval.sql`
   - Kopiraj cijeli sadržaj
   - Zalijepi u Query Tool
   - Execute (F5)

### 3.5 Provjeri tablice

U Query Tool-u, execute:
```sql
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public';
```

Trebao bi vidjeti tablice:
- users
- tenants
- user_tenant_roles
- calls
- call_analytics

---

## 4. Backend (Flask API)

### 4.1 Otvori terminal u `portal-api` folderu

```bash
cd portal-api
```

### 4.2 Kreiraj Python virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Vidjet ćeš `(venv)` ispred command prompt-a kada je aktivan.

### 4.3 Instaliraj dependencies

```bash
pip install -r requirements.txt
```

Ovo će instalirati:
- Flask (web framework)
- flask-cors (za komunikaciju sa frontend-om)
- psycopg (PostgreSQL driver)
- python-dotenv (za .env fileove)

### 4.4 Kreiraj .env file

**Kopiraj template:**
```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

**Uredi `.env` file** (otvori u Notepad++ ili VS Code):

```env
# PostgreSQL - promijeni password!
DATABASE_URL=postgresql://postgres:TVOJ_POSTGRES_PASSWORD@localhost:5432/voice_portal

# Google OAuth (već podešeno, NE MIJENJAJ)
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...

# NextAuth (već podešeno, NE MIJENJAJ)
NEXTAUTH_SECRET=...
NEXTAUTH_URL=http://localhost:3000

# API URL
NEXT_PUBLIC_API_URL=http://localhost:5000
```

**Važno:** Samo promijeni `TVOJ_POSTGRES_PASSWORD` sa passwordom koji si postavio za PostgreSQL!

### 4.5 Pokreni Flask server

```bash
python server_multitenant.py
```

Trebao bi vidjeti:
```
 * Running on http://127.0.0.1:5000
 * Restarting with stat
```

**Ostavi ovaj terminal otvoren!** Flask server mora biti pokrenut dok radiš.

### 4.6 Testiraj backend

Otvori **novi terminal** i testiraj:

```bash
curl http://localhost:5000/api/webhooks/vapi
```

Trebao bi dobiti neki odgovor (čak i ako je greška, znači server radi).

---

## 5. Frontend (Next.js)

### 5.1 Otvori **NOVI terminal** u `portal-ui` folderu

```bash
# Iz root foldera projekta
cd portal-ui
```

**Nemoj zatvarati Flask terminal!** Treba ti otvoreno 2 terminala odjednom.

### 5.2 Instaliraj Node.js dependencies

```bash
npm install
```

Ovo će trajati 1-2 minute. Instalira Next.js, React, i sve potrebne pakete.

### 5.3 Kreiraj .env.local file

**Kopiraj template:**
```bash
# Windows
copy .env.local.example .env.local

# Linux/Mac
cp .env.local.example .env.local
```

**Uredi `.env.local` file:**

```env
# Google OAuth (već podešeno, NE MIJENJAJ)
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...

# NextAuth (već podešeno, NE MIJENJAJ)
NEXTAUTH_SECRET=...
NEXTAUTH_URL=http://localhost:3000

# Backend API
NEXT_PUBLIC_API_URL=http://localhost:5000

# Mock data - za testiranje BEZ pravog backenda
NEXT_PUBLIC_USE_MOCK_DATA=false
```

**Ništa ne mijenjaj** - Google Auth je već podešen.

### 5.4 Pokreni Next.js dev server

```bash
npm run dev
```

Trebao bi vidjeti:
```
  ▲ Next.js 14.0.0
  - Local:        http://localhost:3000
  - Ready in 2.3s
```

### 5.5 Otvori aplikaciju

Otvori browser: **http://localhost:3000**

Trebao bi vidjeti landing page sa "Sign in with Google" buttonom.

---

## 6. Testiranje

### 6.1 Prijavi se

1. Klikni "Sign in with Google"
2. Odaberi Google account
3. **Automatski ćeš biti odobren kao admin!** (za development)
4. Preusmjerit će te na Dashboard

### 6.2 Provjeri Dashboard

Trebao bi vidjeti:
- KPI metrics (Total Calls, AI Answer Rate, itd.)
- Grafikone sa mock podacima
- "Admin" u navigaciji (ako si admin)

### 6.3 Provjeri Admin Panel

1. Klikni "Admin" u top navigaciji
2. Trebao bi vidjeti praznu tablicu korisnika (ili tvoj account)

### 6.4 Provjeri Flask logs

U Flask terminalu trebao bi vidjeti:
```
INFO:__main__:========== SYNC USER REQUEST ==========
INFO:__main__:Email: tvoj_email@gmail.com
INFO:__main__:Full Name: Tvoje Ime
```

---

## 7. VAPI Webhook + ngrok

### 7.1 Što je ngrok?

ngrok stvara public URL koji pokazuje na tvoj localhost. VAPI šalje webhook-e na taj URL.

```
Internet (VAPI)  →  ngrok URL  →  Tvoj localhost:5000
```

### 7.2 Pokreni ngrok

Otvori **TREĆI terminal**:

```bash
ngrok http 5000
```

Vidjet ćeš nešto ovako:
```
Session Status                online
Forwarding                    https://abc123.ngrok-free.app -> http://localhost:5000
```

**Kopiraj taj `https://...ngrok-free.app` URL!**

### 7.3 Postavi u VAPI

1. Prijavi se na https://vapi.ai
2. Idi na svoj Assistant
3. U "Server Settings" ili "Webhook URL":
   - Zalijepi: `https://abc123.ngrok-free.app/api/webhooks/vapi`
   - **Važno:** Mora završavati sa `/api/webhooks/vapi`
4. Spremi

### 7.4 Testiraj webhook

**cURL test:**
```bash
curl -X POST http://localhost:5000/api/webhooks/vapi \
  -H "Content-Type: application/json" \
  -d "{\"message\": {\"type\": \"status-update\", \"status\": \"ended\"}, \"call\": {\"id\": \"test-call-123\", \"startedAt\": \"2026-01-09T10:00:00Z\", \"endedAt\": \"2026-01-09T10:05:00Z\", \"customer\": {\"number\": \"+385991234567\"}}}"
```

U Flask terminalu trebao bi vidjeti:
```
INFO:__main__:========== VAPI WEBHOOK ==========
INFO:__main__:Event: status-update
```

**Test sa VAPI:**
1. Napravi test poziv u VAPI dashboardu
2. Provjeri Flask logs
3. Provjeri pgAdmin: `SELECT * FROM calls;`

---

## 8. RunPod Server

### 8.1 Što je RunPod?

RunPod je cloud GPU servis gdje hostaš voice agent server (`test2/server.py`).

### 8.2 Setup na RunPodu

**1. Prijavi se na RunPod:**
- https://www.runpod.io/
- Sign up / Login

**2. Kreiraj GPU Instance:**
- Klikni "Deploy" → "GPU Instance"
- Odaberi template: "PyTorch" ili "Python 3.10"
- GPU: Najmanja (npr. RTX 4000 ili T4)
- Klikni "Deploy"

**3. Otvori Terminal na RunPod instanci:**
- Klikni na instancu → "Connect" → "Start Web Terminal"

**4. Upload kod:**

**Opcija A: Git Clone**
```bash
cd /workspace
git clone https://github.com/masteliclovre/bilingual-voice-agent.git
cd bilingual-voice-agent/test2
```

**Opcija B: Upload fileove ručno**
- Koristi RunPod file browser
- Upload `test2/` folder

**5. Instaliraj dependencies:**
```bash
cd /workspace/bilingual-voice-agent/test2
pip install -r requirements.txt
```

**6. Kreiraj .env file:**
```bash
nano .env
```

Kopiraj i zalijepi:
```env
# LLM Provider
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Whisper (Croatian STT)
WHISPER_MODEL=GoranS/whisper-base-1m.hr-ctranslate2
WHISPER_DEVICE=cuda
WHISPER_COMPUTE=float16

# ElevenLabs TTS
ELEVENLABS_API_KEY=your_elevenlabs_key_here
ELEVENLABS_VOICE_ID=vFQACl5nAIV0owAavYxE

# Server
PORT=8000
REMOTE_SERVER_AUTH_TOKEN=your_random_secret_token
```

Spremi: `Ctrl+X`, pa `Y`, pa `Enter`

**7. Pokreni server:**
```bash
python server.py
```

ili sa uvicorn:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

**8. Kopiraj RunPod URL:**
- U RunPod dashboardu klikni na instancu
- Kopiraj "TCP Port Mappings" URL za port 8000
- Trebao bi biti poput: `https://abc123-8000.proxy.runpod.net/`

### 8.3 Testiraj RunPod server

```bash
curl https://your-runpod-url.proxy.runpod.net/healthz
```

Trebao bi dobiti:
```json
{
  "status": "ok",
  "llm_provider": "groq",
  "llm_model": "llama-3.3-70b-versatile"
}
```

### 8.4 Poveži Voice Agent sa RunPod serverom

U `test2/.env` na tvom lokalnom kompjuteru:

```env
REMOTE_AGENT_URL=https://your-runpod-url.proxy.runpod.net/
REMOTE_AGENT_TOKEN=your_random_secret_token
REMOTE_AGENT_API_KEY=your_groq_api_key
```

Testiraj:
```bash
cd test2
python voice_agent.py
```

Govori u mikrofon → Trebao bi dobiti odgovor!

---

## 9. Groq API

### 9.1 Što je Groq?

Groq je ultra-brzi LLM inference servis. Besplatan je za development!

### 9.2 Dobij Groq API Key

1. Idi na https://console.groq.com/
2. Sign up / Login (može sa Google accountom)
3. Idi na "API Keys"
4. Klikni "Create API Key"
5. Kopiraj key (počinje sa `gsk_...`)

### 9.3 Postavi u .env

**Na RunPod serveru** (`test2/.env`):
```env
GROQ_API_KEY=gsk_...tvoj_key_ovdje
```

**Na lokalnom kompjuteru** (`test2/.env`):
```env
REMOTE_AGENT_API_KEY=gsk_...tvoj_key_ovdje
```

### 9.4 Odaberi Model

Preporučeni modeli (od najbržeg do najpametnijeg):

```env
# Najbrži - za development
GROQ_MODEL=llama-3.1-8b-instant

# Balansirani - production default
GROQ_MODEL=llama-3.3-70b-versatile

# Najkvalitetniji - za složene zadatke
GROQ_MODEL=llama-3.1-70b-versatile
```

---

## 10. Troubleshooting

### "Cannot connect to PostgreSQL"

**Provjeri:**
```bash
# Je li PostgreSQL pokrenut?
# Windows: Services → postgresql-x64-14
# Linux/Mac: sudo service postgresql status

# Provjeri password u .env file-u
DATABASE_URL=postgresql://postgres:PRAVI_PASSWORD@localhost:5432/voice_portal
```

**Test konekcije:**
```bash
psql -U postgres -d voice_portal
```

### "Failed to fetch" u Admin Panelu

**Provjeri:**
1. Je li Flask server pokrenut? (terminal: `python server_multitenant.py`)
2. Otvori browser DevTools (F12) → Network tab
3. Pogledaj error u Console

**Česta greška: CORS**
- Provjeri da Flask terminal ne pokazuje CORS error
- U `server_multitenant.py` mora biti: `origins: ["http://localhost:3000"]`

### "Module not found" greška

**Python:**
```bash
# Provjeri da je venv aktivan
# Trebao bi vidjeti (venv) ispred prompta

# Reinstaliraj dependencies
pip install -r requirements.txt
```

**Node.js:**
```bash
# Obriši i reinstaliraj
rm -rf node_modules
npm install
```

### ngrok "ERR_NGROK_108"

```bash
# Dodaj authtoken
ngrok config add-authtoken YOUR_AUTHTOKEN
```

Dobij authtoken na: https://dashboard.ngrok.com/get-started/your-authtoken

### RunPod server ne reagira

**Provjeri:**
1. Je li server.py pokrenut? (Terminal na RunPod-u)
2. Kopiraj TOČAN URL sa port mappinga (završava sa `/`)
3. Test: `curl https://runpod-url.proxy.runpod.net/healthz`

### Groq API Error

**"Invalid API Key":**
- Provjeri da key počinje sa `gsk_`
- Generiraj novi key na https://console.groq.com/

**"Rate limit exceeded":**
- Besplatni tier ima limite
- Čekaj 1 minutu ili kreiraj novi account

### "Admin access required"

1. Odjavi se (logout)
2. Ponovno se prijavi sa Google
3. Provjeri u Flask terminalu da vidiš "SYNC USER" logs
4. Provjeri u pgAdmin:
```sql
SELECT u.email, utr.role
FROM users u
JOIN user_tenant_roles utr ON u.id = utr.user_id
WHERE u.email = 'tvoj_email@gmail.com';
```

Trebao bi imati `role = 'admin'`.

### Voice Agent ne čuje mikrofon

**Lista audio devicea:**
```python
import sounddevice as sd
print(sd.query_devices())
```

**U .env:**
```env
# Koristi ime devicea
PREFERRED_INPUT_NAME=USB

# Ili index
INPUT_DEVICE_INDEX=1
```

---

## 📝 Sažetak - Šta treba biti pokrenuto?

Za **lokalni development** trebaš 3 terminala:

```
Terminal 1 (Backend):
cd portal-api
venv\Scripts\activate
python server_multitenant.py

Terminal 2 (Frontend):
cd portal-ui
npm run dev

Terminal 3 (Webhook - opciono):
ngrok http 5000
```

Za **Voice Agent** + RunPod:

```
RunPod Terminal:
cd /workspace/bilingual-voice-agent/test2
python server.py

Lokalni Terminal:
cd test2
python voice_agent.py
```

---

## ✅ Checklist - Jesi li sve napravio?

- [ ] PostgreSQL instaliran i pokrenut
- [ ] Baza `voice_portal` kreirana
- [ ] Migracije izvršene (001 i 002)
- [ ] Backend .env file popunjen (samo password!)
- [ ] Flask server pokrenut (`python server_multitenant.py`)
- [ ] Frontend dependencies instalirane (`npm install`)
- [ ] Frontend .env.local kreiran (ne mijenjaj Google Auth!)
- [ ] Next.js pokrenut (`npm run dev`)
- [ ] Prijavljen na http://localhost:3000
- [ ] Admin panel radi
- [ ] (Opciono) ngrok pokrenut i VAPI webhook podešen
- [ ] (Opciono) RunPod server setup i pokrenut
- [ ] (Opciono) Groq API key dobiven i postavljen

---

## 🆘 Pomoć

Ako nešto ne radi:

1. Provjeri sve terminale za error poruke
2. Pogledaj Troubleshooting sekciju iznad
3. Provjeri da su svi servisi pokrenuti (PostgreSQL, Flask, Next.js)
4. Otvori browser DevTools (F12) i provjeri Console i Network tabove

---

Sretno! 🚀
