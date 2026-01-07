# Quick Start Guide

Brzo pokretanje VAPI Call Portal aplikacije.

## 📋 Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL 15+
- Ngrok account (za webhooks)
- VAPI account

## ⚡ Quick Setup (5 minuta)

### 1. PostgreSQL Setup
```bash
# Uloguj se u PostgreSQL
psql -U postgres

# Kreiraj bazu
CREATE DATABASE voice_portal;

# Pokreni SQL skriptu
\c voice_portal
\i portal-api/init_db.sql
```

### 2. Backend Setup
```bash
cd portal-api
pip install -r requirements.txt

# Kopiraj i prilagodi .env
cp .env.template .env
# Edituj DATABASE_URL u .env fajlu

# Pokreni API
python server.py
```

### 3. Frontend Setup
```bash
cd portal-ui
npm install

# Kopiraj i prilagodi .env.local
cp .env.template .env.local

# Pokreni frontend
npm run dev
```

### 4. Ngrok Setup
```bash
# Autentifikacija (jednom)
ngrok config add-authtoken <TVOJ_TOKEN>

# Pokreni tunnel
ngrok http 5000
```

Kopiraj ngrok HTTPS URL.

### 5. VAPI Webhook Setup
```bash
# Zamijeni <NGROK_URL> i <VAPI_KEY>
curl -X PATCH https://api.vapi.ai/assistant/<ASSISTANT_ID> \
  -H "Authorization: Bearer <VAPI_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"serverUrl": "<NGROK_URL>/api/webhooks/vapi"}'
```

## 🎯 Test

Napravi testni poziv:
```bash
curl -X POST https://api.vapi.ai/call/phone \
  -H "Authorization: Bearer <VAPI_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "phoneNumberId": "<PHONE_NUMBER_ID>",
    "customer": {"number": "+385919345791"},
    "assistantId": "<ASSISTANT_ID>",
    "name": "Test poziv"
  }'
```

Otvori dashboard: http://localhost:3000/dashboard

## 📚 Detaljne Upute

Za više detalja vidi: [PORTAL_SETUP.md](PORTAL_SETUP.md)

## 🔧 URLs

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:5000
- **Ngrok URL:** https://tvoj-url.ngrok-free.dev (mijenja se svaki put)

## ⚠️ Prije Git Push

Provjeri da si dodao u `.gitignore`:
- `.env` fajlove
- `call-test.json` (sadrži API keys)
- `configure-vapi-webhook.json`
- `node_modules/`
- `.next/`

```bash
# Provjeri što će se commitati
git status

# Nemoj commitati .env ili API keys!
git add .
git commit -m "Add VAPI call portal"
git push origin main
```
