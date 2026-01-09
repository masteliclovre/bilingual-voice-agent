# Quick Start Guide - Pokretanje projekta za development

## TL;DR - Brze komande

```bash
# 1. Pokreni PostgreSQL i kreiraj bazu
psql -U postgres
CREATE DATABASE voice_portal;
\q

# 2. Pokreni migracije
psql -U postgres -d voice_portal -f portal-api/migrations/001_initial_schema.sql
psql -U postgres -d voice_portal -f portal-api/migrations/002_add_user_approval.sql

# 3. Backend Setup
cd portal-api
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt

# Kopiraj i popuni .env
cp .env.example .env
# Uredi .env sa svojim podacima

python server_multitenant.py

# 4. Frontend Setup (u novom terminalu)
cd portal-ui
npm install

# Kopiraj i popuni .env.local
cp .env.local.example .env.local
# Uredi .env.local sa svojim podacima

npm run dev

# 5. Ngrok za webhook (u novom terminalu)
ngrok http 5000
```

## Što trebam konfigurirati?

### 1. Google OAuth Setup (10 minuta)

1. Idi na https://console.cloud.google.com/
2. Kreiraj projekt → "APIs & Services" → "Credentials"
3. "Create Credentials" → "OAuth 2.0 Client ID"
4. Authorized redirect: `http://localhost:3000/api/auth/callback/google`
5. Kopiraj **Client ID** i **Client Secret**

### 2. Generiraj NEXTAUTH_SECRET

```bash
# U terminalu:
node -e "console.log(require('crypto').randomBytes(32).toString('base64'))"
```

### 3. PostgreSQL Password

Ako nemaš postavljenu lozinku za postgres korisnika:

```bash
psql -U postgres
ALTER USER postgres PASSWORD 'tvoja_lozinka';
\q
```

## .env fileovi

### portal-api/.env

```env
DATABASE_URL=postgresql://postgres:tvoja_lozinka@localhost:5432/voice_portal
GOOGLE_CLIENT_ID=paste_client_id
GOOGLE_CLIENT_SECRET=paste_client_secret
NEXTAUTH_SECRET=paste_generated_secret
NEXTAUTH_URL=http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:5000
```

### portal-ui/.env.local

```env
GOOGLE_CLIENT_ID=paste_client_id
GOOGLE_CLIENT_SECRET=paste_client_secret
NEXTAUTH_SECRET=paste_generated_secret
NEXTAUTH_URL=http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:5000
NEXT_PUBLIC_USE_MOCK_DATA=false
```

**VAŽNO:** Koristi ISTE vrijednosti za GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET i NEXTAUTH_SECRET u oba file-a!

## Testiranje

1. Otvori http://localhost:3000
2. Sign in sa Google
3. **Automatski si admin** - nema čekanja na odobrenje!
4. Vidi dashboard sa mock podacima

## VAPI Webhook Setup

```bash
# Pokreni ngrok u novom terminalu
ngrok http 5000

# Dobit ćeš URL: https://abc123.ngrok-free.app
# Postavi u VAPI assistant settings:
# Server URL: https://abc123.ngrok-free.app/api/webhooks/vapi
```

## Testiranje webhook-a sa cURL

```bash
curl -X POST http://localhost:5000/api/webhooks/vapi \
  -H "Content-Type: application/json" \
  -d '{
    "message": {"type": "status-update", "status": "ended"},
    "call": {
      "id": "test-123",
      "startedAt": "2026-01-09T10:00:00Z",
      "endedAt": "2026-01-09T10:05:00Z",
      "customer": {"number": "+385991234567"}
    }
  }'
```

## Česte greške

### "Failed to fetch" u admin panelu
- Provjeri da Flask radi na http://localhost:5000
- Pogledaj browser DevTools (F12) → Network tab

### "connection refused" na bazu
- Provjeri DATABASE_URL lozinku u .env
- Provjeri da PostgreSQL radi

### Google OAuth error
- Provjeri da redirect URI u Google Console sadrži `/api/auth/callback/google`
- Provjeri da Client ID i Secret su isti u oba .env file-a

## Za detaljnije upute

Pogledaj [SETUP_GUIDE.md](./SETUP_GUIDE.md) za detaljne upute i objašnjenja.
