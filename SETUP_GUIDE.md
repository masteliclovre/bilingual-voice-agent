# Bilingual Voice Agent - Kompletan Setup Guide

## Pregled projekta

Ovo je multi-tenant voice agent portal sa:
- **Portal UI** (Next.js) - Frontend sa Google OAuth autentifikacijom
- **Portal API** (Flask) - Backend API sa PostgreSQL bazom
- **VAPI Integration** - Voice AI webhook integracija
- **Admin Panel** - Upravljanje korisnicima i pregledavanje poziva

---

## Preduvjeti

Prije početka, instaliraj:
- **Node.js** (v18 ili noviji) - https://nodejs.org/
- **Python** (v3.10 ili noviji) - https://www.python.org/
- **PostgreSQL** (v14 ili noviji) - https://www.postgresql.org/download/
- **ngrok** (za webhook testiranje) - https://ngrok.com/download
- **Git** - https://git-scm.com/

---

## Korak 1: Kloniraj repozitorij

```bash
git clone <repository-url>
cd bilingual-voice-agent
```

---

## Korak 2: PostgreSQL Setup

### 2.1 Pokreni PostgreSQL

Provjeri da PostgreSQL radi:
```bash
# Windows
# PostgreSQL se pokreće automatski kao servis

# Linux/Mac
sudo service postgresql start
```

### 2.2 Kreiraj bazu podataka

```bash
# Otvori psql kao postgres korisnik
psql -U postgres

# U psql konzoli:
CREATE DATABASE voice_portal;
\q
```

### 2.3 Pokreni migracije

```bash
# Pokreni osnovnu shemu
psql -U postgres -d voice_portal -f portal-api/migrations/001_initial_schema.sql

# Pokreni user approval migraciju
psql -U postgres -d voice_portal -f portal-api/migrations/002_add_user_approval.sql
```

---

## Korak 3: Backend API Setup (Flask)

### 3.1 Kreiraj Python virtualenv

```bash
cd portal-api

# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3.2 Instaliraj dependencies

```bash
pip install -r requirements.txt
```

### 3.3 Kreiraj .env file

Kreiraj `portal-api/.env`:

```env
# Database
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/voice_portal

# Google OAuth (dobij sa Google Cloud Console)
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# NextAuth (generiraj random string)
NEXTAUTH_SECRET=your_random_secret_here
NEXTAUTH_URL=http://localhost:3000

# API URL
NEXT_PUBLIC_API_URL=http://localhost:5000
```

**Kako dobiti Google OAuth credentials:**
1. Idi na https://console.cloud.google.com/
2. Kreiraj novi projekt ili odaberi postojeći
3. Idi na "APIs & Services" > "Credentials"
4. Klikni "Create Credentials" > "OAuth 2.0 Client ID"
5. Odaberi "Web application"
6. Dodaj authorized redirect URI: `http://localhost:3000/api/auth/callback/google`
7. Kopiraj Client ID i Client Secret

**Generiraj NEXTAUTH_SECRET:**
```bash
# U Node.js ili online random generator
node -e "console.log(require('crypto').randomBytes(32).toString('base64'))"
```

### 3.4 Pokreni Flask server

```bash
# Provjeri da si u portal-api direktoriju sa aktiviranim venv
python server_multitenant.py
```

Server bi trebao biti pokrenut na `http://localhost:5000`

---

## Korak 4: Frontend Setup (Next.js)

### 4.1 Instaliraj Node dependencies

```bash
# Otvori novi terminal
cd portal-ui
npm install
```

### 4.2 Kreiraj .env.local file

Kreiraj `portal-ui/.env.local`:

```env
# Google OAuth (isti kao u backend .env)
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# NextAuth
NEXTAUTH_SECRET=your_random_secret_here
NEXTAUTH_URL=http://localhost:3000

# API URL
NEXT_PUBLIC_API_URL=http://localhost:5000

# Mock data (za testiranje bez VAPI)
NEXT_PUBLIC_USE_MOCK_DATA=false
```

### 4.3 Pokreni Next.js dev server

```bash
npm run dev
```

Frontend bi trebao biti pokrenut na `http://localhost:3000`

---

## Korak 5: Testiranje aplikacije

1. Otvori browser na `http://localhost:3000`
2. Klikni "Sign in with Google"
3. Prijavi se sa svojim Google accountom
4. **Automatski ćeš biti odobren kao admin** (za development)
5. Trebao bi vidjeti dashboard sa mock podacima

---

## Korak 6: VAPI Webhook Setup

### 6.1 Instaliraj i pokreni ngrok

```bash
# Download sa https://ngrok.com/download

# Pokreni ngrok (u novom terminalu)
ngrok http 5000
```

Dobit ćeš URL poput: `https://abc123.ngrok-free.app`

### 6.2 Konfiguriraj VAPI webhook

1. Prijavi se na https://vapi.ai
2. Idi na svoj assistant
3. U "Server URL" dodaj: `https://abc123.ngrok-free.app/api/webhooks/vapi`
4. **VAŽNO:** Zaključaj webhook signature verification ili postavi VAPI_API_KEY u backend .env

### 6.3 Testiranje webhook-a

Napravi test poziv sa VAPI dashboarda i provjeri Flask logs:

```bash
# U Flask terminalu trebao bi vidjeti:
INFO:__main__:========== VAPI WEBHOOK ==========
INFO:__main__:Event: status-update
INFO:__main__:Call ID: call-abc123
```

---

## Korak 7: cURL Testiranje VAPI Endpointa

```bash
# Test webhook endpoint
curl -X POST http://localhost:5000/api/webhooks/vapi \
  -H "Content-Type: application/json" \
  -d '{
    "message": {
      "type": "status-update",
      "status": "ended"
    },
    "call": {
      "id": "test-call-123",
      "startedAt": "2026-01-09T10:00:00Z",
      "endedAt": "2026-01-09T10:05:00Z",
      "duration": 300,
      "customer": {
        "number": "+385991234567"
      }
    }
  }'

# Test get calls endpoint
curl -X GET http://localhost:5000/api/calls \
  -H "X-Tenant-ID: your_tenant_id_here"
```

---

## Struktura projekta

```
bilingual-voice-agent/
├── portal-api/              # Flask backend
│   ├── migrations/          # SQL migracije
│   ├── server_multitenant.py  # Main Flask app
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # Backend config (kreiraj ručno)
├── portal-ui/               # Next.js frontend
│   ├── app/                 # Next.js App Router
│   │   ├── dashboard/       # Dashboard stranice
│   │   ├── api/auth/        # NextAuth endpoints
│   │   └── page.tsx         # Landing page
│   ├── components/          # React komponente
│   │   ├── AdminSettings.tsx  # Admin panel
│   │   └── Dashboard.tsx    # Main dashboard
│   ├── lib/                 # Utilities
│   │   ├── auth.tsx         # Auth hook
│   │   └── api.ts           # API client
│   ├── package.json
│   └── .env.local           # Frontend config (kreiraj ručno)
└── SETUP_GUIDE.md          # Ova datoteka
```

---

## Baza podataka - Ključne tablice

### `users`
- `id` (UUID) - primarni ključ
- `email` - Google email
- `full_name` - ime korisnika
- `google_id` - Google OAuth ID
- `profile_picture` - URL do slike
- `approval_status` - status odobrenja (approved/pending/rejected)

### `tenants`
- `id` (UUID) - primarni ključ
- `name` - ime tenant-a
- `status` - status (active/inactive)

### `user_tenant_roles`
- `user_id` (UUID) - foreign key na users
- `tenant_id` (UUID) - foreign key na tenants
- `role` - uloga (admin/manager/viewer)

### `calls`
- `vapi_call_id` - ID poziva iz VAPI
- `tenant_id` - kojem tenant-u pripada poziv
- `started_at` - vrijeme početka
- `duration_sec` - trajanje u sekundama
- `outcome` - ishod (resolved/escalated/failed)
- ... i mnogo drugih polja

---

## Admin Panel Features

Kada si prijavljen kao admin, možeš:

✅ **Pregledavati sve pozive** - lista svih poziva sa filterima
✅ **Vidjeti KPI metrike** - ukupni pozivi, resolution rate, SLA compliance
✅ **Upravljati korisnicima** - odobravaj/odbijaj nove korisnike
✅ **Mijenjati uloge** - postavi korisnika kao admin/manager/viewer

**NAPOMENA:** Trenutno je auto-approve uključen za development. Svi novi korisnici automatski dobivaju admin privilegije.

---

## Troubleshooting

### Problem: "Failed to fetch" u admin panelu

**Rješenje:**
1. Provjeri da Flask server radi na http://localhost:5000
2. Provjeri CORS postavke u `server_multitenant.py`
3. Otvori browser DevTools (F12) i pogledaj Network tab za detalje

### Problem: "Admin access required"

**Rješenje:**
1. Odjavi se i ponovno prijavi
2. Provjeri da li korisnik ima admin ulogu u bazi:
```sql
SELECT u.email, utr.role
FROM users u
JOIN user_tenant_roles utr ON u.id = utr.user_id
WHERE u.email = 'your_email@gmail.com';
```

### Problem: PostgreSQL connection error

**Rješenje:**
1. Provjeri da PostgreSQL radi
2. Provjeri DATABASE_URL u `.env` file
3. Provjeri password i port (default: 5432)

### Problem: ngrok webhook ne radi

**Rješenje:**
1. Provjeri da ngrok URL završava sa `/api/webhooks/vapi`
2. Provjeri Flask logs za greške
3. Testiraj webhook sa cURL-om prvo

### Problem: Google OAuth error

**Rješenje:**
1. Provjeri da redirect URI u Google Console sadrži:
   - `http://localhost:3000/api/auth/callback/google`
2. Provjeri da GOOGLE_CLIENT_ID i SECRET odgovaraju u oba .env file-a

---

## Production Deployment (TODO)

Za production deployment trebat će:

1. **PostgreSQL** - Hosted database (npr. AWS RDS, DigitalOcean)
2. **Flask API** - Deploy na Heroku, Railway, ili VPS
3. **Next.js** - Deploy na Vercel ili Netlify
4. **Domain** - Kupiti domain i podesiti DNS
5. **SSL Certificate** - Automatski sa Vercel/Netlify
6. **Environment variables** - Postaviti production vrijednosti
7. **Disable auto-approve** - Vratiti user approval workflow

---

## Kontakt i Support

Za pitanja ili probleme:
- Otvori issue na GitHub repozitoriju
- Kontaktiraj developera

---

## License

Proprietary - Za internu upotrebu.
