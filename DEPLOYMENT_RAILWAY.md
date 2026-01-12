# Railway Deployment - Kompletne Upute

Deploy cijelog projekta (PostgreSQL + Flask + Next.js) na Railway.

---

## 📋 Što ćeš dobiti

- `https://bilingual-voice-agent-production.up.railway.app` - Frontend (Next.js)
- `https://bilingual-voice-agent-backend-production.up.railway.app` - Backend (Flask)
- PostgreSQL baza (internal Railway URL)
- Automatski deploy na git push

---

## Priprema prije deploya

### 1. Kreiraj config fileove za Railway

**Backend - Procfile**

Kreiraj `portal-api/Procfile`:

```
web: gunicorn server_multitenant:app
```

**Backend - dodaj gunicorn**

Dodaj u `portal-api/requirements.txt`:

```
Flask==3.0.0
flask-cors==4.0.0
psycopg==3.2.1
psycopg-binary==3.2.1
python-dotenv==1.0.1
requests==2.32.3
gunicorn==21.2.0
```

**Backend - runtime config**

Kreiraj `portal-api/runtime.txt`:

```
python-3.10.12
```

### 2. Update CORS u Flask-u

Otvori `portal-api/server_multitenant.py`, promijeni liniju ~15:

**PRIJE:**
```python
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000"]}}, supports_credentials=True)
```

**POSLIJE:**
```python
allowed_origins = [
    "http://localhost:3000",
    "https://*.railway.app",
    "https://*.up.railway.app"
]
CORS(app, resources={r"/api/*": {"origins": allowed_origins, "supports_credentials": True}})
```

### 3. Update Google OAuth redirect

Idi na Google Cloud Console → OAuth credentials:

Dodaj u **Authorized redirect URIs**:
```
https://YOUR-FRONTEND-URL.up.railway.app/api/auth/callback/google
```

(Dodat ćeš točan URL nakon što Railway kreira frontend)

---

## Railway Setup

### 1. Registracija

1. https://railway.app/
2. Sign up sa GitHub accountom
3. Verifikuj email
4. **BITNO:** Dodaj payment method (besplatno $5 kredit, ali traže karticu)

### 2. Install Railway CLI

PowerShell:

```powershell
npm install -g @railway/cli
```

Login:

```powershell
railway login
```

Otvrit će browser → authorize.

---

## Deploy - Korak po korak

### 1. Kreiraj Railway Project

PowerShell u root folderu projekta:

```powershell
cd c:\Users\Marko\bilingual-voice-agent
railway init
```

Pitanja:
- Project name: `bilingual-voice-agent`
- Environment: `production` (Enter)

### 2. Dodaj PostgreSQL

Railway dashboard ili CLI:

```powershell
railway add
```

Odaberi: **PostgreSQL**

Railway će automatski kreirati bazu i dodati `DATABASE_URL` environment varijablu.

### 3. Pokreni migracije na Railway PostgreSQL

**Dohvati connection string:**

```powershell
railway variables
```

Kopiraj `DATABASE_URL` (počinje sa `postgresql://postgres:...railway.app`)

**Povezi se sa bazom:**

```powershell
psql "PASTE_DATABASE_URL_OVDJE"
```

**Pokreni migracije:**

```sql
-- Kopiraj i paste cijeli sadržaj iz:
-- portal-api/migrations/001_multi_tenant_schema.sql
-- Execute

-- Zatim kopiraj i paste:
-- portal-api/migrations/002_add_user_approval.sql
-- Execute

-- Provjeri:
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
```

Trebao bi vidjeti: users, tenants, user_tenant_roles, calls, call_analytics

Exit: `\q`

---

## Deploy Backend (Flask)

### 1. Kreiraj Backend Service

U Railway dashboardu:
1. Klikni `+ New`
2. Odaberi `GitHub Repo`
3. Odaberi `bilingual-voice-agent` repo
4. Branch: `feature/vapi-call-portal`
5. Root directory: `/portal-api`
6. Service name: `backend`

Ili CLI:

```powershell
cd portal-api
railway up
```

### 2. Postavi Environment Variables

Railway dashboard → Backend service → Variables:

```
DATABASE_URL=${{Postgres.DATABASE_URL}}
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
NEXTAUTH_SECRET=your_nextauth_secret
NEXTAUTH_URL=https://FRONTEND_URL_OVDJE.up.railway.app
NEXT_PUBLIC_API_URL=https://BACKEND_URL_OVDJE.up.railway.app
PORT=5000
```

**BITNO:**
- `DATABASE_URL=${{Postgres.DATABASE_URL}}` - Railway automatski povuče iz PostgreSQL servisa
- `NEXTAUTH_URL` - zamijenit ćeš nakon što deploy-aš frontend
- `NEXT_PUBLIC_API_URL` - kopiraj URL backend servisa

**Kopiraj Backend URL:**

Railway dashboard → Backend service → Domain → kopiraj URL

Npr: `https://bilingual-voice-agent-backend-production.up.railway.app`

### 3. Deploy Backend

Railway automatski buildа i deploya.

Provjeri logs:

```powershell
railway logs
```

Trebao bi vidjeti:
```
Running on https://0.0.0.0:5000
```

**Testiraj:**

```powershell
Invoke-WebRequest https://TVOJ_BACKEND_URL.up.railway.app/api/webhooks/vapi
```

---

## Deploy Frontend (Next.js)

### 1. Kreiraj Frontend Service

U Railway dashboardu:
1. Klikni `+ New`
2. Odaberi `GitHub Repo`
3. Repo: `bilingual-voice-agent`
4. Branch: `feature/vapi-call-portal`
5. Root directory: `/portal-ui`
6. Service name: `frontend`

### 2. Postavi Environment Variables

Railway dashboard → Frontend service → Variables:

```
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
NEXTAUTH_SECRET=your_nextauth_secret
NEXTAUTH_URL=${{RAILWAY_PUBLIC_DOMAIN}}
NEXT_PUBLIC_API_URL=https://TVOJ_BACKEND_URL.up.railway.app
NEXT_PUBLIC_USE_MOCK_DATA=false
```

**BITNO:**
- `NEXTAUTH_URL=${{RAILWAY_PUBLIC_DOMAIN}}` - Railway automatski postavlja na frontend URL
- `NEXT_PUBLIC_API_URL` - paste backend URL iz prethodnog koraka

### 3. Deploy Frontend

Railway automatski builda.

**Kopiraj Frontend URL:**

Railway dashboard → Frontend service → Domain

Npr: `https://bilingual-voice-agent-production.up.railway.app`

---

## Finalne izmjene

### 1. Update Backend NEXTAUTH_URL

Railway dashboard → Backend service → Variables:

```
NEXTAUTH_URL=https://TVOJ_FRONTEND_URL.up.railway.app
```

Redeploy backend.

### 2. Update Google OAuth

Google Cloud Console → Credentials → OAuth 2.0 Client ID:

Dodaj u **Authorized redirect URIs**:
```
https://TVOJ_FRONTEND_URL.up.railway.app/api/auth/callback/google
```

Save.

### 3. Update VAPI Webhook

VAPI dashboard → Assistant → Server Settings:

```
Webhook URL: https://TVOJ_BACKEND_URL.up.railway.app/api/webhooks/vapi
```

---

## Testiranje

### 1. Otvori Frontend

```
https://TVOJ_FRONTEND_URL.up.railway.app
```

### 2. Login

Klikni "Sign in with Google" → Odaberi account → Automatski admin

### 3. Provjeri Dashboard

Trebao bi vidjeti KPI metrics, grafove.

### 4. Provjeri Admin Panel

Klikni "Admin" → Vidiš tablicu korisnika.

### 5. Test VAPI Webhook

Napravi test poziv u VAPI → Check Railway backend logs:

```powershell
railway logs
```

Trebao bi vidjeti:
```
INFO:__main__:========== VAPI WEBHOOK ==========
```

---

## Auto-Deploy sa Git Push

Railway je već povezan sa GitHub-om. Svaki put kada pushaš na branch:

```powershell
git add .
git commit -m "Update"
git push origin feature/vapi-call-portal
```

Railway automatski:
1. Detektira promjene
2. Builda backend i frontend
3. Deploya

Možeš pratiti u Railway dashboardu ili:

```powershell
railway logs --follow
```

---

## Environment Variables - Kompletan Popis

### Backend Service

| Variable | Value | Opis |
|----------|-------|------|
| `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` | Automatski iz Railway PostgreSQL |
| `GOOGLE_CLIENT_ID` | `your_id` | Iz Google Cloud Console |
| `GOOGLE_CLIENT_SECRET` | `your_secret` | Iz Google Cloud Console |
| `NEXTAUTH_SECRET` | `your_secret` | Random string |
| `NEXTAUTH_URL` | `https://frontend-url.up.railway.app` | Frontend URL |
| `NEXT_PUBLIC_API_URL` | `https://backend-url.up.railway.app` | Backend URL |
| `PORT` | `5000` | Flask port |

### Frontend Service

| Variable | Value | Opis |
|----------|-------|------|
| `GOOGLE_CLIENT_ID` | `your_id` | Isti kao backend |
| `GOOGLE_CLIENT_SECRET` | `your_secret` | Isti kao backend |
| `NEXTAUTH_SECRET` | `your_secret` | Isti kao backend |
| `NEXTAUTH_URL` | `${{RAILWAY_PUBLIC_DOMAIN}}` | Auto-populate |
| `NEXT_PUBLIC_API_URL` | `https://backend-url.up.railway.app` | Backend URL |
| `NEXT_PUBLIC_USE_MOCK_DATA` | `false` | Production mode |

---

## Struktura na Railway

```
Railway Project: bilingual-voice-agent
├── PostgreSQL (managed database)
│   └── DATABASE_URL: postgresql://postgres:...@...railway.app/railway
│
├── Backend Service (Flask - portal-api/)
│   ├── Source: GitHub feature/vapi-call-portal
│   ├── Root: /portal-api
│   ├── Build: pip install -r requirements.txt
│   ├── Start: gunicorn server_multitenant:app
│   └── URL: https://...backend-production.up.railway.app
│
└── Frontend Service (Next.js - portal-ui/)
    ├── Source: GitHub feature/vapi-call-portal
    ├── Root: /portal-ui
    ├── Build: npm install && npm run build
    ├── Start: npm start
    └── URL: https://...production.up.railway.app
```

---

## Monitoring

### Railway Dashboard

https://railway.app/dashboard

Vidiš:
- Real-time logs
- Metrics (CPU, RAM, Network)
- Build history
- Deploy status

### Provjeri status servisa

```powershell
railway status
```

### Prati logs

```powershell
railway logs --service backend
railway logs --service frontend
railway logs --service postgres
```

---

## Troubleshooting

### Backend 500 error

1. Check logs: `railway logs --service backend`
2. Provjeri DATABASE_URL: `railway variables`
3. Provjeri da migracije su izvršene

### Frontend 404 error

1. Check build logs
2. Provjeri da `npm run build` radi lokalno
3. Provjeri environment variables

### CORS error

Provjeri da backend ima:
```python
allowed_origins = [
    "http://localhost:3000",
    "https://*.railway.app",
    "https://*.up.railway.app"
]
```

### Google OAuth error

1. Provjeri redirect URI u Google Console
2. Mora biti: `https://FRONTEND_URL/api/auth/callback/google`
3. NEXTAUTH_URL u backendu mora biti frontend URL

### Database connection error

```powershell
railway variables
```

Provjeri da `DATABASE_URL` postoji. Ako ne:

```powershell
railway link
railway add --database postgres
```

---

## Costs

Railway pricing:
- $5 besplatno mjesečno (trial credit)
- Nakon toga: pay-as-you-go
- ~$1-2/mjesec za ovaj projekt (small usage)
- PostgreSQL: $0
- Compute: $0.000463/GB-hour

Za development/testing, $5 je više nego dovoljno.

---

## Rollback

Ako nešto krene po zlu:

Railway dashboard → Service → Deployments → Odaberi prethodni → Rollback

Ili:

```powershell
railway rollback
```

---

## Custom Domain (opciono)

Ako kasnije želiš dodati vlastitu domenu:

1. Kupi domenu (Namecheap, GoDaddy)
2. Railway dashboard → Service → Settings → Domains
3. Add custom domain
4. Dodaj CNAME record u DNS postavkama domene

---

## Checklist

- [ ] Railway account kreiran
- [ ] Payment method dodan (za trial credit)
- [ ] Railway CLI instaliran
- [ ] Procfile kreiran u portal-api/
- [ ] gunicorn dodan u requirements.txt
- [ ] CORS update u server_multitenant.py
- [ ] PostgreSQL servis dodan
- [ ] Migracije izvršene na Railway bazi
- [ ] Backend service deployан
- [ ] Backend environment variables postavljene
- [ ] Frontend service deployан
- [ ] Frontend environment variables postavljene
- [ ] Google OAuth redirect URI update
- [ ] VAPI webhook update
- [ ] Testiranje - login radi
- [ ] Testiranje - admin panel radi
- [ ] Testiranje - VAPI webhook radi

---

Gotovo! Sve tri komponente su sada live na Railway. 🚀
