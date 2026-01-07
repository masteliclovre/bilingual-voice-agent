# Checklist Prije Git Push

## ✅ Provjeri Prije Pusha

### 1. Nebitni/Osjetljivi Fajlovi
**Provjer da NE commitaš:**

```bash
# Provjeri što će se pushati
git status

# Provjeri .gitignore
cat .gitignore
```

**NIKADA ne commitaj:**
- ❌ `.env` fajlove (sadrže DB lozinke)
- ❌ `.env.local` (sadrži API URLs)
- ❌ `call-test.json` (sadrži VAPI keys)
- ❌ `configure-vapi-webhook.json`
- ❌ `node_modules/`
- ❌ `.next/` build folder
- ❌ `__pycache__/` Python cache

**Commitaj samo:**
- ✅ `.env.template` fajlove
- ✅ Source code (.py, .tsx, .ts, .css)
- ✅ `requirements.txt`, `package.json`
- ✅ Documentation (.md fajlovi)
- ✅ `init_db.sql`

### 2. Obriši Testne Fajlove

Testni JSON fajlovi u `C:\Users\Marko\` direktoriju:
```bash
# Ovi NE TREBAJU biti u repou:
call-test.json
configure-vapi-webhook.json
use-elevenlabs-transcriber.json
call-inline.json
configure-org-webhook.json
create-assistant.json
enable-server-messages.json
patch-assistant-url.json
patch-transcriber.json
patch-transcriber-root.json
set-inbound.json
test-call.json
test-webhook.json
update-assistant.json
update-assistant-ivan.json
```

**Ovi su već u `.gitignore` tako da se neće commitati.**

### 3. Obriši Osjetljive Podatke iz Koda

Provjeri da nema hardkodanih:
- API keys
- Database passwords
- Phone numbers
- VAPI Assistant IDs
- Ngrok URLs

```bash
# Pretraži za potencijalne secrets
grep -r "sk-" . --exclude-dir={node_modules,.next,venv,__pycache__}
grep -r "Bearer" . --exclude-dir={node_modules,.next,venv,__pycache__}
grep -r "postgresql://" . --exclude-dir={node_modules,.next,venv,__pycache__}
```

### 4. Očisti Auth Komantare

**U produkciji OMOGUĆI AUTH!**

Trenutno je auth onemogućen za testiranje:
- `portal-ui/app/dashboard/page.tsx` - linija 17-21, 43-46
- `portal-ui/lib/api.ts` - linija 212-216

**Nakon testiranja, vrati auth!**

### 5. Ažuriraj README

Provjeri:
- ✅ `PORTAL_SETUP.md` - setup upute
- ✅ `QUICK_START.md` - brzi start
- ✅ `DATABASE_SCHEMA.md` - schema dokumentacija
- ✅ Glavni `README.md` - poveznice na portal upute

### 6. Test Lokalno

Prije pusha, testiraj da sve radi:

```bash
# Backend
cd portal-api
python server.py
# Provjeri: http://localhost:5000/api/calls

# Frontend
cd portal-ui
npm run dev
# Provjeri: http://localhost:3000/dashboard
```

### 7. Git Commands

```bash
# Provjeri status
git status

# Dodaj samo željene fajlove
git add portal-api/server.py
git add portal-ui/
git add *.md
git add .gitignore

# Nemoj git add . (možeš slučajno dodati .env!)

# Commit
git commit -m "Add VAPI call portal with Flask API and Next.js UI"

# Push
git push origin main
```

## 📝 Final Checklist

- [ ] `.env` fajlovi nisu u gitu
- [ ] Testni JSON fajlovi nisu u gitu
- [ ] API keys nisu hardkodirani
- [ ] Database lozinke nisu hardkodirane
- [ ] README fajlovi ažurirani
- [ ] Auth omogućen za produkciju (ili dokumentirano da je onemogućen)
- [ ] `.gitignore` postavljen pravilno
- [ ] Lokalni test uspješan

## 🚀 Nakon Pusha

Za deploy na produkciju:
1. Deploy Flask na Heroku/Railway/Render
2. Deploy Next.js na Vercel/Netlify
3. Postavi PostgreSQL na Supabase/Neon/AWS RDS
4. Omogući Auth0 autentifikaciju
5. Postavi VAPI webhook na produkcijski URL
6. Postavi environment varijable u hosting provideru
