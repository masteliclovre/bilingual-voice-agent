# ENNA Next - Voice Agent Portal

Dashboard za VAPI glasovni agent s Croatian Whisper transcriberom.

## Quick Start

```bash
cd portal-ui
npm install
npm run dev
```

Portal dostupan na: http://localhost:3000

## Konfiguracija

Portal radi s **mock podacima** bez backenda. Sve je spremno za testiranje.

`.env.local` već je konfiguriran:
```env
NEXT_PUBLIC_USE_MOCK_DATA=true
```

Za povezivanje s pravim backendom, postavi:
```env
NEXT_PUBLIC_USE_MOCK_DATA=false
```

## Build

```bash
npm run build
npm start
```
