# Pokretanje Portal API-ja za Vapi webhookove

## Korak 1: Pokreni Flask API

```bash
cd portal-api
python server.py
```

API će biti dostupan na http://localhost:5000

## Korak 2: Izloži API javno (za Vapi webhookove)

### Opcija A: Ngrok (preporučeno za testiranje)

1. Preuzmi ngrok: https://ngrok.com/download
2. Pokreni:
```bash
ngrok http 5000
```

3. Ngrok će ti dati javni URL, npr: `https://abc123.ngrok.io`

### Opcija B: Localhost.run (bez instalacije)

```bash
ssh -R 80:localhost:5000 nokey@localhost.run
```

## Korak 3: Konfiguriraj Vapi webhook

Nakon što dobiješ javni URL (npr. https://abc123.ngrok.io), dodaj webhook na Vapi:

```bash
curl.exe -X PATCH "https://api.vapi.ai/assistant/b128ee37-46ef-4937-9f56-ff0695f90a4a" ^
  -H "Authorization: Bearer 96976ec0-80ad-4c54-98a3-f993c6555e4b" ^
  -H "Content-Type: application/json" ^
  -d "{\"serverUrl\": \"https://abc123.ngrok.io/api/webhooks/vapi\"}"
```

## Korak 4: Testiraj

Napravi poziv i provjeri logove u Flask konzoli.

---

## Za produkciju:

Treba hostati Flask API na cloud servisu (npr. Railway, Render, Fly.io, ili VPS) i koristiti pravi domen.
