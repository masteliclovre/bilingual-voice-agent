# 🚀 Runpod Upload Guide - WinSCP metoda

## **NAJLAKŠI NAČIN: WinSCP (5 minuta)**

---

## **Korak 1: Download WinSCP**

**Link:** https://winscp.net/eng/download.php

Preuzmi **Installation package** (5.2 MB)

---

## **Korak 2: Instalacija**

1. Pokreni installer
2. Klikni **Next** → **Next** → **Install** → **Finish**
3. WinSCP će se automatski otvoriti

---

## **Korak 3: Login Setup**

Popuni formu:

```
File protocol:        SCP
Host name:            213.173.108.139
Port number:          11527
User name:            root
Password:             (ostavi prazno)
```

### **Private Key Setup:**

1. Klikni **Advanced...**
2. Sidebar lijevo: **SSH** → **Authentication**
3. **Private key file:** Klikni **...** (browse button)
4. Navigate do: `C:\Users\Marko\.ssh\id_ed25519`
5. WinSCP će pitati "Convert to PuTTY format?" → Klikni **OK**
6. Spremi konvertirani key (npr. `id_ed25519.ppk`)
7. Klikni **OK** (zatvori Advanced settings)

---

## **Korak 4: Login**

1. Klikni **Save** (da sačuvaš settings)
2. **Site name:** Runpod GPU
3. Klikni **OK**
4. Klikni **Login**

**Ako pita "Continue connecting?"** → **Yes**

---

## **Korak 5: Upload Fileova**

Sad vidiš **dual-pane interface**:
- **Lijevo:** Tvoj PC
- **Desno:** Runpod server

### **Upload procedure:**

1. **Lijevo:** Navigate do:
   ```
   C:\Users\Marko\.claude-worktrees\banking-voice-agent-rag-complete\bold-jemison
   ```

2. **Desno:** Navigate do:
   ```
   /workspace/
   ```

3. **Drag & Drop** ove fileove s lijeve na desnu stranu:
   ```
   ✓ smart_rag.py
   ✓ knowledge.json
   ✓ server.py
   ✓ .env.runpod
   ```

✅ **Upload completed!**

---

## **Korak 6: Setup na serveru**

1. U WinSCP-u, klikni **Commands** → **Open Terminal** (ili Ctrl+T)

2. Terminal će se otvoriti - upiši:

```bash
# 1. Rename .env
mv .env.runpod .env

# 2. Install dependencies
pip install fastapi uvicorn faster-whisper openai elevenlabs python-dotenv scipy numpy

# 3. Start server
python server.py
```

---

## **Očekivani Output:**

```
============================================================
🤖 Bilingual Voice Agent Server with Smart RAG
============================================================
├─ LLM provider: groq
├─ Model: llama-3.1-8b-instant
├─ RAG: Enabled ✓
├─ Knowledge topics: 11
└─ Topics: greeting, hours, contact, pricing, support...
============================================================
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

✅ **SERVER JE AKTIVAN!**

---

## **Korak 7: Update Local Client**

### **Otvori:**
```
C:\Users\Marko\.claude-worktrees\banking-voice-agent-rag-complete\bold-jemison\.env.local
```

### **Update URL:**
```bash
# Stara vrijednost:
REMOTE_AGENT_URL=https://nd4bzk6mu8qr0h-8000.proxy.runpod.net/

# NOVA vrijednost (provjeri u Runpod dashboard):
REMOTE_AGENT_URL=https://YOUR-NEW-RUNPOD-ID-8000.proxy.runpod.net/
```

**Gdje naći URL:**
1. Runpod dashboard → Tvoj pod
2. **Connect** → **HTTP Services [Port 8000]**
3. Kopiraj URL

---

## **Korak 8: Test**

### **Na Runpodu (u terminalu):**
```bash
curl http://localhost:8000/healthz
```

Output:
```json
{
  "status": "ok",
  "rag": "enabled",
  "rag_topics": 11
}
```

### **Lokalno (s klijentom):**
```bash
# Navigate do voice agent foldera
cd C:\Users\Marko\Desktop\AGENTT\voice-test\test2

# Kopiraj .env.local
copy C:\Users\Marko\.claude-worktrees\banking-voice-agent-rag-complete\bold-jemison\.env.local .env

# Run voice agent
python voice_agent.py
```

---

## **TROUBLESHOOTING**

### **"Authentication failed"**
- Check da li si dobro konvertirao SSH key
- Probaj ponovno: Advanced → SSH → Authentication → Browse key

### **"Cannot find module 'smart_rag'"**
- Provjeri da su svi fileovi u `/workspace/`
- Run: `ls /workspace/` u terminalu

### **"ModuleNotFoundError: elevenlabs"**
- Dependencies nisu instalirani
- Run ponovo: `pip install fastapi uvicorn faster-whisper openai elevenlabs python-dotenv scipy numpy`

### **Server se restartuje**
- Normalno ako ima error
- Check output za greške
- Možda fali neki file

---

## **QUICK REFERENCE**

### **WinSCP Login Info:**
```
Host:     213.173.108.139
Port:     11527
User:     root
Key:      C:\Users\Marko\.ssh\id_ed25519
```

### **Files to upload:**
```
smart_rag.py
knowledge.json
server.py
.env.runpod
```

### **Server start command:**
```bash
cd /workspace
python server.py
```

### **Stop server:**
```
Ctrl+C (u terminalu)
```

---

## **SUMMARY**

```
✅ Download WinSCP
✅ Setup connection (213.173.108.139:11527)
✅ Upload 4 files
✅ Open terminal
✅ Run setup commands
✅ Server started
✅ Update local .env
✅ Test with voice_agent.py
```

**Total time:** 5-7 minuta

---

🎉 Gotovo! Imaš Smart RAG na Runpodu!
