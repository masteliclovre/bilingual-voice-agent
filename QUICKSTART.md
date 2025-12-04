# 🚀 Quick Start Guide

## Prerequisites

- Python 3.10+
- Microphone and speakers/headphones
- API keys (Groq/OpenAI + ElevenLabs)

---

## Installation (5 minutes)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup environment

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```bash
GROQ_API_KEY=gsk_your_groq_key_here
ELEVENLABS_API_KEY=sk_your_elevenlabs_key_here
```

**Get API keys:**
- Groq: https://console.groq.com/keys (FREE tier available)
- ElevenLabs: https://elevenlabs.io/app/settings/api-keys (FREE tier: 10k chars/month)

---

## Running the Voice Agent

### Terminal 1: Start Server

```bash
python server.py
```

Wait for:
```
🤖 Bilingual Voice Agent Server with Smart RAG
============================================================
├─ LLM provider: groq
├─ Model: llama-3.3-70b-versatile
├─ RAG: Enabled ✓
├─ Knowledge topics: 11
```

### Terminal 2: Start Client

```bash
python voice_agent.py
```

Wait for:
```
🎙️ Bilingual voice agent ready. (HR/EN)
```

---

## Usage

1. **Start talking** - The client is listening
2. **Pause briefly** - When you finish speaking (0.2s silence)
3. **Listen** - AI response plays back automatically

### Example conversation:

```
You: "Hello, what are your working hours?"
🤖: "Our working hours are Monday-Friday 8:00-20:00, Saturday 9:00-14:00."

You: "Hvala!"
🤖: "Nema na čemu! Rado pomažem."
```

---

## Testing RAG System

```bash
python test_smart_rag.py
```

Should show:
```
[OK] Smart RAG initialized successfully
[i] Total topics: 11
```

---

## Troubleshooting

### "Missing API key"
- Check `.env` file exists
- Verify `GROQ_API_KEY` and `ELEVENLABS_API_KEY` are set

### "No audio devices found"
List available devices:
```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

Then set in `.env`:
```bash
INPUT_DEVICE_INDEX=1  # Your device number
```

### "Connection refused"
- Make sure server is running first (`python server.py`)
- Check server URL in `.env`: `REMOTE_AGENT_URL=http://localhost:8000`

### "RAG not matching"
- Test with: `python test_smart_rag.py`
- Add keywords to [knowledge.json](knowledge.json)
- See [README.md](README.md) for customization

---

## Next Steps

✅ **Working?** Great! Try:
1. Customize [knowledge.json](knowledge.json) for your use case
2. Change voice in `.env` (`ELEVENLABS_VOICE_ID`)
3. Try different LLM models (`GROQ_MODEL`)

📞 **Want phone integration?**
- Check README "Next Steps" section
- Consider Infobip or Twilio integration

---

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Test RAG system
python test_smart_rag.py

# Run server
python server.py

# Run client
python voice_agent.py

# List audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"
```

---

Need help? Check [README.md](README.md) for detailed documentation.
