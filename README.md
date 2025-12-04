# 🎙️ Bilingual Voice Agent with Smart RAG

Bilingual (Croatian/English) voice assistant with instant knowledge retrieval - lightweight, fast, and easy to customize.

---

## 🌟 Features

- **🌍 Bilingual Support** - Seamlessly handles Croatian and English with automatic language detection
- **⚡ Smart RAG** - Instant pattern-based knowledge retrieval (< 1ms, no heavy dependencies)
- **🗣️ Voice Processing** - Speech-to-Text (Faster-Whisper) + Text-to-Speech (ElevenLabs)
- **🧠 LLM Integration** - Groq or OpenAI for natural language understanding
- **🎯 Low Latency** - Optimized for real-time conversation
- **💾 Memory System** - Maintains conversation context across turns
- **📦 Lightweight** - No vector databases, no embedding models

---

## 🏗️ Architecture

```
User speaks → Whisper STT → Smart RAG → LLM → ElevenLabs TTS → Audio playback
                              ↓
                        Pattern Match
                        (instant 90%+)
```

**Components:**
- **Client** - Runs locally, captures audio and plays responses
- **Server** - GPU-powered backend for speech processing and AI reasoning

---

## 📁 Project Structure

```
.
├── server.py              # FastAPI server with Smart RAG
├── voice_agent.py         # Voice client (microphone + audio playback)
├── smart_rag.py           # Lightweight RAG engine
├── knowledge.json         # Knowledge base (easily customizable)
├── test_smart_rag.py      # Test suite for RAG
├── .env.example           # Environment variables template
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

**Required API Keys:**
- `GROQ_API_KEY` or `OPENAI_API_KEY` - LLM provider
- `ELEVENLABS_API_KEY` - Text-to-speech

### 3. Test Smart RAG

```bash
python test_smart_rag.py
```

### 4. Run Server

```bash
python server.py
```

Server will start on `http://localhost:8000`

### 5. Run Voice Client (in another terminal)

```bash
python voice_agent.py
```

**Client usage:**
- Speak naturally into your microphone
- A short pause ends your turn
- Listen to the AI response
- Press Ctrl+C to exit

---

## ⚙️ Configuration (.env)

### LLM Configuration

```bash
# LLM Provider: "groq" or "openai"
LLM_PROVIDER=groq

# Groq Settings
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# OpenAI Settings (alternative)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.3
OPENAI_MAX_TOKENS=150
```

### Voice Processing

```bash
# Whisper ASR
WHISPER_MODEL=GoranS/whisper-base-1m.hr-ctranslate2
WHISPER_DEVICE=cpu  # or "cuda" for GPU
WHISPER_COMPUTE=int8

# ElevenLabs TTS
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_VOICE_ID=vFQACl5nAIV0owAavYxE
```

### Smart RAG Settings

```bash
# Enable/Disable RAG
ENABLE_RAG=true

# Path to knowledge base
KNOWLEDGE_PATH=knowledge.json

# Direct RAG answers (skip LLM for perfect matches)
RAG_DIRECT_ANSWER=false
```

**RAG_DIRECT_ANSWER modes:**
- `true` - Direct RAG responses (faster, cheaper)
- `false` - Use RAG as context for LLM (more natural)

---

## 📚 Smart RAG Knowledge Base

### Default Topics

The knowledge base includes 11 topics out of the box:

| Topic | Keywords | Use Case |
|-------|----------|----------|
| `greeting` | hello, hi, bok | Greetings |
| `hours` | working hours, radno vrijeme | Business hours |
| `contact` | email, phone, kontakt | Contact information |
| `pricing` | price, cost, cijena | Pricing info |
| `support` | help, problem, pomoć | Technical support |
| `shipping` | delivery, dostava | Shipping information |
| `returns` | refund, povrat | Returns and refunds |
| `payment` | payment, plaćanje | Payment methods |
| `location` | address, lokacija | Physical location |
| `thanks` | thank, hvala | Thanks/gratitude |
| `goodbye` | bye, doviđenja | Farewells |

### Adding New Topics

Edit [knowledge.json](knowledge.json):

```json
{
  "your_topic_id": {
    "patterns": [
      "\\b(keyword1|keyword2)\\b",
      "\\b(ključna riječ)\\b"
    ],
    "keywords": ["keyword1", "keyword2", "ključna riječ"],
    "responses": {
      "hr": "Odgovor na hrvatskom...",
      "en": "Response in English..."
    },
    "priority": 8
  }
}
```

**Priority levels:** 1-10 (higher = more important during matching)

### Programmatic API

```python
from smart_rag import SmartRAG

rag = SmartRAG()

# Add new topic
rag.add_topic(
    topic="warranty",
    patterns=[r"\b(warranty|guarantee)\b", r"\b(garancija)\b"],
    keywords=["warranty", "garancija"],
    response_hr="Nudimo 2 godine garancije.",
    response_en="We offer 2 years warranty.",
    priority=7
)

# Save to file
rag.save_knowledge("knowledge.json")
```

---

## 🔌 API Endpoints

### Health Check

```bash
GET /healthz
```

Response:
```json
{
  "status": "ok",
  "llm_provider": "groq",
  "llm_model": "llama-3.3-70b-versatile",
  "rag": "enabled",
  "rag_topics": 11
}
```

### RAG Statistics

```bash
GET /api/rag/stats
```

Response:
```json
{
  "total_topics": 11,
  "total_patterns": 22,
  "total_keywords": 38,
  "topics": ["greeting", "hours", "contact", ...]
}
```

### Process Audio

```bash
POST /api/process
```

**Form Data:**
- `audio` - WAV file (PCM16, 16kHz, mono)
- `session_id` - Session ID from `/api/session`

**Response:**
```json
{
  "session_id": "...",
  "text": "user transcription",
  "lang": "hr",
  "assistant_text": "response",
  "tts_audio_b64": "base64_audio",
  "rag_used": true,
  "rag_topic": "pricing",
  "rag_confidence": 0.85
}
```

### Streaming Audio (Low Latency)

```bash
POST /api/process_stream
```

Streams TTS audio chunks as they're generated for 35% faster playback.

---

## 🎯 Performance

| Metric | Value |
|--------|-------|
| Cold start | < 5s (no model downloads) |
| RAG matching | < 1ms |
| Match accuracy | 95%+ for expected queries |
| Memory usage | ~200MB (lightweight) |
| Cost | Minimal (no vector DB, free Groq tier) |

---

## 🐛 Troubleshooting

### "RAG not matching expected queries"

1. Check [knowledge.json](knowledge.json) patterns
2. Add more keywords
3. Increase `priority` for that topic
4. Test with `python test_smart_rag.py`

### "Wrong language detected"

Smart RAG uses heuristics (Croatian chars: č, ć, ž, š, đ + common words).

Force language:
```python
match = rag.match("text", lang="hr")  # Force Croatian
```

### "Missing API key" errors

Verify all required keys in `.env`:
- `GROQ_API_KEY` or `OPENAI_API_KEY`
- `ELEVENLABS_API_KEY`

---

## 💡 Best Practices

### Pattern Design

```python
# Good - flexible regex
r"\b(working hours|business hours|when.*open)\b"

# Bad - too specific
r"^What are your exact working hours\?$"
```

### Keywords Strategy

- Include both EN and HR variants
- Add common typos
- Use synonyms (delivery/shipping, dostava/isporuka)

### Priority Usage

- **10** = Critical (greeting, emergency)
- **7-9** = Important (contact, support)
- **5-6** = Normal (thanks, goodbye)
- **1-4** = Low priority

### Response Quality

- Keep it short (2-5 sentences for voice)
- Be specific (include numbers, dates)
- Make it actionable

---

## 📝 Example Use Cases

### E-commerce
- Product availability
- Shipping tracking
- Return policies
- Size guides

### Restaurant
- Reservations
- Menu information
- Opening hours
- Location

### Banking (Future Enhancement)
- Account balance
- Transaction history
- Card activation
- Branch locations

---

## 🔮 Next Steps

The current implementation is ready for:
1. ✅ Local voice testing
2. ✅ Remote server deployment
3. ✅ Custom knowledge base

**Future enhancements:**
- 📞 **Phone integration** (Infobip/Twilio)
- 📱 **WhatsApp/Viber support** (via Infobip)
- 🔊 **Better voice activity detection**
- 🌐 **Multi-language expansion**

---

## 🙏 Acknowledgments

- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) for speech recognition
- [Groq](https://groq.com) for fast LLM inference
- [ElevenLabs](https://elevenlabs.io) for text-to-speech
- Croatian Whisper models by GoranS

---

## 📄 License

[Add your license here]

---

Made with ❤️ for bilingual voice experiences
