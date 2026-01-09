# Bilingual Voice Agent

A bilingual (Croatian/English) voice assistant that captures audio from your microphone, processes it through a remote GPU-backed server, and plays back synthesized speech responses. The system intelligently detects which language you're speaking and responds in the same language.

> **Note:**  
> This project includes a locally stored copy of the Croatian Whisper speech-to-text model  
> **`whisper-large-v3-turbo-hr-parla`**  
> originally published by **GoranS** under the **Apache 2.0 License**.  
> The model is no longer available on Hugging Face, so it is distributed within this repository with proper attribution.


## 🎯 Features

### Voice Agent
- **Bilingual support** - Seamlessly handles Croatian and English
- **Remote processing** - Heavy computation runs on GPU server
- **Low latency** - Optimized for real-time conversation
- **Voice Activity Detection** - Automatically detects when you stop speaking
- **Memory system** - Maintains conversation context across turns
- **Audio feedback** - Subtle beep indicates when waiting for response

### Portal (NEW!)
- **Multi-tenant architecture** - Support for multiple organizations
- **Google OAuth authentication** - Secure sign-in with Google accounts
- **Admin panel** - Manage users and permissions
- **Call analytics** - View KPIs, call history, and performance metrics
- **VAPI webhook integration** - Receive and store call data from VAPI
- **Role-based access** - Admin, Manager, and Viewer roles

## 🏗️ Architecture

The project consists of three main components:

### 1. Voice Agent (Client + Server)
- **Client** (`test2/voice_agent.py`) – Runs locally on your machine
  - Captures microphone input
  - Performs lightweight Voice Activity Detection (VAD)
  - Sends audio to remote server
  - Plays back synthesized responses

- **Server** (`test2/server.py`) – Runs on GPU host (RunPod, Lambda Labs, etc.)
  - Speech-to-Text using Faster-Whisper
  - Language detection and reasoning via Groq/OpenAI
  - Text-to-Speech synthesis via ElevenLabs
  - Session management and conversation memory

### 2. Portal Backend (`portal-api/`)
- **Flask API** with PostgreSQL database
- Multi-tenant data isolation
- User authentication and authorization
- VAPI webhook endpoints
- Admin management APIs

### 3. Portal Frontend (`portal-ui/`)
- **Next.js 14** with App Router
- Google OAuth via NextAuth.js
- Real-time dashboard with analytics
- Admin panel for user management
- Responsive design with Tailwind CSS

## 📁 Repository Layout

```text
bilingual-voice-agent/
├── README.md                     # This file
├── SETUP_GUIDE.md               # Detailed portal setup guide
├── QUICKSTART.md                # Quick start instructions
├── DATABASE_SCHEMA.md           # Database schema documentation
│
├── test2/                       # Voice Agent (original)
│   ├── requirements.txt
│   ├── voice_agent.py           # Local client
│   ├── server.py                # Remote GPU server
│   └── .env.template
│
├── portal-api/                  # Backend API (Flask)
│   ├── server_multitenant.py   # Main Flask app
│   ├── requirements.txt
│   ├── .env.example             # Environment template
│   └── migrations/              # SQL migrations
│       ├── 001_initial_schema.sql
│       └── 002_add_user_approval.sql
│
├── portal-ui/                   # Frontend (Next.js)
│   ├── app/                     # Next.js App Router
│   │   ├── dashboard/           # Dashboard pages
│   │   ├── api/auth/            # NextAuth endpoints
│   │   └── page.tsx             # Landing page
│   ├── components/              # React components
│   │   ├── AdminSettings.tsx   # Admin panel
│   │   └── Dashboard.tsx       # Main dashboard
│   ├── lib/                     # Utilities
│   │   ├── auth.tsx             # Auth hook
│   │   └── api.ts               # API client
│   ├── package.json
│   └── .env.local.example       # Frontend environment template
│
└── models/                      # AI models
    └── whisper-large-v3-turbo-hr-parla-ctranslate2/
```

## 🚀 Getting Started

Choose your setup path:

### Portal Setup (Recommended for new users)
**For the complete call management portal with admin panel:**

See **[QUICKSTART.md](./QUICKSTART.md)** for quick setup or **[SETUP_GUIDE.md](./SETUP_GUIDE.md)** for detailed instructions.

**Prerequisites:**
- Python 3.10+
- Node.js 18+
- PostgreSQL 14+
- ngrok (for webhook testing)

### Voice Agent Setup (Original)
**For direct voice interaction without portal:**

Prerequisites:
- Python 3.10+
- Microphone and audio output

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/bilingual-voice-agent.git
   cd bilingual-voice-agent
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.template .env
   # Edit .env with your API keys and settings
   ```

### Running the Client

The client runs on your local machine and requires a remote server to be running:

```bash
python voice_agent.py
```

**Requirements:**
- `REMOTE_AGENT_URL` must be set in your `.env`
- Remote server must be running and accessible
- Microphone access

**Usage:**
1. Start the client
2. Speak naturally into your microphone
3. Pause briefly when you finish speaking
4. Listen to the response

### Running the Server

Deploy on a GPU-capable machine (RunPod, Lambda Labs, etc.):

```bash
python server.py
```

Or using uvicorn:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

**Server provides two API endpoints:**
- `POST /api/session` – Create a conversational session
- `POST /api/process` – Process audio and return response
- `GET /healthz` – Health check endpoint

## 🔧 Configuration

All configuration is done via environment variables. Copy `.env.template` to `.env` and fill in your values.

### Essential Variables

#### For Client:
```bash
REMOTE_AGENT_URL=https://your-server-url.proxy.runpod.net/
REMOTE_AGENT_TOKEN=your_auth_token
REMOTE_AGENT_API_KEY=your_groq_or_openai_key
```

#### For Server:
```bash
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
WHISPER_MODEL=GoranS/whisper-base-1m.hr-ctranslate2
WHISPER_DEVICE=cuda
ELEVENLABS_API_KEY=your_elevenlabs_key
REMOTE_SERVER_AUTH_TOKEN=your_auth_token
```

### Complete Configuration Reference

| Category | Variable | Description | Default |
|----------|----------|-------------|---------|
| **LLM** | `LLM_PROVIDER` | `groq` or `openai` | `groq` |
| | `GROQ_API_KEY` | Groq API key | - |
| | `GROQ_MODEL` | Groq model name | `llama-3.3-70b-versatile` |
| | `OPENAI_API_KEY` | OpenAI API key (if using OpenAI) | - |
| | `OPENAI_MODEL` | OpenAI model name | `gpt-4o-mini` |
| | `OPENAI_TEMPERATURE` | Sampling temperature | `0.3` |
| | `OPENAI_MAX_TOKENS` | Max response tokens | `150` |
| **STT** | `WHISPER_MODEL` | Whisper model identifier | `GoranS/whisper-base-1m.hr-ctranslate2` |
| | `WHISPER_DEVICE` | `cpu` or `cuda` | `cuda` |
| | `WHISPER_COMPUTE` | Compute precision | `float16` |
| **TTS** | `ELEVENLABS_API_KEY` | ElevenLabs API key | - |
| | `ELEVENLABS_VOICE_ID` | Voice ID to use | `vFQACl5nAIV0owAavYxE` |
| **Remote** | `REMOTE_AGENT_URL` | Server base URL | - |
| | `REMOTE_AGENT_TOKEN` | Auth token for server | - |
| | `REMOTE_AGENT_API_KEY` | API key forwarded to server | - |
| | `REMOTE_SERVER_AUTH_TOKEN` | Server-side auth token | - |
| **VAD** | `SILENCE_TIMEOUT_SECS` | Silence duration to end turn | `0.2` |
| | `MIN_SPEECH_SECS` | Minimum valid speech duration | `0.3` |
| | `RMS_THRESH` | Volume threshold for speech | `0.003` |
| | `RMS_HANGOVER` | Post-speech recording time | `0.12` |
| | `FRAME_DURATION_MS` | Audio frame size | `10` |
| **Memory** | `MAX_TURNS_IN_WINDOW` | Context window size | `8` |
| | `SUMMARY_UPDATE_EVERY` | Summary update frequency | `8` |
| **Audio** | `BEEP_DELAY_MS` | Wait-beep delay | `1000` |
| | `PREFERRED_INPUT_NAME` | Audio device name substring | - |
| | `INPUT_DEVICE_INDEX` | Specific device index | - |
| **HTTP** | `HTTP_CONNECT_TIMEOUT` | Connection timeout | `2.0` |
| | `HTTP_READ_TIMEOUT` | Read timeout | `30.0` |
| | `PORT` | Server port | `8000` |

## 🎤 Audio Device Selection

To list available audio devices:
```python
import sounddevice as sd
print(sd.query_devices())
```

Then set in `.env`:
```bash
# Use device name substring
PREFERRED_INPUT_NAME=USB

# Or use specific device index
INPUT_DEVICE_INDEX=1
```

## 🔐 Security

**Important security practices:**

1. **Never commit `.env`** - Contains sensitive API keys
2. **Use strong auth tokens** - Generate secure random tokens:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
3. **Match auth tokens** - `REMOTE_AGENT_TOKEN` (client) must equal `REMOTE_SERVER_AUTH_TOKEN` (server)
4. **Rotate keys regularly** - Change API keys periodically
5. **HTTPS only** - Use HTTPS for remote server URLs

## 🐛 Troubleshooting

### "REMOTE_AGENT_URL is not set" error
- Ensure `.env` file exists in the project directory
- Verify `REMOTE_AGENT_URL` is uncommented and set
- Check for extra spaces or quotes

### "Remote agent unavailable" error
- Verify server is running: `curl https://your-url/healthz`
- Check `REMOTE_AGENT_TOKEN` matches `REMOTE_SERVER_AUTH_TOKEN`
- Ensure server URL ends with `/` (e.g., `https://example.com/`)

### Audio device issues
- List devices: `python -m sounddevice`
- Set `INPUT_DEVICE_INDEX` or `PREFERRED_INPUT_NAME`
- Check microphone permissions (especially on macOS)
- Try running without `latency='low'` setting

### Slow responses
- Use Groq instead of OpenAI for faster inference
- Use smaller model: `llama-3.1-8b-instant`
- Reduce `OPENAI_MAX_TOKENS`
- Check GPU availability on server

### "Missing API key" errors
- Verify all required keys are set in `.env`
- Check for typos in variable names
- Ensure no extra spaces around `=` in `.env`

### Whisper model download issues
- Enable fast downloads: `HF_HUB_ENABLE_HF_TRANSFER=1`
- Install: `pip install hf_transfer`
- Check internet connection
- Verify model name is correct

## 📚 API Reference

### Server Endpoints

#### `GET /healthz`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "llm_provider": "groq",
  "llm_model": "llama-3.3-70b-versatile"
}
```

#### `POST /api/session`
Create a new conversation session.

**Headers:**
- `X-Auth`: Authentication token (if configured)
- `X-API-Key`: Optional API key override

**Response:**
```json
{
  "session_id": "abc123..."
}
```

#### `POST /api/process`
Process audio and get response.

**Form Data:**
- `audio`: WAV file (PCM16, 16kHz, mono)
- `session_id`: Session ID from `/api/session`
- `api_key`: Optional API key override

**Headers:**
- `X-Auth`: Authentication token (if configured)

**Response:**
```json
{
  "session_id": "abc123...",
  "text": "What you said",
  "lang": "hr",
  "assistant_text": "Response text",
  "tts_audio_b64": "base64_encoded_audio",
  "tts_sample_rate": 16000
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test locally with both client and server
5. Commit: `git commit -am 'Add feature'`
6. Push: `git push origin feature-name`
7. Submit a pull request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/bilingual-voice-agent.git
cd bilingual-voice-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure for local development
cp .env.template .env
# Edit .env with test API keys
```

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) for speech recognition
- [Groq](https://groq.com) for fast LLM inference
- [ElevenLabs](https://elevenlabs.io) for text-to-speech
- Croatian Whisper models by GoranS

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/masteliclovre/bilingual-voice-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/masteliclovre/bilingual-voice-agent/discussions)

---