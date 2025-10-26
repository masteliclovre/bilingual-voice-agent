# Bilingual Voice Agent

This repository contains a bilingual (Croatian/English) voice assistant that captures audio from a
local microphone, streams the utterance to a remote GPU-backed inference service, and plays the
synthesised speech response. The project is split into two main pieces:

- **Client** – captures microphone input, performs lightweight voice-activity detection (VAD),
  sends speech to the remote server, and plays back the answer.
- **Server** – runs on a GPU host, performs speech-to-text (Faster-Whisper), reasoning via an LLM
  (Groq or OpenAI), and text-to-speech synthesis (ElevenLabs).

## Repository layout

```text
bilingual-voice-agent/
├── README.md
├── test2/
|   ├── requirements.txt          # Python dependencies for both client and server
│   ├── voice_agent.py
│   ├── server.py
│   └── .gitignore
```

## Getting started

The client and server share the same dependency set. You can install the requirements into a Python
3.10+ virtual environment:

```bash
cd bilingual-voice-agent/
cd test2/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with the configuration relevant to the component you plan to
run. See the environment variable reference below for details.

### Running the local client

The client is designed to run on your laptop/desktop. It requires access to a microphone and audio
output device:

```bash
python -m voice_agent
```

The client will refuse to start until you configure a remote server URL in your `.env` file (see
`REMOTE_AGENT_URL`). While it listens for speech it provides a short “beep” if the remote server takes
more than a couple of seconds to respond.

### Running the GPU server

Deploy the FastAPI application on a GPU-capable machine (RunPod, Lambda Labs, etc.). You can run it
locally for development:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

The server exposes two endpoints under `/api` that the client consumes:

- `POST /api/session` – creates a conversational session and returns a `session_id`
- `POST /api/process` – accepts a PCM16 WAV file and returns transcription, response text, and
  optional TTS audio

### Environment variables

The project reads configuration from the environment (via [`python-dotenv`](https://saurabh-kumar.com/python-dotenv/)).
Set the variables in a local `.env` file.

#### Shared / client-side

| Variable | Description |
| --- | --- |
| `REMOTE_AGENT_URL` | Base URL of the remote server (e.g. `https://example.runpod.run`) |
| `REMOTE_AGENT_TOKEN` | Optional auth token that will be passed as the `X-Auth` header |
| `REMOTE_AGENT_OPENAI_KEY` | Optional API key forwarded to the server for model access |
| `FRAME_DURATION_MS` | Size of audio frames used by the VAD (default: `15`) |
| `MAX_UTTERANCE_SECS` | Maximum length of a captured utterance (default: `45`) |
| `SILENCE_TIMEOUT_SECS` | How long to wait before ending an utterance once silence is detected (default: `0.2`) |
| `MIN_SPEECH_SECS` | Minimum duration of speech that counts as a valid utterance (default: `0.3`) |
| `RMS_THRESH` | RMS threshold for VAD triggering (default: `0.003`) |
| `RMS_HANGOVER` | Additional frames to keep recording after silence (default: `0.12`) |
| `BEEP_DELAY_MS` | Delay before the waiting-beep is played (default: `2000`) |
| `PREFERRED_INPUT_NAME` | Optional substring of the audio input device to use |
| `INPUT_DEVICE_INDEX` | Explicit index of the audio input device to use |

#### Server-side

| Variable | Description |
| --- | --- |
| `LLM_PROVIDER` | `groq` (default) or `openai` |
| `GROQ_API_KEY` / `OPENAI_API_KEY` | Default API key for the chosen provider |
| `GROQ_MODEL` / `OPENAI_MODEL` | Model identifier to use for chat completions |
| `ELEVENLABS_API_KEY` | API key for ElevenLabs TTS (optional but recommended) |
| `ELEVENLABS_VOICE_ID` | Voice ID to use for ElevenLabs TTS |
| `WHISPER_MODEL` | Faster-Whisper model name (default: `GoranS/whisper-base-1m.hr-ctranslate2`) |
| `WHISPER_DEVICE` | `cpu` or `cuda` |
| `WHISPER_COMPUTE` | Compute precision, e.g. `int8`, `float16` |
| `OPENAI_TEMPERATURE` | Sampling temperature when generating responses (default: `0.3`) |
| `OPENAI_MAX_TOKENS` | Max tokens for the response (default: `150`) |
| `MAX_TURNS_IN_WINDOW` | Number of turns kept in the rolling context window (default: `8`) |
| `SUMMARY_UPDATE_EVERY` | How frequently to refresh the long-term memory summary (default: `8`) |
| `REMOTE_SERVER_AUTH_TOKEN` | Optional shared secret for protecting the API |
| `PORT` | Port that the FastAPI application should bind to (default: `8000`) |

### Troubleshooting

- Use `python -m sounddevice` to list available audio devices if the client cannot select the right
  microphone.
- Set `REMOTE_AGENT_TOKEN` on both the client and server to protect the API when exposing it over the
  internet.
- The server caches language-model clients per API key. Restart the process if you rotate keys.

## Contributing

1. Fork and clone the repository.
2. Create a feature branch and make your changes inside the `bilingual_voice_agent/test2` package.
3. Run the client/server locally to ensure everything works as expected.
4. Submit a pull request with a clear description of the changes.
