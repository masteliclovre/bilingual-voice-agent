# Bilingual Voice Agent

This repository contains experimental scripts for building a low-latency Croatian/English voice
assistant. The `test2` directory includes:

- `voice_agent.py` – an always-listening assistant that performs transcription (Whisper), reasoning
  (OpenAI) and speech synthesis (ElevenLabs) on a single machine.
- `client_mic.py` – a lightweight client that captures microphone audio and sends it to a remote
  automatic speech recognition (ASR) server.
- `server_asr.py` – a FastAPI service that exposes Faster-Whisper transcription for use by remote
  clients.

If you plan to serve users over the internet, read the networking guide for advice on avoiding
bandwidth bottlenecks and deploying the ASR server to the cloud: [`docs/networking.md`](docs/networking.md).
