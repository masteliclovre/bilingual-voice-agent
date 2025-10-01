# Networking and Deployment Considerations

This project ships with two reference flows for capturing microphone audio and turning it into
assistant replies:

1. `test2/client_mic.py` captures audio on a laptop/desktop and POSTs a WAV file to
   `test2/server_asr.py`, which runs a Faster-Whisper model. The server answers with text, the
   client forwards it to OpenAI for reasoning, and finally performs TTS playback.
2. `test2/voice_agent.py` performs everything locally (capture, transcription, OpenAI reasoning and
   TTS). It can optionally be configured to call out to a remote ASR service via the
   `ASR_REMOTE_URL` environment variable.

Because the sample server binds to `http://localhost:8765` by default, running both the client and
server on the same home network is convenient for testing but can become a bottleneck once you try
real-time conversations over the public internet. Home broadband links typically provide
5–25 Mbps upload, and even fibre plans rarely exceed 100–150 Mbps upstream. When you proxy the audio
through your own residential connection, every stream needs to be uploaded by the caller and then
re-uploaded by your server to the AI provider, doubling the traffic that must squeeze through the
slowest link. Latency also increases because packets make an extra hop.

## When you should move the ASR server off localhost

If you expect users to connect from outside your local network, or you plan to serve more than one
person at a time, you should host `server_asr.py` on a machine with a fast and reliable uplink—ideally
a cloud VM with a GPU that sits close (network-wise) to the OpenAI/ElevenLabs endpoints.

Key benefits of hosting the ASR server in the cloud:

- **Higher throughput** – cloud GPUs often sit behind multi-gigabit networking, so the ASR server can
  receive and respond to many users simultaneously without saturating your home connection.
- **Lower latency** – fewer network hops and shorter round trips to OpenAI reduce the delay between
  speaking and hearing the reply.
- **Availability** – cloud infrastructure can run 24/7 and is easier to monitor, secure and scale.

## How to deploy `server_asr.py` remotely

1. **Provision a host** – pick a cloud provider that offers GPU instances (e.g. AWS g5, GCP A2,
   Lambda Labs, Paperspace). Ubuntu 22.04 with CUDA drivers works well.
2. **Copy the repo** – `git clone` this project or copy the `test2` directory onto the server.
3. **Install dependencies** – create a virtual environment and install requirements:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r test2/requirements.txt
   ```
4. **Expose the API** – run the FastAPI app with Uvicorn and listen on all interfaces:
   ```bash
   uvicorn server_asr:app --host 0.0.0.0 --port 8765
   ```
   Place the process behind a reverse proxy (Nginx, Caddy, Cloudflare Tunnel, etc.) so that the
   endpoint is reachable over HTTPS (`https://your-domain/transcribe`). If you need authentication,
   add an API key check before accepting audio.
5. **Point clients to the remote server** – on every machine running `client_mic.py` set:
   ```bash
   export SERVER_URL="https://your-domain/transcribe_and_reply"
   ```
   If you expose only the ASR `/transcribe` endpoint, adjust the client to match the remote path.

## Streaming directly to OpenAI

Your friend is correct that removing unnecessary hops can improve latency. In principle you can skip
`server_asr.py` entirely and stream raw audio from the client to OpenAI. To do that you need to:

1. Use the [OpenAI Realtime or Responses API](https://platform.openai.com/docs/guides/realtime) to
   send audio chunks as they are captured. This requires a WebRTC or WebSocket connection rather than
   the current HTTP file upload.
2. Move any secret API keys out of the client—or wrap the OpenAI call behind a minimal proxy that
   only forwards audio and never reveals credentials to end users.
3. Update the client loop so that it awaits streaming transcripts/replies from OpenAI and forwards
   them to ElevenLabs (or another TTS).


See `test2/realtime_client.py` for a working implementation that replaces the HTTP upload with a
WebSocket session against OpenAI's Realtime API.

## Streaming directly to OpenAI – step by step

The new `test2/realtime_proxy.py` and `test2/realtime_client.py` give you an end-to-end example of
how to stream microphone audio to OpenAI without relaying through the Faster-Whisper server.

1. **Install dependencies** – ensure your virtual environment includes the extra `websockets`
   dependency:
   ```bash
   pip install -r test2/requirements.txt
   ```
2. **Launch the proxy (on a trusted machine)** – this process stores the OpenAI API key and opens a
   WebSocket endpoint that clients can hit without seeing the credential:
   ```bash
   export OPENAI_API_KEY="sk-..."
   export OPENAI_REALTIME_MODEL="gpt-4o-realtime-preview-2024-12-17"  # optional override
   python test2/realtime_proxy.py
   ```
   By default it listens on `0.0.0.0:8081`. Use a reverse proxy or VPN if you need to expose it on
   the public internet.
3. **Configure your client machine** – no OpenAI secrets are required on the client. Provide only
   the proxy URL (and your ElevenLabs key if you want audio playback):
   ```bash
   export REALTIME_PROXY_URL="ws://your-proxy-host:8081"
   export ELEVENLABS_API_KEY="el-..."  # optional but recommended
   python test2/realtime_client.py
   ```
4. **Talk to the assistant** – the client captures 20 ms PCM frames, streams them to the proxy as
   they are recorded, waits for the Realtime API to return streaming transcripts, and forwards the
   final reply to ElevenLabs for TTS playback.
5. **Customise behaviour** – adjust `ASSISTANT_INSTRUCTIONS`, RMS thresholds, or ElevenLabs voice by
   setting the corresponding environment variables before launching the client.

Because the audio travels straight from the microphone to OpenAI (with the proxy just relaying
bytes), you avoid double-uploading through a slow residential network link. Latency drops to what the
OpenAI Realtime service and ElevenLabs require.

## Summary

Running the audio relay on your laptop is great for prototyping, but it throttles performance when
other people connect over the internet. Deploy the ASR service to a data centre (or implement direct
streaming to OpenAI) to eliminate the home-network bottleneck your friend pointed out.
