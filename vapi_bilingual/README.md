Clone + env: set DEEPGRAM_API_KEY, YOUR_VAPI_PUBLIC_KEY, ASSISTANT_ID, and ElevenLabs in Vapi provider keys.

Run GPU HR STT: docker compose up --build (or run hr-stt on a GPU VM).

Expose router: ngrok http 3001 \u2192 set wss://.../router in the assistant or in UI overrides.

Open apps/preview-ui/index.html and click Start. Speak HR or EN — router locks and streams transcripts back.

Latency tuning:

Deepgram: endpointing, interim_results options.

HR STT: reduce CHUNK_SECONDS (0.5–0.8s) as GPU allows; use float16 on CUDA.

ElevenLabs in Vapi: pick Multilingual v2/Flash voices for quicker TTFB (set voice in UI).

Phone numbers: Import your Twilio/Telnyx number into Vapi. (Vapi is BYO-number.)