# bilingual-voice-agent

# 1) Setup
# python -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt
# export GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-key.json
# cp .env.example .env && edit values
#
# 2) Run locally
# uvicorn app:app --host 0.0.0.0 --port 8000
# (Expose with ngrok/cloudflared if testing Twilio)
#
# 3) Twilio Media Streams (two legs)
# - Configure two call legs (e.g., two <Dial> legs into your app) each with <Start><Stream url="wss://YOUR_HOST/twilio/hr"></Stream></Start>
# and the other with /twilio/en. Alternatively, two separate numbers each streaming to one endpoint.
# - Media format: audio/x-mulaw, 8kHz, base64 frames (~20ms).
# - When both legs are connected, POST http(s)://YOUR_HOST/bridge/start to begin bridging.
#
# 4) Cloud Run deploy (recommended EU latency)
# gcloud auth login
# gcloud config set project $GCP_PROJECT_ID
# gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/hr-en-bridge:latest
# gcloud run deploy hr-en-bridge \
# --image gcr.io/$GCP_PROJECT_ID/hr-en-bridge:latest \
# --platform managed \
# --region europe-west4 \
# --allow-unauthenticated \
# --port 8080
# # Set env vars in Cloud Run console: DEEPGRAM_API_KEY, AZURE_TTS_KEY, AZURE_TTS_REGION, ...
#
# Notes
# - Barge-in: čim dođe RMS>threshold s druge strane, šalje se Twiliu {"event":"clear"}.
# - Latenciju podešavaj chunkanjem (20ms) i radom samo na final ASR rezultatima.
# - Za još niži lag: sintetiziraj na interim segmentima s punjenjem buffer-a i prekidaj clear-om.