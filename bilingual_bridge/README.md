# bilingual-voice-agent

## 1) Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-key.json
cp .env.example .env && edit values
```


## 2) Run locally
```
uvicorn app:app --host 0.0.0.0 --port 8000
```


### 2a) Expose locally via ngrok (for Twilio)
Install ngrok, then:
```
ngrok http 8000
```
Copy the HTTPS host, and use the **`wss://`** URL form for Twilio, e.g. `wss://YOUR-ID.ngrok.io/twilio/hr`.
> ngrok automatically supports WebSocket upgrades over HTTPS. If Twilio fails to connect, ensure your firewall allows inbound from Twilio and that your server returns 101 Switching Protocols.


## 3) Twilio Media Streams (two legs)
You can start a bidirectional stream using TwiML `<Connect><Stream>` or `<Start><Stream>`:


**TwiML example (bidirectional stream):**
```xml
<Response>
<Connect>
<Stream url="wss://YOUR-HOST/twilio/hr"/>
</Connect>
</Response>
```
Messages you can send back to Twilio over the WebSocket: `media` (with base64 mu-law 8k), `mark`, and `clear` (flushes buffered audio).


When both legs are connected to `/twilio/hr` and `/twilio/en`, start the bridge:
```
curl -X POST https://YOUR-HOST/bridge/start
```


## 4) Glossary (Google Translate v3)
Use a CSV like:
```
Rimac,Rimac
e-Građanin,e-Građanin
Zagreb,Zagreb
```
Upload to GCS, then create a glossary resource (one-time):
```bash
gcloud auth login
PROJECT=$GCP_PROJECT_ID
BUCKET=gs://YOUR_BUCKET
GLOSSARY_ID=hr-en-domain
CSV_URI=$BUCKET/glossary.csv


gcloud alpha translate glossaries create $GLOSSARY_ID \
--source-language=hr --target-language=en \
--input-uri=$CSV_URI --project=$PROJECT --location=global
```
Set `GLOSSARY_ID` in code by constructing `Translator(project_id, glossary_id="hr-en-domain")`.


## 5) Cloud Run deploy (recommended EU latency)
```bash
gcloud config set project $GCP_PROJECT_ID
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/hr-en-bridge:latest
gcloud run deploy hr-en-bridge \
--image gcr.io/$GCP_PROJECT_ID/hr-en-bridge:latest \
--platform managed \
--region europe-west4 \
--allow-unauthenticated \
--port 8080
```
Set env vars in Cloud Run console: `DEEPGRAM_API_KEY`, `AZURE_TTS_KEY`, `AZURE_TTS_REGION`, `AZURE_TTS_EN_VOICE`, `AZURE_TTS_HR_VOICE`, `GCP_PROJECT_ID`.


## 6) Low-latency tips
- TTS is synthesized directly as `raw-8khz-8bit-mulaw-mono`; we send in 20ms chunks.
- Interim mode enabled: we speak chunks on punctuation or every ~12 riječi; on correction or barge-in we send `clear` to Twilio before new audio.
- Keep both legs in EU regions where possible (Cloud Run europe-west4, etc.).