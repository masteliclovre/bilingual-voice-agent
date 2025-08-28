import os
import base64
import requests
from flask import Flask, request, jsonify

CROATIAN_URL = os.getenv("CROATIAN_URL", "http://localhost:8000/transcribe")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Memory: callId -> "hr" or "en"
CALL_LANG = {}

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "croatian_url": CROATIAN_URL,
        "has_deepgram": bool(DEEPGRAM_API_KEY)
    })

def deepgram_transcribe(audio_b64: str) -> str:
    audio = base64.b64decode(audio_b64)
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav"
    }
    resp = requests.post(
        "https://api.deepgram.com/v1/listen?model=nova-2&language=en",
        headers=headers,
        data=audio,
        timeout=30
    )
    resp.raise_for_status()
    j = resp.json()
    return j["results"]["channels"][0]["alternatives"][0].get("transcript", "").strip()

def croatian_transcribe(payload: dict) -> str:
    resp = requests.post(CROATIAN_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json().get("text", "").strip()

@app.route("/transcribe", methods=["POST"])
def transcribe():
    body = request.get_json() or {}
    audio_b64 = body.get("audio")
    call_id = body.get("callId", "unknown")
    if not audio_b64:
        return jsonify({"error": "Missing 'audio' base64"}), 400

    lang = CALL_LANG.get(call_id)
    if not lang:
        lang = "hr"  # Default to Croatian first
        CALL_LANG[call_id] = lang

    try:
        if lang == "hr":
            text = croatian_transcribe(body)
            if not text:
                CALL_LANG[call_id] = "en"
        else:
            text = deepgram_transcribe(audio_b64)
            if not text:
                CALL_LANG[call_id] = "hr"
        return jsonify({"text": text, "lang": CALL_LANG[call_id]})
    except Exception as e:
        return jsonify({"error": str(e)}), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
