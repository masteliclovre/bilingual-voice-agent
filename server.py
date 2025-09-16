# server.py
import io, os, time, uvicorn
from typing import Optional, Dict
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf

# ============ Config ============
MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3")     # or "medium" / "small" for cheaper VRAM
DEVICE = "cuda" if os.getenv("FORCE_CPU", "0") != "1" else "cpu"
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")     # use "int8_float16" if VRAM is tight
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT",
    "You are a concise, friendly voice assistant. Keep answers short unless asked to elaborate."
)

# ============ App ============
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Warm Whisper once (loads on GPU)
print("Loading faster-whisper on", DEVICE, COMPUTE_TYPE, "…")
whisper = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

# ============ Helpers ============
def read_wav_pcm16(buf: bytes):
    """Return mono float32 @ native sr"""
    data, sr = sf.read(io.BytesIO(buf), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)  # robust mono fold
    return mono, sr

def openai_reply(user_text: str, summary: str = "") -> str:
    """Use OpenAI for the text reply. Replace with your own LLM if you like."""
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        return "(No OPENAI_API_KEY set on server.)"

    # Lightweight, low-latency formatting
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *( [{"role": "system", "content": f"Conversation summary so far: {summary}"}] if summary else [] ),
        {"role": "user", "content": user_text},
    ]
    t0 = time.perf_counter()
    resp = openai.chat.completions.create(  # works with the Chat Completions API
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.5,
        max_tokens=256,
    )
    dt = time.perf_counter() - t0
    reply = resp.choices[0].message.content.strip()
    return reply, dt

# ============ Route ============
@app.post("/transcribe_and_reply")
async def transcribe_and_reply(
    audio: UploadFile = File(...),
    summary: Optional[str] = Form(default="")
):
    t_total0 = time.perf_counter()

    # ---------- Read audio ----------
    audio_bytes = await audio.read()
    samples, sr = read_wav_pcm16(audio_bytes)

    # ---------- Transcribe (GPU) ----------
    t0 = time.perf_counter()
    # Use fast settings; tweak for accuracy vs speed
    segments, info = whisper.transcribe(
        samples,
        language=None,             # autodetect
        vad_filter=True,
        condition_on_previous_text=False,
        temperature=0.0,
        beam_size=1,               # 1 = greedy; fastest
        best_of=1,
        no_speech_threshold=0.6,
    )
    text_parts = []
    for seg in segments:
        text_parts.append(seg.text)
    transcript = " ".join(s.strip() for s in text_parts).strip()
    language = info.language if hasattr(info, "language") else None
    whisper_sec = time.perf_counter() - t0

    # ---------- LLM reply ----------
    t1 = time.perf_counter()
    reply, llm_sec = openai_reply(transcript, summary or "")
    t2 = time.perf_counter()

    out = {
        "transcript": transcript,
        "reply": reply,
        "language": language,
        "latency": {
            "whisper_sec": round(whisper_sec, 3),
            "llm_sec": round(llm_sec, 3),
            "total_sec": round(time.perf_counter() - t_total0, 3),
        },
    }
    return out

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8765"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, workers=1)
