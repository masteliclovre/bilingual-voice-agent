"""FastAPI server that runs the bilingual voice agent pipeline on a GPU backend.

VERSION: 2025-10-19-instant (instant thinking sounds via streaming)
"""

import base64
import io
import os
import threading
import uuid
import random
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from openai import OpenAI
from elevenlabs import ElevenLabs

from debug_tools import install_exception_logging, log_startup_diagnostics

load_dotenv()

install_exception_logging("runpod_voice_server")
log_startup_diagnostics("runpod_voice_server")

def ensure_hf_transfer_optional():
    enable_fast = os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "").strip()
    if not enable_fast:
        return
    truthy = {"1", "true", "yes", "on"}
    if enable_fast.lower() not in truthy:
        return
    try:
        __import__("hf_transfer")
    except ModuleNotFoundError:
        print("HF_HUB_ENABLE_HF_TRANSFER is set but hf_transfer is not installed.")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

ensure_hf_transfer_optional()

# Config
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "vFQACl5nAIV0owAavYxE")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "GoranS/whisper-large-v3-turbo-hr-parla-ctranslate2")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "180"))

MAX_TURNS_IN_WINDOW = int(os.getenv("MAX_TURNS_IN_WINDOW", "12"))
SUMMARY_UPDATE_EVERY = int(os.getenv("SUMMARY_UPDATE_EVERY", "4"))

TARGET_SR = 16000
CHANNELS = 1

WAKE_WORD = os.getenv("WAKE_WORD", "").strip() or None
SERVER_AUTH_TOKEN = os.getenv("REMOTE_SERVER_AUTH_TOKEN", "").strip() or None

THINKING_PHRASES_HR = ["Hmm,", "Pa,", "Dobro,", "Hm, vidimo,", "Aha,", "Dakle,"]
THINKING_PHRASES_EN = ["Hmm,", "Well,", "Let me think,", "Okay,", "Right,", "So,"]

def _env_truthy(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

USE_THINKING_SOUNDS = _env_truthy(os.getenv("USE_THINKING_SOUNDS"), True)
ELEVEN_STREAM_LATENCY = os.getenv("ELEVEN_STREAM_LATENCY", "0")

THINKING_SOUNDS_CACHE: Dict[str, bytes] = {}

app = FastAPI(title="Bilingual Voice Agent GPU Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def load_whisper():
    kwargs = dict(
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
        cpu_threads=os.cpu_count(),
        num_workers=1,
    )
    model = WhisperModel(WHISPER_MODEL, **kwargs)
    
    if WHISPER_DEVICE == "cuda":
        print("Warming up GPU...")
        dummy = np.zeros(16000, dtype=np.float32)
        try:
            list(model.transcribe(dummy, beam_size=1, vad_filter=False))
            print("✓ GPU warmup complete")
        except Exception as e:
            print(f"Warmup warning: {e}")
    
    return model

def resample_to_16k(audio_np: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return audio_np
    target_len = int(len(audio_np) * TARGET_SR / src_sr)
    if target_len <= 0:
        return np.zeros(1, dtype=np.float32)
    from scipy.signal import resample
    return resample(audio_np, target_len).astype(np.float32)

def whisper_transcribe(whisper: WhisperModel, wav_buf: io.BytesIO):
    if wav_buf.getbuffer().nbytes < 32000:
        return "", None
    import wave

    wav_buf.seek(0)
    with wave.open(wav_buf, 'rb') as wf:
        sr = wf.getframerate()
        nframes = wf.getnframes()
        pcm = wf.readframes(nframes)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != TARGET_SR:
        audio = resample_to_16k(audio, sr)
    
    segments, info = whisper.transcribe(
        audio=audio,
        beam_size=1,
        vad_filter=True,
        temperature=0.0,
        language=None,
        condition_on_previous_text=False,
        word_timestamps=False,
        without_timestamps=True,
    )
    text = "".join(seg.text for seg in segments).strip()
    lang = getattr(info, "language", None)
    return text, lang

llm_clients: Dict[str, "LLMClient"] = {}

class LLMClient:
    def __init__(self, api_key: str):
        api_key = api_key.strip()
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for remote voice agent server.")
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.model = OPENAI_MODEL

    def complete(self, messages, temperature=OPENAI_TEMPERATURE, max_tokens=OPENAI_MAX_TOKENS):
        params = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        if max_tokens:
            params["max_tokens"] = max_tokens
        resp = self.client.chat.completions.create(**params)
        return resp.choices[0].message.content.strip()

class Memory:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.summary = ""
        self.window = []
        self.user_turns_since_summary = 0

    def add_user(self, content: str):
        self.window.append({"role": "user", "content": content})
        self.user_turns_since_summary += 1
        self._trim_window()

    def add_assistant(self, content: str):
        self.window.append({"role": "assistant", "content": content})
        self._trim_window()

    def _trim_window(self):
        if len(self.window) > MAX_TURNS_IN_WINDOW * 2:
            self.window = self.window[-MAX_TURNS_IN_WINDOW * 2:]

    def maybe_summarize(self):
        if self.user_turns_since_summary < SUMMARY_UPDATE_EVERY:
            return
        sys_prompt = (
            "You are a memory compressor. Summarize the following conversation into concise bullet points "
            "capturing user preferences, facts, goals, and unresolved tasks. Keep neutral tone. Max ~150 words."
        )
        msgs = [{"role": "system", "content": sys_prompt}]
        if self.summary:
            msgs.append({"role": "system", "content": f"Existing summary memory:\n{self.summary}"})
        msgs.extend(self.window)
        try:
            new_summary = self.llm.complete(msgs, temperature=0.2)
            self.summary = new_summary
            self.user_turns_since_summary = 0
        except Exception as exc:
            print("Memory summarize error:", exc)

    def build_prompt(self, user_lang_hint: Optional[str]):
        system = (
            "You are a concise, helpful bilingual assistant for Croatian and English. Detect the user's language "
            "and ALWAYS reply in that language. If the user mixes languages, keep their dominant language. "
            "Keep answers short for voice (2–5 sentences). Be context-aware and remember prior details from the summary."
        )
        if (user_lang_hint or "").startswith("hr"):
            system += " Prefer Croatian if the user speaks Croatian."
        else:
            system += " Prefer English if the user speaks English."
        msgs = [{"role": "system", "content": system}]
        if self.summary:
            msgs.append({"role": "system", "content": f"Conversation summary memory:\n{self.summary}"})
        msgs.extend(self.window)
        return msgs

def init_elevenlabs() -> Optional[ElevenLabs]:
    if not ELEVENLABS_API_KEY:
        return None
    return ElevenLabs(api_key=ELEVENLABS_API_KEY)

def get_llm_client(api_key_override: Optional[str]) -> LLMClient:
    key = (api_key_override or DEFAULT_OPENAI_API_KEY).strip()
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY for remote voice agent server.")
    if key not in llm_clients:
        llm_clients[key] = LLMClient(key)
    return llm_clients[key]

def elevenlabs_tts_pcm(el: ElevenLabs, text: str) -> bytes:
    audio_gen = el.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        optimize_streaming_latency=str(ELEVEN_STREAM_LATENCY),
        output_format="pcm_16000",
        text=text,
        model_id="eleven_multilingual_v2",
    )
    pcm = bytearray()
    for chunk in audio_gen:
        if chunk:
            pcm.extend(chunk)
    if not pcm:
        raise RuntimeError("ElevenLabs returned empty audio")
    return bytes(pcm)

def get_thinking_phrase(lang: Optional[str]) -> str:
    phrases = THINKING_PHRASES_HR if (lang or "").startswith("hr") else THINKING_PHRASES_EN
    return random.choice(phrases)

def preload_thinking_sounds(el: Optional[ElevenLabs]):
    global THINKING_SOUNDS_CACHE
    
    if not el:
        print("⚠️  ElevenLabs not available - thinking sounds disabled on server")
        return
    
    unique_phrases = list(set(THINKING_PHRASES_HR + THINKING_PHRASES_EN))
    
    print(f"🔊 Pre-generating {len(unique_phrases)} thinking sounds on server...")
    
    success_count = 0
    for i, phrase in enumerate(unique_phrases, 1):
        try:
            pcm = elevenlabs_tts_pcm(el, phrase)
            THINKING_SOUNDS_CACHE[phrase] = pcm
            success_count += 1
            if i % 3 == 0 or i == len(unique_phrases):
                print(f"  ... {i}/{len(unique_phrases)} cached", end="\r")
        except Exception as e:
            print(f"\n⚠️  Failed to cache '{phrase}': {e}")
    
    print(f"\n✓ Server cached {success_count}/{len(unique_phrases)} thinking sounds")

def get_cached_thinking_sound(lang: Optional[str]) -> Optional[bytes]:
    if not THINKING_SOUNDS_CACHE:
        return None
    phrase = get_thinking_phrase(lang)
    return THINKING_SOUNDS_CACHE.get(phrase)

@dataclass
class SessionState:
    memory: Memory
    api_key: str
    last_lang: Optional[str] = None
    turns: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

whisper_model = load_whisper()
eleven_client = init_elevenlabs()

if USE_THINKING_SOUNDS and eleven_client:
    preload_thinking_sounds(eleven_client)

sessions: Dict[str, SessionState] = {}

def require_auth(x_auth: Optional[str] = Header(None)):
    if SERVER_AUTH_TOKEN and x_auth != SERVER_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid auth token")

def get_session(session_id: Optional[str], api_key_override: Optional[str]) -> tuple[str, SessionState]:
    llm_client = get_llm_client(api_key_override)
    if not session_id or session_id not in sessions:
        session_id = uuid.uuid4().hex
        sessions[session_id] = SessionState(memory=Memory(llm_client), api_key=llm_client.api_key)
    else:
        state = sessions[session_id]
        if state.api_key != llm_client.api_key:
            state.memory.llm = llm_client
            state.api_key = llm_client.api_key
    state = sessions[session_id]
    return session_id, state

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/api/session")
def create_session(
    _: None = Depends(require_auth),
    x_openai_key: Optional[str] = Header(None),
):
    try:
        session_id, _ = get_session(None, x_openai_key)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"session_id": session_id}

@app.post("/api/process")
async def process_turn(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    _: None = Depends(require_auth),
    x_openai_key: Optional[str] = Header(None),
    openai_api_key: Optional[str] = Form(None),
):
    """Original single-response endpoint - kept for backwards compatibility."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    api_key_override = openai_api_key or x_openai_key
    try:
        session_id, state = get_session(session_id, api_key_override)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    wav_buf = io.BytesIO(audio_bytes)
    user_text, lang = whisper_transcribe(whisper_model, wav_buf)

    if WAKE_WORD:
        cleaned = user_text.lower().strip()
        if not cleaned.startswith(WAKE_WORD.lower()):
            return {
                "session_id": session_id,
                "text": user_text,
                "lang": lang,
                "assistant_text": "",
                "tts_audio_b64": None,
                "tts_sample_rate": TARGET_SR,
                "skipped": True,
                "reason": "wake_word_missing",
            }
        user_text = user_text[len(WAKE_WORD):].lstrip(" ,.-:") or "Hej!"

    if not user_text:
        return {
            "session_id": session_id,
            "text": "",
            "lang": lang,
            "assistant_text": "",
            "tts_audio_b64": None,
            "tts_sample_rate": TARGET_SR,
        }

    with state.lock:
        state.memory.add_user(user_text)
        messages = state.memory.build_prompt(lang)
        assistant_text = state.memory.llm.complete(messages)
        state.memory.add_assistant(assistant_text)
        state.memory.maybe_summarize()
        state.last_lang = lang or state.last_lang
        state.turns += 1

    # Generate TTS with thinking sound prepended
    audio_b64 = None
    if eleven_client:
        try:
            thinking_pcm = b""
            if USE_THINKING_SOUNDS:
                cached = get_cached_thinking_sound(lang)
                if cached:
                    thinking_pcm = cached
            
            response_pcm = elevenlabs_tts_pcm(eleven_client, assistant_text)
            combined_pcm = thinking_pcm + response_pcm
            audio_b64 = base64.b64encode(combined_pcm).decode("ascii")
        except Exception as exc:
            print("TTS error:", exc)

    return {
        "session_id": session_id,
        "text": user_text,
        "lang": lang or state.last_lang,
        "assistant_text": assistant_text,
        "tts_audio_b64": audio_b64,
        "tts_sample_rate": TARGET_SR,
    }

@app.post("/api/process_instant")
async def process_turn_instant(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    _: None = Depends(require_auth),
    x_openai_key: Optional[str] = Header(None),
    openai_api_key: Optional[str] = Form(None),
):
    """
    NEW: Instant response endpoint with thinking sound sent immediately.
    Returns JSON-lines stream:
    1. {"type": "thinking", "audio_b64": "...", "lang": "hr"}  ← INSTANT
    2. {"type": "response", "text": "...", "audio_b64": "...", "assistant_text": "..."}
    """
    import json
    
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    api_key_override = openai_api_key or x_openai_key
    
    async def generate():
        try:
            session_id_final, state = get_session(session_id, api_key_override)
        except RuntimeError as exc:
            yield f'{{"type": "error", "message": "{str(exc)}"}}\n'
            return

        wav_buf = io.BytesIO(audio_bytes)
        user_text, lang = whisper_transcribe(whisper_model, wav_buf)

        # 1. INSTANT thinking sound (immediately after transcription)
        if USE_THINKING_SOUNDS and THINKING_SOUNDS_CACHE:
            thinking_pcm = get_cached_thinking_sound(lang)
            if thinking_pcm:
                thinking_b64 = base64.b64encode(thinking_pcm).decode("ascii")
                yield json.dumps({
                    "type": "thinking",
                    "audio_b64": thinking_b64,
                    "lang": lang,
                    "sample_rate": TARGET_SR,
                }) + "\n"

        # Check wake word
        if WAKE_WORD:
            cleaned = user_text.lower().strip()
            if not cleaned.startswith(WAKE_WORD.lower()):
                yield json.dumps({
                    "type": "skipped",
                    "reason": "wake_word_missing",
                    "text": user_text,
                    "session_id": session_id_final,
                }) + "\n"
                return
            user_text = user_text[len(WAKE_WORD):].lstrip(" ,.-:") or "Hej!"

        if not user_text:
            yield json.dumps({
                "type": "empty",
                "session_id": session_id_final,
            }) + "\n"
            return

        # 2. Generate LLM response + TTS (slow part)
        with state.lock:
            state.memory.add_user(user_text)
            messages = state.memory.build_prompt(lang)
            assistant_text = state.memory.llm.complete(messages)
            state.memory.add_assistant(assistant_text)
            state.memory.maybe_summarize()
            state.last_lang = lang or state.last_lang
            state.turns += 1

        response_audio_b64 = None
        if eleven_client:
            try:
                response_pcm = elevenlabs_tts_pcm(eleven_client, assistant_text)
                response_audio_b64 = base64.b64encode(response_pcm).decode("ascii")
            except Exception as exc:
                print("TTS error:", exc)

        # 3. Send main response
        yield json.dumps({
            "type": "response",
            "session_id": session_id_final,
            "text": user_text,
            "lang": lang or state.last_lang,
            "assistant_text": assistant_text,
            "audio_b64": response_audio_b64,
            "sample_rate": TARGET_SR,
        }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)