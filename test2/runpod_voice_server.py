"""FastAPI server that runs the bilingual voice agent pipeline on a GPU backend.

Run this on RunPod (or any GPU host) and point REMOTE_AGENT_URL from the
voice_agent.py client to this server. The server keeps per-session memory so the
conversation stays coherent across turns. Audio is exchanged as 16 kHz mono WAV
(PCM16) payloads to keep latency low.
"""

import base64
import io
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from openai import OpenAI
from elevenlabs import ElevenLabs

from debug_tools import install_exception_logging, log_startup_diagnostics

load_dotenv()

install_exception_logging("runpod_voice_server")
log_startup_diagnostics("runpod_voice_server")

def ensure_hf_transfer_optional():
    """Disable the Hugging Face fast-transfer path if the package is missing."""

    enable_fast = os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "").strip()
    if not enable_fast:
        return

    truthy = {"1", "true", "yes", "on"}
    if enable_fast.lower() not in truthy:
        return

    try:
        __import__("hf_transfer")
    except ModuleNotFoundError:
        print(
            "HF_HUB_ENABLE_HF_TRANSFER is set but hf_transfer is not installed. "
            "Falling back to the standard downloader."
        )
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


ensure_hf_transfer_optional()

# =========================
# Config (mirrors voice_agent.py)
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
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

app = FastAPI(title="Bilingual Voice Agent GPU Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# =========================
# Utilities shared with voice_agent
# =========================


def load_whisper():
    kwargs = dict(
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
        cpu_threads=os.cpu_count(),
        num_workers=2,
    )
    return WhisperModel(WHISPER_MODEL, **kwargs)


def resample_to_16k(audio_np: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return audio_np
    target_len = int(len(audio_np) * TARGET_SR / src_sr)
    if target_len <= 0:
        return np.zeros(1, dtype=np.float32)
    from scipy.signal import resample  # lazy import to avoid overhead before needed

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


class LLMClient:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY for remote voice agent server.")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
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


def elevenlabs_tts_pcm(el: ElevenLabs, text: str) -> bytes:
    audio_gen = el.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        optimize_streaming_latency="2",
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


@dataclass
class SessionState:
    memory: Memory
    last_lang: Optional[str] = None
    turns: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


whisper_model = load_whisper()
llm_client = LLMClient()
eleven_client = init_elevenlabs()
sessions: Dict[str, SessionState] = {}


def require_auth(x_auth: Optional[str] = Header(None)):
    if SERVER_AUTH_TOKEN and x_auth != SERVER_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid auth token")


def get_session(session_id: Optional[str]) -> tuple[str, SessionState]:
    if not session_id or session_id not in sessions:
        session_id = uuid.uuid4().hex
        sessions[session_id] = SessionState(memory=Memory(llm_client))
    return session_id, sessions[session_id]


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/api/session")
def create_session(_: None = Depends(require_auth)):
    session_id = uuid.uuid4().hex
    sessions[session_id] = SessionState(memory=Memory(llm_client))
    return {"session_id": session_id}


@app.post("/api/process")
async def process_turn(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    _: None = Depends(require_auth),
):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    session_id, state = get_session(session_id)

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
        assistant_text = llm_client.complete(messages)
        state.memory.add_assistant(assistant_text)
        state.memory.maybe_summarize()
        state.last_lang = lang or state.last_lang
        state.turns += 1

    audio_b64 = None
    if eleven_client:
        try:
            pcm_bytes = elevenlabs_tts_pcm(eleven_client, assistant_text)
            audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")
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


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)