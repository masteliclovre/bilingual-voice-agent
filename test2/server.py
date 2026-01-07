"""FastAPI server that runs the bilingual voice agent pipeline on a GPU backend.

Run this on any GPU host and point REMOTE_AGENT_URL from the
voice_agent.py client to this server. The server keeps per-session memory so the
conversation stays coherent across turns. Audio is exchanged as 16 kHz mono WAV
(PCM16) payloads to keep latency low.
"""

import numpy as np
import base64
import io
import os
import threading
import uuid
import asyncio
import concurrent.futures
from dataclasses import dataclass, field
from typing import Dict, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from openai import OpenAI
from elevenlabs import ElevenLabs
import json
import wave

load_dotenv()

# =========================
# Config (mirrors voice_agent.py)
# =========================

# LLM Provider configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# API Keys
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "vFQACl5nAIV0owAavYxE")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "GoranS/whisper-base-1m.hr-ctranslate2")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")

# LLM Model configuration
if LLM_PROVIDER == "groq":
    LLM_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
else:
    LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

LLM_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "150"))

MAX_TURNS_IN_WINDOW = int(os.getenv("MAX_TURNS_IN_WINDOW", "8"))
SUMMARY_UPDATE_EVERY = int(os.getenv("SUMMARY_UPDATE_EVERY", "8"))

# Thread pool configuration
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))

TARGET_SR = 16000

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
# Thread Pool Executor
# =========================

# Thread pool for CPU-bound operations (LLM calls, TTS)
cpu_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_REQUESTS,
    thread_name_prefix="cpu_worker"
)

print(f"🔧 Initialized thread pool with {MAX_CONCURRENT_REQUESTS} workers")

# =========================
# Utilities shared with voice_agent
# =========================


def load_whisper():
    kwargs = dict(
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
        cpu_threads=os.cpu_count(),
        num_workers=1,
    )
    model = WhisperModel(WHISPER_MODEL, **kwargs)
    
    # GPU warmup
    if WHISPER_DEVICE == "cuda":
        print("🔥 Warming up GPU...")
        dummy = np.zeros(16000, dtype=np.float32)
        try:
            list(model.transcribe(dummy, beam_size=1, vad_filter=False))
            print("✅ GPU warmup complete")
        except Exception as e:
            print(f"⚠️  Warmup warning: {e}")
    
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
    def __init__(self, api_key: str, provider: str = LLM_PROVIDER):
        api_key = api_key.strip()
        if not api_key:
            raise RuntimeError(f"Missing API key for {provider} LLM provider.")
        
        self.api_key = api_key
        self.provider = provider
        self.model = LLM_MODEL
        
        # Configure client based on provider
        if provider == "groq":
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
        else:  # openai
            self.client = OpenAI(api_key=api_key)

    def complete(self, messages, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS):
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
    """Get or create LLM client based on provider."""
    if LLM_PROVIDER == "groq":
        key = (api_key_override or DEFAULT_GROQ_API_KEY).strip()
    else:
        key = (api_key_override or DEFAULT_OPENAI_API_KEY).strip()
    
    if not key:
        raise RuntimeError(f"Missing API key for {LLM_PROVIDER} LLM provider.")
    
    cache_key = f"{LLM_PROVIDER}:{key}"
    if cache_key not in llm_clients:
        llm_clients[cache_key] = LLMClient(key, LLM_PROVIDER)
    return llm_clients[cache_key]


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
    api_key: str
    last_lang: Optional[str] = None
    turns: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


# =========================
# Initialize models at startup
# =========================

print("\n" + "="*60)
print("🚀 Initializing Bilingual Voice Agent Server")
print("="*60)

whisper_model = load_whisper()
eleven_client = init_elevenlabs()
sessions: Dict[str, SessionState] = {}

print("="*60)
print("✅ Server initialization complete")
print("="*60 + "\n")


# =========================
# API Endpoints
# =========================


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
    return {
        "status": "ok",
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "max_concurrent": MAX_CONCURRENT_REQUESTS
    }


@app.post("/api/session")
def create_session(
    _: None = Depends(require_auth),
    x_api_key: Optional[str] = Header(None),
):
    """Create a new session. Accepts API key via x_api_key header."""
    try:
        session_id, _ = get_session(None, x_api_key)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"session_id": session_id}


@app.post("/api/process")
async def process_turn(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    _: None = Depends(require_auth),
    x_api_key: Optional[str] = Header(None),
    api_key: Optional[str] = Form(None),
):
    """Process audio turn. Accepts API key via x_api_key header or api_key form field."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    api_key_override = api_key or x_api_key
    try:
        session_id, state = get_session(session_id, api_key_override)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    wav_buf = io.BytesIO(audio_bytes)
    
    # GPU-bound operation (Whisper) - run synchronously
    user_text, lang = whisper_transcribe(whisper_model, wav_buf)

    if not user_text:
        return {
            "session_id": session_id,
            "text": "",
            "lang": lang,
            "assistant_text": "",
            "tts_audio_b64": None,
            "tts_sample_rate": TARGET_SR,
        }

    # CPU-bound operation (LLM) - run in thread pool
    loop = asyncio.get_event_loop()
    
    def process_llm():
        """CPU-intensive LLM processing in thread pool"""
        with state.lock:
            state.memory.add_user(user_text)
            messages = state.memory.build_prompt(lang)
            assistant_text = state.memory.llm.complete(messages)
            state.memory.add_assistant(assistant_text)
            state.memory.maybe_summarize()
            state.last_lang = lang or state.last_lang
            state.turns += 1
            return assistant_text
    
    # Run LLM in thread pool
    assistant_text = await loop.run_in_executor(cpu_executor, process_llm)

    # CPU-bound operation (TTS) - run in thread pool if available
    audio_b64 = None
    if eleven_client:
        def generate_tts():
            """CPU-intensive TTS processing in thread pool"""
            try:
                pcm_bytes = elevenlabs_tts_pcm(eleven_client, assistant_text)
                return base64.b64encode(pcm_bytes).decode("ascii")
            except Exception as exc:
                print(f"TTS error: {exc}")
                return None
        
        # Run TTS in thread pool
        audio_b64 = await loop.run_in_executor(cpu_executor, generate_tts)

    return {
        "session_id": session_id,
        "text": user_text,
        "lang": lang or state.last_lang,
        "assistant_text": assistant_text,
        "tts_audio_b64": audio_b64,
        "tts_sample_rate": TARGET_SR,
    }


@app.websocket("/api/custom-transcriber")
async def custom_transcriber_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for VAPI custom transcriber integration.
    VAPI sends audio chunks, we transcribe with Whisper and return text.
    """
    await websocket.accept()
    print("🔌 VAPI WebSocket connected")

    audio_buffer = bytearray()
    sample_rate = 16000  # VAPI uses 16kHz

    try:
        while True:
            data = await websocket.receive()

            # Handle text messages (control messages from VAPI)
            if "text" in data:
                msg = json.loads(data["text"])
                msg_type = msg.get("type")

                if msg_type == "config":
                    # VAPI sends config at the start
                    print(f"📋 VAPI config: {msg}")
                    await websocket.send_text(json.dumps({"type": "config_ack"}))

                elif msg_type == "stop":
                    print("🛑 VAPI requested stop")
                    break

            # Handle binary audio data
            elif "bytes" in data:
                audio_chunk = data["bytes"]
                audio_buffer.extend(audio_chunk)

                # Process when we have enough audio (e.g., 1 second = 16000 samples * 2 bytes)
                min_buffer_size = sample_rate * 2  # 1 second of 16-bit PCM

                if len(audio_buffer) >= min_buffer_size:
                    # Convert buffer to numpy array
                    audio_np = np.frombuffer(bytes(audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0

                    # Transcribe with Whisper
                    try:
                        segments, info = whisper_model.transcribe(
                            audio=audio_np,
                            beam_size=1,
                            vad_filter=True,
                            temperature=0.0,
                            language=None,
                            condition_on_previous_text=False,
                            word_timestamps=False,
                            without_timestamps=True,
                        )

                        text = "".join(seg.text for seg in segments).strip()
                        lang = getattr(info, "language", "en")

                        if text:
                            # Send transcription back to VAPI
                            response = {
                                "type": "transcript",
                                "text": text,
                                "language": lang,
                                "isFinal": True
                            }
                            await websocket.send_text(json.dumps(response))
                            print(f"🎤 Transcribed ({lang}): {text}")

                    except Exception as e:
                        print(f"❌ Transcription error: {e}")

                    # Clear buffer after processing
                    audio_buffer.clear()

    except WebSocketDisconnect:
        print("🔌 VAPI WebSocket disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


@app.on_event("shutdown")
def shutdown_event():
    """Cleanup thread pool on shutdown"""
    print("🛑 Shutting down thread pool...")
    cpu_executor.shutdown(wait=True)
    print("✅ Thread pool shutdown complete")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    print(f"Starting server with LLM provider: {LLM_PROVIDER}")
    print(f"Using model: {LLM_MODEL}")
    print(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    uvicorn.run(app, host="0.0.0.0", port=port)