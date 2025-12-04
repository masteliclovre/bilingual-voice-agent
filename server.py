"""FastAPI server with Smart RAG - bilingual voice agent with knowledge base.

Run this on any GPU host (Runpod, etc.) and point REMOTE_AGENT_URL from the
voice_agent.py client to this server. The server keeps per-session memory and
uses Smart RAG for instant knowledge retrieval.
"""

import numpy as np
import base64
import io
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel
from openai import OpenAI
from elevenlabs import ElevenLabs

# Import Smart RAG
from smart_rag import SmartRAG

load_dotenv()

# =========================
# Config
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

# Smart RAG Configuration
ENABLE_RAG = os.getenv("ENABLE_RAG", "true").lower() == "true"
KNOWLEDGE_PATH = os.getenv("KNOWLEDGE_PATH", "knowledge.json")
RAG_DIRECT_ANSWER = os.getenv("RAG_DIRECT_ANSWER", "false").lower() == "true"

TARGET_SR = 16000

SERVER_AUTH_TOKEN = os.getenv("REMOTE_SERVER_AUTH_TOKEN", "").strip() or None

app = FastAPI(title="Bilingual Voice Agent with Smart RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# =========================
# Initialize Smart RAG
# =========================

smart_rag = None
if ENABLE_RAG:
    try:
        knowledge_file = Path(KNOWLEDGE_PATH)
        if knowledge_file.exists():
            smart_rag = SmartRAG(knowledge_path=str(knowledge_file))
            print(f"✓ Smart RAG initialized with {len(smart_rag.knowledge_base)} topics")
        else:
            smart_rag = SmartRAG()
            print("✓ Smart RAG initialized with default knowledge base")
            # Save default knowledge base for reference
            smart_rag.save_knowledge("knowledge.json")
    except Exception as e:
        print(f"⚠️ Smart RAG initialization failed: {e}")
        print("Continuing without RAG support...")
        smart_rag = None

# =========================
# Utilities
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


class EnhancedMemory:
    """Memory class with Smart RAG integration."""

    def __init__(self, llm: LLMClient, rag: Optional[SmartRAG] = None):
        self.llm = llm
        self.rag = rag
        self.summary = ""
        self.window = []
        self.user_turns_since_summary = 0
        self.last_rag_match = None

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

    def build_prompt(self, user_text: str, user_lang_hint: Optional[str]):
        """Build prompt with Smart RAG integration (multi-document retrieval)."""

        # Try RAG matching with multi-document retrieval
        rag_context = None

        if self.rag and user_text:
            try:
                # Get top-3 matches instead of single match
                matches = self.rag.match_multiple(user_text, user_lang_hint, top_k=3)

                if matches:
                    # Store best match for tracking
                    self.last_rag_match = matches[0]

                    # Detect language
                    lang = user_lang_hint or self.rag.detect_language(user_text)

                    # If RAG_DIRECT_ANSWER mode, use best match directly
                    if RAG_DIRECT_ANSWER:
                        best_match = matches[0]
                        rag_response = best_match.response_hr if lang == "hr" else best_match.response_en
                        return self._build_system_prompt(user_lang_hint, rag_context=rag_response)

                    # Build structured context from multiple matches
                    rag_context = self._build_structured_context(matches, lang)
            except Exception as e:
                print(f"RAG matching error: {e}")

        # Build standard prompt (with or without RAG context)
        return self._build_system_prompt(user_lang_hint, rag_context=rag_context)

    def _build_structured_context(self, matches: List, lang: str) -> str:
        """Build structured context from multiple RAG matches.

        Args:
            matches: List of MatchResult objects from match_multiple()
            lang: Language code ('hr' or 'en')

        Returns:
            Formatted context string with metadata
        """
        if not matches:
            return None

        if lang == "hr":
            context = "KONTEKST IZ BAZE ZNANJA:\n\n"
            for i, match in enumerate(matches, 1):
                context += f"[{i}] Tema: {match.topic} (pouzdanost: {match.confidence:.2f})\n"

                # Add source if available
                metadata = match.metadata or {}
                source = metadata.get('source', 'N/A')
                context += f"    Izvor: {source}\n"

                # Add response content
                context += f"    Sadržaj: {match.response_hr}\n\n"

            context += "UPUTA: Koristi navedene izvore za odgovor. Ako korisnik pita o specifičnim detaljima (npr. 'Subota'), izvuci točan podatak iz konteksta."
        else:
            context = "CONTEXT FROM KNOWLEDGE BASE:\n\n"
            for i, match in enumerate(matches, 1):
                context += f"[{i}] Topic: {match.topic} (confidence: {match.confidence:.2f})\n"

                # Add source if available
                metadata = match.metadata or {}
                source = metadata.get('source', 'N/A')
                context += f"    Source: {source}\n"

                # Add response content
                context += f"    Content: {match.response_en}\n\n"

            context += "INSTRUCTION: Use the provided sources for your answer. If the user asks about specific details (e.g., 'Saturday'), extract the exact information from the context."

        return context

    def _build_system_prompt(self, user_lang_hint: Optional[str], rag_context: Optional[str] = None):
        """Build system prompt with optional RAG context."""

        # Base system message
        if (user_lang_hint or "").startswith("hr"):
            system = (
                "Ti si ljubazan i stručan virtualni asistent. "
                "UVIJEK odgovori na istom jeziku na kojem korisnik pita. "
                "Daj kratke, jasne odgovore prikladne za glas (2-5 rečenica). "
            )
        else:
            system = (
                "You are a friendly and professional virtual assistant. "
                "ALWAYS reply in the same language the user speaks. "
                "Keep answers short and clear for voice (2-5 sentences). "
            )

        # Add RAG context if available
        if rag_context:
            if (user_lang_hint or "").startswith("hr"):
                system += f"\n\nKONTEKST IZ BAZE ZNANJA:\n{rag_context}\n\nKoristi ovaj kontekst za odgovor, ali možeš dodati dodatne informacije ako je potrebno."
            else:
                system += f"\n\nCONTEXT FROM KNOWLEDGE BASE:\n{rag_context}\n\nUse this context for your answer, but you can add extra information if needed."

        msgs = [{"role": "system", "content": system}]

        # Add conversation summary if exists
        if self.summary:
            if (user_lang_hint or "").startswith("hr"):
                msgs.append({"role": "system", "content": f"Sažetak prethodnog razgovora:\n{self.summary}"})
            else:
                msgs.append({"role": "system", "content": f"Previous conversation summary:\n{self.summary}"})

        # Add conversation window
        msgs.extend(self.window)

        return msgs

    def get_direct_rag_answer(self, user_text: str, lang: Optional[str]) -> Optional[str]:
        """Get direct answer from RAG without LLM (if RAG_DIRECT_ANSWER mode)."""
        if not self.rag or not RAG_DIRECT_ANSWER:
            return None

        try:
            match = self.rag.match(user_text, lang)
            if match.matched:
                self.last_rag_match = match
                return match.response_hr if lang == "hr" else match.response_en
        except Exception as e:
            print(f"RAG direct answer error: {e}")

        return None


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


def elevenlabs_tts_stream(el: ElevenLabs, text: str):
    """Stream TTS audio chunks as they're generated."""
    audio_gen = el.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        optimize_streaming_latency="2",
        output_format="pcm_16000",
        text=text,
        model_id="eleven_multilingual_v2",
    )
    for chunk in audio_gen:
        if chunk:
            yield chunk


@dataclass
class SessionState:
    memory: EnhancedMemory
    api_key: str
    last_lang: Optional[str] = None
    turns: int = 0
    rag_hits: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


whisper_model = load_whisper()
eleven_client = init_elevenlabs()
sessions: Dict[str, SessionState] = {}


def require_auth(x_auth: Optional[str] = Header(None)):
    if SERVER_AUTH_TOKEN and x_auth != SERVER_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid auth token")


def get_session(session_id: Optional[str], api_key_override: Optional[str]) -> tuple[str, SessionState]:
    llm_client = get_llm_client(api_key_override)
    if not session_id or session_id not in sessions:
        session_id = uuid.uuid4().hex
        memory = EnhancedMemory(llm_client, smart_rag)
        sessions[session_id] = SessionState(memory=memory, api_key=llm_client.api_key)
    else:
        state = sessions[session_id]
        if state.api_key != llm_client.api_key:
            state.memory.llm = llm_client
            state.api_key = llm_client.api_key
    state = sessions[session_id]
    return session_id, state


@app.get("/healthz")
def healthz():
    rag_status = "enabled" if smart_rag else "disabled"
    rag_topics = len(smart_rag.knowledge_base) if smart_rag else 0
    return {
        "status": "ok",
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "rag": rag_status,
        "rag_topics": rag_topics
    }


@app.get("/api/rag/stats")
def get_rag_stats(_: None = Depends(require_auth)):
    """Get Smart RAG statistics."""
    if not smart_rag:
        raise HTTPException(status_code=503, detail="RAG not enabled")

    return smart_rag.get_stats()


@app.get("/api/rag/topics")
def list_rag_topics(_: None = Depends(require_auth)):
    """List all available RAG topics."""
    if not smart_rag:
        raise HTTPException(status_code=503, detail="RAG not enabled")

    topics = []
    for topic_id, data in smart_rag.knowledge_base.items():
        topics.append({
            "id": topic_id,
            "keywords": data.get("keywords", []),
            "priority": data.get("priority", 5),
            "has_hr": "hr" in data.get("responses", {}),
            "has_en": "en" in data.get("responses", {})
        })

    return {"topics": topics}


@app.post("/api/session")
def create_session(
    _: None = Depends(require_auth),
    x_api_key: Optional[str] = Header(None),
):
    """Create a new session."""
    try:
        session_id, _ = get_session(None, x_api_key)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "session_id": session_id,
        "rag_enabled": bool(smart_rag)
    }


@app.post("/api/process")
async def process_turn(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    _: None = Depends(require_auth),
    x_api_key: Optional[str] = Header(None),
    api_key: Optional[str] = Form(None),
):
    """Process audio turn with Smart RAG support."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    api_key_override = api_key or x_api_key
    try:
        session_id, state = get_session(session_id, api_key_override)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    wav_buf = io.BytesIO(audio_bytes)
    user_text, lang = whisper_transcribe(whisper_model, wav_buf)

    if not user_text:
        return {
            "session_id": session_id,
            "text": "",
            "lang": lang,
            "assistant_text": "",
            "tts_audio_b64": None,
            "tts_sample_rate": TARGET_SR,
            "rag_used": False,
        }

    with state.lock:
        state.memory.add_user(user_text)

        # Check if RAG can provide direct answer
        direct_answer = state.memory.get_direct_rag_answer(user_text, lang)

        if direct_answer:
            # Use direct RAG answer without LLM
            assistant_text = direct_answer
            state.rag_hits += 1
            rag_used = True
        else:
            # Build prompt with RAG context and use LLM
            messages = state.memory.build_prompt(user_text, lang)
            assistant_text = state.memory.llm.complete(messages)
            rag_used = bool(state.memory.last_rag_match)
            if rag_used:
                state.rag_hits += 1

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
        "rag_used": rag_used,
        "rag_topic": state.memory.last_rag_match.topic if state.memory.last_rag_match else None,
        "rag_confidence": state.memory.last_rag_match.confidence if state.memory.last_rag_match else 0.0,
    }


@app.post("/api/process_stream")
async def process_turn_stream(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    _: None = Depends(require_auth),
    x_api_key: Optional[str] = Header(None),
    api_key: Optional[str] = Form(None),
):
    """Process audio turn with streaming TTS response."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    api_key_override = api_key or x_api_key
    try:
        session_id, state = get_session(session_id, api_key_override)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    wav_buf = io.BytesIO(audio_bytes)
    user_text, lang = whisper_transcribe(whisper_model, wav_buf)

    if not user_text:
        # Return empty response for silence
        async def empty_stream():
            yield b""
        return StreamingResponse(
            empty_stream(),
            media_type="application/octet-stream",
            headers={
                "X-Session-ID": session_id,
                "X-User-Text": "",
                "X-Lang": lang or "",
                "X-Assistant-Text": "",
                "X-RAG-Used": "false",
            }
        )

    with state.lock:
        state.memory.add_user(user_text)

        # Check if RAG can provide direct answer
        direct_answer = state.memory.get_direct_rag_answer(user_text, lang)

        if direct_answer:
            # Use direct RAG answer without LLM
            assistant_text = direct_answer
            state.rag_hits += 1
            rag_used = True
        else:
            # Build prompt with RAG context and use LLM
            messages = state.memory.build_prompt(user_text, lang)
            assistant_text = state.memory.llm.complete(messages)
            rag_used = bool(state.memory.last_rag_match)
            if rag_used:
                state.rag_hits += 1

        state.memory.add_assistant(assistant_text)
        state.memory.maybe_summarize()
        state.last_lang = lang or state.last_lang
        state.turns += 1

    # Stream TTS audio chunks
    if not eleven_client:
        # No TTS available, return empty stream
        async def empty_stream():
            yield b""
        return StreamingResponse(
            empty_stream(),
            media_type="application/octet-stream",
            headers={
                "X-Session-ID": session_id,
                "X-User-Text": user_text,
                "X-Lang": lang or state.last_lang or "",
                "X-Assistant-Text": assistant_text,
                "X-RAG-Used": str(rag_used).lower(),
                "X-RAG-Topic": state.memory.last_rag_match.topic if state.memory.last_rag_match else "",
                "X-RAG-Confidence": str(state.memory.last_rag_match.confidence) if state.memory.last_rag_match else "0.0",
            }
        )

    def audio_stream():
        """Generator that yields TTS audio chunks."""
        try:
            for chunk in elevenlabs_tts_stream(eleven_client, assistant_text):
                yield chunk
        except Exception as exc:
            print(f"TTS streaming error: {exc}")
            yield b""

    return StreamingResponse(
        audio_stream(),
        media_type="application/octet-stream",
        headers={
            "X-Session-ID": session_id,
            "X-User-Text": user_text,
            "X-Lang": lang or state.last_lang or "",
            "X-Assistant-Text": assistant_text,
            "X-RAG-Used": str(rag_used).lower(),
            "X-RAG-Topic": state.memory.last_rag_match.topic if state.memory.last_rag_match else "",
            "X-RAG-Confidence": str(state.memory.last_rag_match.confidence) if state.memory.last_rag_match else "0.0",
        }
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    print(f"\n{'='*60}")
    print(f"🤖 Bilingual Voice Agent Server with Smart RAG")
    print(f"{'='*60}")
    print(f"├─ LLM provider: {LLM_PROVIDER}")
    print(f"├─ Model: {LLM_MODEL}")
    print(f"├─ RAG: {'Enabled ✓' if smart_rag else 'Disabled ✗'}")

    if smart_rag:
        stats = smart_rag.get_stats()
        print(f"├─ Knowledge topics: {stats['total_topics']}")
        print(f"└─ Topics: {', '.join(stats['topics'][:5])}...")
    else:
        print(f"└─ Port: {port}")

    print(f"{'='*60}\n")

    uvicorn.run(app, host="0.0.0.0", port=port)
