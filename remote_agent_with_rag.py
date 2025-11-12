"""
Enhanced FastAPI server with RAG integration for bilingual banking voice agent.
This extends the original remote_agent.py with vector database support.
"""

import numpy as np
import base64
import io
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from openai import OpenAI
from elevenlabs import ElevenLabs

# Import RAG module
from rag_banking_module import RAGBankingAssistant, BankingDocument

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

# RAG Configuration
ENABLE_RAG = os.getenv("ENABLE_RAG", "true").lower() == "true"
RAG_CONFIDENCE_THRESHOLD = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.7"))

TARGET_SR = 16000

SERVER_AUTH_TOKEN = os.getenv("REMOTE_SERVER_AUTH_TOKEN", "").strip() or None

app = FastAPI(title="Bilingual Banking Voice Agent with RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# =========================
# Initialize RAG Assistant
# =========================

rag_assistant = None
if ENABLE_RAG:
    try:
        rag_assistant = RAGBankingAssistant()
        print("✓ RAG Banking Assistant initialized")
    except Exception as e:
        print(f"⚠️ RAG initialization failed: {e}")
        print("Continuing without RAG support...")

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
    """Memory class with RAG integration for banking context."""
    
    def __init__(self, llm: LLMClient, rag: Optional[RAGBankingAssistant] = None):
        self.llm = llm
        self.rag = rag
        self.summary = ""
        self.window = []
        self.user_turns_since_summary = 0
        self.last_rag_sources = []
        self.conversation_context = {
            "topics_discussed": [],
            "user_intents": [],
            "pending_tasks": []
        }

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
            "You are a memory compressor for a banking assistant. Summarize the following conversation "
            "into concise bullet points capturing user's banking needs, account preferences, "
            "completed actions, and pending requests. Keep neutral tone. Max ~150 words."
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

    def build_prompt_with_rag(self, user_message: str, user_lang_hint: Optional[str]):
        """Build prompt with RAG-augmented context."""
        
        # Detect if this is a banking-related query
        is_banking_query = self._is_banking_related(user_message)
        
        # Get RAG context if available and relevant
        rag_context = ""
        if self.rag and is_banking_query:
            try:
                context, sources = self.rag.generate_context(user_message, user_lang_hint)
                if context:
                    rag_context = context
                    self.last_rag_sources = sources
            except Exception as e:
                print(f"RAG error: {e}")
        
        # Build system prompt
        if user_lang_hint and user_lang_hint.startswith("hr"):
            system = (
                "Ti si stručni bankarski savjetnik koji govori hrvatski i engleski. "
                "UVIJEK odgovori na istom jeziku na kojem korisnik pita. "
                "Daj kratke, jasne odgovore prikladne za glas (2-5 rečenica). "
                "Koristi formalan ali prijateljski ton. "
            )
            if rag_context:
                system += "\n\nAktualne bankarske informacije:\n" + rag_context
        else:
            system = (
                "You are a professional banking advisor fluent in Croatian and English. "
                "ALWAYS reply in the same language the user speaks. "
                "Keep answers short and clear for voice (2-5 sentences). "
                "Use formal but friendly tone. "
            )
            if rag_context:
                system += "\n\nCurrent banking information:\n" + rag_context
        
        msgs = [{"role": "system", "content": system}]
        
        # Add conversation summary if exists
        if self.summary:
            if user_lang_hint and user_lang_hint.startswith("hr"):
                msgs.append({"role": "system", "content": f"Sažetak prethodnog razgovora:\n{self.summary}"})
            else:
                msgs.append({"role": "system", "content": f"Previous conversation summary:\n{self.summary}"})
        
        # Add conversation window
        msgs.extend(self.window)
        
        return msgs

    def _is_banking_related(self, message: str) -> bool:
        """Check if message is banking-related."""
        banking_keywords = [
            # Croatian
            "račun", "kredit", "kartica", "plaćanje", "novac", "štednja",
            "kamata", "transfer", "uplata", "isplata", "saldo", "stanje",
            "bankomat", "banka", "naknada", "troškovi",
            # English
            "account", "credit", "card", "payment", "money", "savings",
            "interest", "transfer", "deposit", "withdrawal", "balance",
            "atm", "bank", "fee", "charge", "loan", "mortgage"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in banking_keywords)

    def build_prompt(self, user_lang_hint: Optional[str]):
        """Legacy method - redirects to RAG-enabled version."""
        # Get the last user message for RAG context
        last_user_msg = ""
        for msg in reversed(self.window):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break
        
        return self.build_prompt_with_rag(last_user_msg, user_lang_hint)


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
    memory: EnhancedMemory
    api_key: str
    last_lang: Optional[str] = None
    turns: int = 0
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
        # Create memory with RAG support
        memory = EnhancedMemory(llm_client, rag_assistant)
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
    rag_status = "enabled" if rag_assistant else "disabled"
    return {
        "status": "ok",
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "rag": rag_status
    }


@app.get("/api/rag/stats")
def get_rag_stats(_: None = Depends(require_auth)):
    """Get RAG knowledge base statistics."""
    if not rag_assistant:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    return rag_assistant.get_statistics()


@app.post("/api/rag/search")
def search_knowledge_base(
    query: str = Form(...),
    lang: Optional[str] = Form(None),
    _: None = Depends(require_auth)
):
    """Search the knowledge base directly."""
    if not rag_assistant:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    results = rag_assistant.search(query)
    return {
        "query": query,
        "detected_language": rag_assistant.detect_language(query) if not lang else lang,
        "results": results
    }


@app.post("/api/rag/add_document")
async def add_document(
    content_hr: str = Form(...),
    content_en: str = Form(...),
    category: str = Form(...),
    keywords: str = Form(...),
    _: None = Depends(require_auth)
):
    """Add new document to knowledge base."""
    if not rag_assistant:
        raise HTTPException(status_code=503, detail="RAG not available")
    
    doc = BankingDocument(
        id=f"custom_{uuid.uuid4().hex[:8]}",
        content_hr=content_hr,
        content_en=content_en,
        category=category,
        keywords=keywords.split(","),
        metadata={"source": "api", "timestamp": datetime.now().isoformat()}
    )
    
    rag_assistant.add_document(doc)
    return {"status": "success", "document_id": doc.id}


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
    return {"session_id": session_id, "rag_enabled": bool(rag_assistant)}


@app.post("/api/process")
async def process_turn(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    _: None = Depends(require_auth),
    x_api_key: Optional[str] = Header(None),
    api_key: Optional[str] = Form(None),
):
    """Process audio turn with RAG support."""
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
        
        # Build prompt with RAG if available
        messages = state.memory.build_prompt_with_rag(user_text, lang)
        
        # Generate response
        assistant_text = state.memory.llm.complete(messages)
        state.memory.add_assistant(assistant_text)
        state.memory.maybe_summarize()
        state.last_lang = lang or state.last_lang
        state.turns += 1
        
        # Check if RAG was used
        rag_used = bool(state.memory.last_rag_sources)

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
        "rag_sources": len(state.memory.last_rag_sources) if rag_used else 0
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    print(f"🏦 Starting Banking Voice Agent Server")
    print(f"├─ LLM provider: {LLM_PROVIDER}")
    print(f"├─ Model: {LLM_MODEL}")
    print(f"└─ RAG: {'Enabled ✓' if rag_assistant else 'Disabled ✗'}")
    
    if rag_assistant:
        stats = rag_assistant.get_statistics()
        print(f"\n📚 Knowledge Base:")
        print(f"├─ Documents: {stats['total_documents']}")
        print(f"└─ Categories: {list(stats['categories'].keys())}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
