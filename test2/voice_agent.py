# voice_agent_continuous.py
# Continuous bilingual (HR/EN) voice agent — low-latency edition
# - Always-listening RMS VAD turn-taking (no click)
# - Local transcription (faster-whisper) — in-memory (no temp WAV)
# - OpenAI or Groq streaming for reasoning (speak sentence-by-sentence)
# - ElevenLabs TTS streamed to speakers (pcm_16000)
# - Offline TTS fallback (pyttsx3) when ElevenLabs is unavailable
# - Conversation memory: rolling history + auto summary compression (background)
# Version: 2025-09-16

import os
import io
import sys
import re
import time
import wave
import threading
import queue
import base64
from dataclasses import dataclass
from typing import Optional
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openai import OpenAI
from elevenlabs import ElevenLabs
from scipy.signal import resample
import requests  # add this

try:
    from groq import Groq  # Optional ultra-low-latency LLM backend
    HAS_GROQ = True
except Exception:  # pragma: no cover - optional dependency
    Groq = None
    HAS_GROQ = False

ASR_REMOTE_URL = os.getenv("ASR_REMOTE_URL", "").strip() or None
REMOTE_AGENT_URL = os.getenv("REMOTE_AGENT_URL", "").strip() or None
REMOTE_AGENT_TOKEN = os.getenv("REMOTE_AGENT_TOKEN", "").strip() or None

# Optional offline TTS
try:
    import pyttsx3  # offline fallback
    HAS_PYTTXS3 = True
except Exception:
    HAS_PYTTXS3 = False

# =========================
# Config
# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "vFQACl5nAIV0owAavYxE")
# Ako model nije lokalno, faster-whisper će ga povući s HF:
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "GoranS/whisper-large-v3-turbo-hr-parla-ctranslate2")

PREFERRED_INPUT_NAME = os.getenv("PREFERRED_INPUT_NAME", "").strip() or None
INPUT_DEVICE_INDEX = os.getenv("INPUT_DEVICE_INDEX", "").strip()
INPUT_DEVICE_INDEX = int(INPUT_DEVICE_INDEX) if INPUT_DEVICE_INDEX.isdigit() else None

# Whisper perf tuning (GPU if available)
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")          # "cuda" | "cpu" | "auto"
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16")        # "float16" | "int8_float16" | "int8" | "auto"

# Audio
TARGET_SR = 16000
CHANNELS = 1
FRAME_DURATION_MS = int(os.getenv("FRAME_DURATION_MS", "20"))   # tighter frames for faster VAD
MAX_UTTERANCE_SECS = 45
SILENCE_TIMEOUT_SECS = float(os.getenv("SILENCE_TIMEOUT_SECS", "1.2"))
MIN_SPEECH_SECS = float(os.getenv("MIN_SPEECH_SECS", "0.3"))
RMS_THRESH = float(os.getenv("RMS_THRESH", "0.003"))  # lower (e.g. 0.002) if your mic is quiet
RMS_HANGOVER = float(os.getenv("RMS_HANGOVER", "0.18"))

# OpenAI
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
if LLM_PROVIDER not in {"openai", "groq"}:
    LLM_PROVIDER = "openai"
if LLM_PROVIDER == "openai" and not OPENAI_API_KEY and GROQ_API_KEY:
    # Auto-fallback when OpenAI key missing but Groq key present
    LLM_PROVIDER = "groq"
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
if not LLM_MODEL:
    LLM_MODEL = GROQ_MODEL if LLM_PROVIDER == "groq" else OPENAI_MODEL
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "180"))  # keep replies short for voice

# Memory
MAX_TURNS_IN_WINDOW = int(os.getenv("MAX_TURNS_IN_WINDOW", "12"))
SUMMARY_UPDATE_EVERY = int(os.getenv("SUMMARY_UPDATE_EVERY", "4"))

# Wake word (optional)
WAKE_WORD = os.getenv("WAKE_WORD", "").strip() or None

# Offline TTS preferences (optional)
OFFLINE_TTS_RATE = int(os.getenv("OFFLINE_TTS_RATE", "180"))  # words per min approx
OFFLINE_TTS_VOICE_HINT_HR = os.getenv("OFFLINE_TTS_VOICE_HINT_HR", "hr;croat;hrv;hr-HR;Hrvatski")
OFFLINE_TTS_VOICE_HINT_EN = os.getenv("OFFLINE_TTS_VOICE_HINT_EN", "en;eng;en-US;English")

# ElevenLabs latency tuning
ELEVEN_STREAM_LATENCY = os.getenv("ELEVEN_STREAM_LATENCY", "2")  # "0".."4" string. 2 is a good balance.

# Streaming chunker tuning
STREAMING_MIN_CHARS = int(os.getenv("STREAMING_MIN_CHARS", "48"))
STREAMING_MAX_WAIT = float(os.getenv("STREAMING_MAX_WAIT", "1.4"))

# =========================
# Utilities
# =========================

def list_audio_devices():
    print("Audio devices:")
    for i, d in enumerate(sd.query_devices()):
        name = d.get('name', '?')
        in_ch = d.get('max_input_channels', 0)
        out_ch = d.get('max_output_channels', 0)
        sr = d.get('default_samplerate', None)
        print(f"[{i:02d}] {name}  in:{in_ch} out:{out_ch} sr:{sr}")

def pick_input_device(prefer_name_substr=None, prefer_index=None):
    devices = sd.query_devices()
    # prefer index
    if isinstance(prefer_index, int) and 0 <= prefer_index < len(devices):
        if devices[prefer_index]['max_input_channels'] > 0:
            return prefer_index
    # prefer name
    if prefer_name_substr:
        p = prefer_name_substr.lower()
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0 and p in d.get('name', '').lower():
                return i
    # first input-capable
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            return i
    raise RuntimeError("No input audio devices with capture channels.")

def resample_to_16k(audio_np: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return audio_np
    target_len = int(len(audio_np) * TARGET_SR / src_sr)
    if target_len <= 0:
        return np.zeros(1, dtype=np.float32)
    return resample(audio_np, target_len).astype(np.float32)

def float32_to_wav_bytes(audio_np: np.ndarray, sr: int) -> io.BytesIO:
    audio_16k = resample_to_16k(audio_np, sr)
    int16 = np.clip(audio_16k * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(int16.tobytes())
    buf.seek(0)
    return buf


# =========================
# Engines (Whisper, OpenAI, ElevenLabs)
# =========================

def load_whisper():
    print("Loading Whisper model (first run may take a bit)…")
    # Add CPU/GPU perf knobs
    kwargs = dict(
        device=WHISPER_DEVICE,          # "cuda" for GPU if available
        compute_type=WHISPER_COMPUTE,   # "float16" on GPU, "int8_float16" on CPU
        cpu_threads=os.cpu_count(),
        num_workers=2,
    )
    return WhisperModel(WHISPER_MODEL, **kwargs)

def init_openai():
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    return OpenAI(api_key=OPENAI_API_KEY)

class LLMClient:
    """Thin wrapper so we can swap OpenAI with Groq without touching the main loop."""

    def __init__(self):
        provider = LLM_PROVIDER or "openai"
        self.provider = provider
        self.model = LLM_MODEL

        if provider == "groq":
            if not GROQ_API_KEY:
                raise RuntimeError("LLM_PROVIDER=groq but GROQ_API_KEY is missing.")
            if not HAS_GROQ:
                raise RuntimeError(
                    "LLM_PROVIDER=groq requested but groq package is not installed. Run: pip install groq"
                )
            self.client = Groq(api_key=GROQ_API_KEY)
        else:
            self.client = init_openai()
            self.provider = "openai"  # fallback to canonical value

    def _common_params(self, messages, temperature, max_tokens, stream):
        params = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        if max_tokens:
            params["max_tokens"] = max_tokens
        if stream:
            params["stream"] = True
        return params

    def stream_text(self, messages, temperature=OPENAI_TEMPERATURE, max_tokens=OPENAI_MAX_TOKENS):
        params = self._common_params(messages, temperature, max_tokens, stream=True)
        response = self.client.chat.completions.create(**params)
        for chunk in response:
            try:
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", None)
            except Exception:
                token = None
            if token:
                yield token

    def complete(self, messages, temperature=OPENAI_TEMPERATURE, max_tokens=OPENAI_MAX_TOKENS):
        params = self._common_params(messages, temperature, max_tokens, stream=False)
        resp = self.client.chat.completions.create(**params)
        return resp.choices[0].message.content.strip()


def init_elevenlabs():
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("Missing ELEVENLABS_API_KEY in .env")
    return ElevenLabs(api_key=ELEVENLABS_API_KEY)


# =========================
# ASR (in-memory) — faster-whisper
# =========================

def whisper_transcribe(whisper: WhisperModel, wav_buf: io.BytesIO):
    """
    Read 16k mono WAV from memory, feed directly to faster-whisper (no disk I/O).
    """
    if wav_buf.getbuffer().nbytes < 32000:
        return "", None

    wav_buf.seek(0)
    with wave.open(wav_buf, 'rb') as wf:
        sr = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != TARGET_SR:
        audio = resample_to_16k(audio, sr)

    segments, info = whisper.transcribe(
        audio=audio,
        beam_size=1,                       # greedy for speed
        vad_filter=True,                   # trim silence
        temperature=0.0,
        language=None,                     # auto-detect
        condition_on_previous_text=False,
        word_timestamps=False,
        without_timestamps=True,
    )
    text = "".join(seg.text for seg in segments).strip()
    lang = getattr(info, "language", None)
    return text, lang


# =========================
# LLM streaming utilities
# =========================

_SENT_END_RE = re.compile(r"[\.!\?…]\s+$")


def stream_text_segments(llm: "LLMClient", messages, temperature=OPENAI_TEMPERATURE, max_tokens=OPENAI_MAX_TOKENS):
    """Stream tokens from the LLM and release speech-sized segments ASAP."""

    buf: list[str] = []
    last_flush = time.time()
    for token in llm.stream_text(messages, temperature=temperature, max_tokens=max_tokens):
        if not token:
            continue
        buf.append(token)
        joined = "".join(buf)
        now = time.time()
        should_flush = False

        if _SENT_END_RE.search(joined):
            should_flush = True
        elif len(joined) >= STREAMING_MIN_CHARS:
            should_flush = True
        elif (now - last_flush) >= STREAMING_MAX_WAIT and joined.strip():
            should_flush = True

        if should_flush:
            yield joined
            buf.clear()
            last_flush = now

    if buf:
        yield "".join(buf)


# =========================
# TTS: ElevenLabs + Offline fallback
# =========================

class OutputAudio:
    """
    Persistent output stream to reduce device open/close overhead.
    """
    def __init__(self, samplerate=TARGET_SR, channels=1):
        self.stream = sd.OutputStream(samplerate=samplerate, channels=channels, dtype='float32')
        self.stream.start()

    def write_int16_bytes(self, pcm_bytes: bytes):
        if not pcm_bytes:
            return
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.stream.write(pcm.reshape(-1, 1))

    def write_float_np(self, pcm: np.ndarray):
        if pcm.size == 0:
            return
        self.stream.write(pcm.reshape(-1, 1))

    def close(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

def tts_elevenlabs_stream_to_output(el: ElevenLabs, text: str, out: OutputAudio):
    """
    ElevenLabs TTS (pcm_16000) streamed directly to the persistent OutputAudio.
    On auth/plan/network failure, raise to let caller decide on fallback.
    """
    audio_gen = el.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        optimize_streaming_latency=str(ELEVEN_STREAM_LATENCY),  # "0".."4"
        output_format="pcm_16000",   # raw 16kHz PCM
        text=text,
        model_id="eleven_multilingual_v2",
    )
    got_audio = False
    for chunk in audio_gen:
        if not chunk:
            continue
        out.write_int16_bytes(chunk)
        got_audio = True
    if not got_audio:
        raise RuntimeError("ElevenLabs returned empty audio")

def _select_offline_voice(engine, lang_hint: Optional[str]):
    """
    Try to pick a voice matching language hints (Croatian vs English).
    Fallback to engine default if no match.
    """
    try:
        voices = engine.getProperty('voices')
    except Exception:
        return None

    hr_hints = [s.strip().lower() for s in OFFLINE_TTS_VOICE_HINT_HR.split(";") if s.strip()]
    en_hints = [s.strip().lower() for s in OFFLINE_TTS_VOICE_HINT_EN.split(";") if s.strip()]
    prefer_hints = hr_hints if (lang_hint or "").startswith("hr") else en_hints

    def match(v, hint_list):
        name = (getattr(v, "name", "") or "").lower()
        lang = ""
        try:
            langs = getattr(v, "languages", []) or []
            if langs:
                lang = ",".join([str(x).lower() for x in langs])
        except Exception:
            pass
        idf = (getattr(v, "id", "") or "").lower()
        blob = " ".join([name, lang, idf])
        return any(h in blob for h in hint_list)

    for v in voices:
        if match(v, prefer_hints):
            return v.id
    if (lang_hint or "").startswith("hr"):
        for v in voices:
            if match(v, en_hints):
                return v.id
    if voices:
        return voices[0].id
    return None

def tts_offline_pyttsx3(text: str, lang_hint: Optional[str]):
    """
    Offline TTS using pyttsx3 (SAPI5 on Windows, NSSpeech on macOS, eSpeak on Linux).
    Note: Actual Croatian voice availability depends on installed system voices.
    """
    if not HAS_PYTTXS3:
        print("Offline TTS fallback requested, but pyttsx3 is not installed. Run: pip install pyttsx3")
        return
    try:
        engine = pyttsx3.init()
        vid = _select_offline_voice(engine, lang_hint)
        if vid:
            engine.setProperty('voice', vid)
        try:
            engine.setProperty('rate', OFFLINE_TTS_RATE)
        except Exception:
            pass
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("Offline TTS error:", e)

def say_sentence_with_fallback(el: Optional[ElevenLabs], out: OutputAudio, text: str, lang_hint: Optional[str]):
    """
    Speak a single sentence. Try ElevenLabs; on failure, use pyttsx3.
    Never raise to the main loop.
    """
    if not text.strip():
        return
    try:
        if el is None:
            tts_offline_pyttsx3(text, lang_hint)
        else:
            tts_elevenlabs_stream_to_output(el, text, out)
    except Exception as e:
        msg = str(e)
        print("TTS error:", msg)
        if ("401" in msg or "403" in msg or
            "detected_unusual_activity" in msg or
            "Invalid API key" in msg or
            "insufficient" in msg or
            "quota" in msg):
            print("⚠️ ElevenLabs rejected the request (key/plan/antabuse). Falling back to offline TTS.")
        else:
            print("⚠️ ElevenLabs unavailable. Falling back to offline TTS.")
        tts_offline_pyttsx3(text, lang_hint)


# =========================
# Conversation Memory
# =========================

class Memory:
    """
    Keeps rolling verbatim turns + an accumulated summary.
    We summarize every N user turns to keep context compact.
    """
    def __init__(self, llm: "LLMClient"):
        self.llm = llm
        self.summary = ""           # long-term compressed memory
        self.window = []            # recent messages: [{'role':'user'|'assistant','content':...}, ...]
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
            self.window = self.window[-MAX_TURNS_IN_WINDOW*2:]

    def maybe_summarize(self):
        if self.user_turns_since_summary < SUMMARY_UPDATE_EVERY:
            return
        sys_prompt = (
            "You are a memory compressor. Summarize the following conversation "
            "into concise bullet points capturing user preferences, facts, goals, and unresolved tasks. "
            "Keep neutral tone. Max ~150 words."
        )
        msgs = [{"role": "system", "content": sys_prompt}]
        if self.summary:
            msgs.append({"role": "system", "content": f"Existing summary memory:\n{self.summary}"})
        for m in self.window:
            msgs.append(m)
        try:
            new_summary = self.llm.complete(msgs, temperature=0.2)
            self.summary = new_summary
            self.user_turns_since_summary = 0
        except Exception as e:
            print("Memory summarize error:", e)

    def build_prompt(self, user_lang_hint: Optional[str]):
        system = (
            "You are a concise, helpful bilingual assistant for Croatian and English. "
            "Detect the user's language (Croatian or English) and ALWAYS reply in that language. "
            "If the user mixes languages, keep their dominant language. "
            "Keep answers short for voice (2–5 sentences). "
            "Be context-aware and remember prior details from the summary."
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


# =========================
# Turn-taking capture (RMS VAD)
# =========================

class ContinuousListener:
    """
    Opens and closes the input stream for each utterance to avoid TTS echo.
    Capture ends on silence timeout and minimum speech checks.
    """

    def __init__(self):
        self.device_index = pick_input_device(PREFERRED_INPUT_NAME, INPUT_DEVICE_INDEX)
        self.dev_info = sd.query_devices(self.device_index)
        # Force capture at TARGET_SR (skips resampling later if supported)
        self.device_sr = TARGET_SR

    def record_utterance(self):
        """Capture one utterance based on RMS VAD with hangover and silence timeout."""
        print("\n🎙️ Listening… (speak; short pause = end)")
        chunks = []
        start_time = time.time()
        last_above = None
        frame_samples = max(256, int(TARGET_SR * FRAME_DURATION_MS / 1000))
        started_speaking = False

        def callback(indata, frames, time_info, status):
            nonlocal last_above, started_speaking
            if status:
                # Don't spam; print once per state
                print(status, file=sys.stderr)
            mono = indata.copy().reshape(-1)
            rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
            chunks.append(mono.tobytes())
            if rms > RMS_THRESH:
                last_above = time.time()
                started_speaking = True

        try:
            with sd.InputStream(device=self.device_index,
                                channels=CHANNELS,
                                samplerate=self.device_sr,
                                dtype='float32',
                                blocksize=frame_samples,
                                callback=callback):
                while True:
                    time.sleep(0.04)
                    now = time.time()
                    if now - start_time > MAX_UTTERANCE_SECS:
                        break
                    if started_speaking:
                        if last_above is not None and (now - last_above) > max(SILENCE_TIMEOUT_SECS, RMS_HANGOVER):
                            break
        except Exception as e:
            print("Stream error:", e)
            return io.BytesIO()

        if not chunks:
            print("⏹️ End (no audio).")
            return io.BytesIO()

        audio_float = b"".join(chunks)
        audio_np = np.frombuffer(audio_float, dtype=np.float32)

        duration_sec = len(audio_np) / float(self.device_sr)
        if duration_sec < MIN_SPEECH_SECS:
            print("⏹️ End (too short).")
            return io.BytesIO()

        print("⏹️ End of utterance.")
        return float32_to_wav_bytes(audio_np, self.device_sr)


# =========================
# Main loop
# =========================

def main():
    list_audio_devices()

    remote_client: Optional[RemoteAgentClient] = None
    llm: Optional[LLMClient] = None
    whisper: Optional[WhisperModel] = None
    mem: Optional[Memory] = None
    el: Optional[ElevenLabs] = None

    asr_url = ASR_REMOTE_URL
    if REMOTE_AGENT_URL:
        if ASR_REMOTE_URL:
            print("REMOTE_AGENT_URL set — ignoring ASR_REMOTE_URL (full pipeline handled remotely).")
        asr_url = None
        print(f"Using remote voice agent backend at: {REMOTE_AGENT_URL}")
        remote_client = RemoteAgentClient(REMOTE_AGENT_URL, REMOTE_AGENT_TOKEN)
    else:
        llm = LLMClient()
        if not asr_url:
            whisper = load_whisper()
        else:
            print(f"Using remote ASR at: {asr_url}")

        try:
            el = init_elevenlabs()
        except Exception as e:
            print("ElevenLabs init warning:", e)
            print("Proceeding with offline TTS only.")

        mem = Memory(llm)

    # Persistent audio output to reduce latency
    out = OutputAudio(samplerate=TARGET_SR, channels=1)

    listener = ContinuousListener()

    print("\nBilingual voice agent ready. (HR/EN)")
    print("Tips:")
    print("- Speak naturally; a short pause ends your turn.")
    print("- Lower RMS_THRESH in .env if it misses quiet speech (e.g. 0.002).")
    if WAKE_WORD:
        print(f"- Wake word enabled: say “{WAKE_WORD}” to start a turn.")

    try:
        while True:
            try:
                # 1) Capture one utterance
                wav_buf = listener.record_utterance()
                if wav_buf.getbuffer().nbytes < 32000:
                    continue

                if remote_client:
                    try:
                        result = remote_client.process(wav_buf)
                    except Exception as e:
                        print("Remote agent error:", e)
                        continue
                    if not result or not result.user_text:
                        continue
                    if result.skipped:
                        if (result.reason == "wake_word_missing") and WAKE_WORD:
                            print(f"(Ignored — missing wake word '{WAKE_WORD}')")
                        continue
                    flag = "🇭🇷" if (result.lang or "").startswith("hr") else "🇬🇧"
                    print(f"{flag} You: {result.user_text}")
                    print("🤖 Assistant: ", end="", flush=True)
                    print(result.assistant_text)
                    if result.audio_pcm16:
                        if result.sample_rate != TARGET_SR:
                            pcm = np.frombuffer(result.audio_pcm16, dtype=np.int16).astype(np.float32) / 32768.0
                            pcm_16k = resample_to_16k(pcm, result.sample_rate)
                            out.write_float_np(pcm_16k)
                        else:
                            out.write_int16_bytes(result.audio_pcm16)
                    else:
                        print("(Remote agent returned no audio — skipping playback)")
                    continue

                # 2) Transcribe
                if asr_url:
                    user_text, lang = remote_transcribe(asr_url, wav_buf)
                else:
                    user_text, lang = whisper_transcribe(whisper, wav_buf)

                if not user_text:
                    continue

                # 2a) Wake word (optional)
                if WAKE_WORD:
                    if user_text.lower().strip().startswith(WAKE_WORD.lower()):
                        user_text = user_text[len(WAKE_WORD):].lstrip(" ,.-:") or "Hej!"
                    else:
                        print(f"(Ignored — missing wake word '{WAKE_WORD}')")
                        continue

                flag = "🇭🇷" if (lang or "").startswith("hr") else "🇬🇧"
                print(f"{flag} You: {user_text}")

                # 3) Build contextful prompt with memory
                mem.add_user(user_text)
                messages = mem.build_prompt(user_lang_hint=lang)

                # 4) LLM reply — stream sentences and speak each sentence immediately
                assistant_text_parts = []
                print("🤖 Assistant: ", end="", flush=True)
                for sent in stream_text_segments(llm, messages):
                    # print and speak as we go
                    print(sent, end="", flush=True)
                    assistant_text_parts.append(sent)
                    say_sentence_with_fallback(el, out, sent, lang)
                print()  # newline
                assistant_text = "".join(assistant_text_parts)

                # 5) Add to memory
                mem.add_assistant(assistant_text)

                # 5b) Summarize in background (non-blocking)
                def _bg_sum():
                    try:
                        mem.maybe_summarize()
                    except Exception as e:
                        print("Memory summarize error:", e)
                threading.Thread(target=_bg_sum, daemon=True).start()

            except KeyboardInterrupt:
                print("\nExit. Bye!")
                break
            except Exception as e:
                print("Error:", e)
                time.sleep(0.2)
    finally:
        out.close()


def remote_transcribe(url: str, wav_buf: io.BytesIO):
    try:
        wav_buf.seek(0)
        files = {"file": ("audio.wav", wav_buf.read(), "audio/wav")}
        r = requests.post(url, files=files, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("text", "") or "", data.get("lang", "") or None
    except Exception as e:
        print("Remote ASR error:", e)
        return "", None


@dataclass
class RemoteAgentResult:
    user_text: str
    assistant_text: str
    lang: Optional[str]
    audio_pcm16: bytes
    sample_rate: int
    session_id: Optional[str] = None
    skipped: bool = False
    reason: Optional[str] = None


class RemoteAgentClient:
    def __init__(self, base_url: str, token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/api"):
            self.base_url = f"{self.base_url}/api"
        self.token = token.strip() if token else None
        self.session_id: Optional[str] = None

    def _headers(self):
        headers = {}
        if self.token:
            headers["X-Auth-Token"] = self.token
        return headers

    def ensure_session(self):
        if self.session_id:
            return
        resp = requests.post(
            f"{self.base_url}/session",
            headers=self._headers(),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        sid = data.get("session_id")
        if not sid:
            raise RuntimeError("Remote agent did not return a session_id")
        self.session_id = sid

    def process(self, wav_buf: io.BytesIO) -> Optional[RemoteAgentResult]:
        payload = wav_buf.getvalue()
        if not payload:
            return None
        attempts = 0
        while attempts < 2:
            attempts += 1
            self.ensure_session()
            files = {"audio": ("audio.wav", payload, "audio/wav")}
            data = {"session_id": self.session_id}
            resp = requests.post(
                f"{self.base_url}/process",
                headers=self._headers(),
                files=files,
                data=data,
                timeout=120,
            )
            if resp.status_code in (401, 403):
                raise RuntimeError("Remote agent authentication failed")
            if resp.status_code in (404, 410):
                # Session expired — request a new one and retry
                self.session_id = None
                continue
            resp.raise_for_status()
            body = resp.json()
            if body.get("error"):
                raise RuntimeError(body["error"])
            sid = body.get("session_id")
            if sid:
                self.session_id = sid
            user_text = body.get("text", "") or ""
            assistant_text = body.get("assistant_text", "") or ""
            lang = body.get("lang") or None
            audio_b64 = body.get("tts_audio_b64")
            audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""
            sr = int(body.get("tts_sample_rate", TARGET_SR) or TARGET_SR)
            skipped = bool(body.get("skipped"))
            return RemoteAgentResult(
                user_text=user_text,
                assistant_text=assistant_text,
                lang=lang,
                audio_pcm16=audio_bytes,
                sample_rate=sr,
                session_id=self.session_id,
                skipped=skipped,
                reason=body.get("reason"),
            )
        raise RuntimeError("Remote agent unavailable after retries")


if __name__ == "__main__":
    main()