# voice_agent_minimal.py
# Minimal bilingual voice agent - Remote-only version
# - Audio capture with VAD
# - Audio feedback (beep + mhm)
# - Remote agent communication
# - Enhanced latency diagnostics
# - Audio playback
# Version: 2025-10-25 (Minimal)

import os
import io
import sys
import time
import wave
import base64
import contextlib
from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    import sounddevice as sd
except ImportError as exc:
    print("ERROR: sounddevice not installed. Run: pip install sounddevice", file=sys.stderr)
    raise SystemExit(1) from exc

from dotenv import load_dotenv
from scipy.signal import resample
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

# =========================
# Configuration
# =========================

REMOTE_AGENT_URL = os.getenv("REMOTE_AGENT_URL", "").strip() or None
REMOTE_AGENT_TOKEN = os.getenv("REMOTE_AGENT_TOKEN", "").strip() or None
REMOTE_AGENT_OPENAI_KEY = os.getenv("REMOTE_AGENT_OPENAI_KEY", "").strip() or None

if not REMOTE_AGENT_OPENAI_KEY:
    forward_flag = os.getenv("REMOTE_AGENT_FORWARD_OPENAI_KEY", "1").strip().lower() in {"1", "true", "yes", "on"}
    if forward_flag:
        REMOTE_AGENT_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip() or None

# Audio settings
TARGET_SR = 16000
CHANNELS = 1
FRAME_DURATION_MS = int(os.getenv("FRAME_DURATION_MS", "15"))
MAX_UTTERANCE_SECS = 45
SILENCE_TIMEOUT_SECS = float(os.getenv("SILENCE_TIMEOUT_SECS", "0.2"))
MIN_SPEECH_SECS = float(os.getenv("MIN_SPEECH_SECS", "0.3"))
RMS_THRESH = float(os.getenv("RMS_THRESH", "0.003"))
RMS_HANGOVER = float(os.getenv("RMS_HANGOVER", "0.12"))

# Audio feedback
BEEP_DELAY_MS = int(os.getenv("BEEP_DELAY_MS", "1000"))

# Device selection
PREFERRED_INPUT_NAME = os.getenv("PREFERRED_INPUT_NAME", "").strip() or None
INPUT_DEVICE_INDEX = os.getenv("INPUT_DEVICE_INDEX", "").strip()
INPUT_DEVICE_INDEX = int(INPUT_DEVICE_INDEX) if INPUT_DEVICE_INDEX.isdigit() else None

# Wake word
WAKE_WORD = os.getenv("WAKE_WORD", "").strip() or None

# HTTP timeouts
HTTP_CONNECT_TIMEOUT = float(os.getenv("HTTP_CONNECT_TIMEOUT", "4.0"))
HTTP_READ_TIMEOUT = float(os.getenv("HTTP_READ_TIMEOUT", "60.0"))
HTTP_TIMEOUT = (HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT)


# =========================
# HTTP Session
# =========================

def _configure_http_session(session: requests.Session) -> requests.Session:
    adapter = HTTPAdapter(
        pool_connections=8,
        pool_maxsize=16,
        max_retries=Retry(total=2, backoff_factor=0.1, status_forcelist=[429, 502, 503, 504]),
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


_http_session_lock = __import__('threading').Lock()
_shared_http_session: Optional[requests.Session] = None


def _get_shared_http_session() -> requests.Session:
    global _shared_http_session
    with _http_session_lock:
        if _shared_http_session is None:
            _shared_http_session = _configure_http_session(requests.Session())
    return _shared_http_session


# =========================
# Audio Feedback
# =========================

def generate_rising_beep(start_freq=200, end_freq=300, duration_ms=600):
    """Frequency rises from low to high (swoosh up)."""
    duration_sec = duration_ms / 1000.0
    samples = int(TARGET_SR * duration_sec)
    t = np.linspace(0, duration_sec, samples, False)
    
    # Linear frequency sweep
    freq = np.linspace(start_freq, end_freq, samples)
    phase = 2 * np.pi * np.cumsum(freq) / TARGET_SR
    tone = np.sin(phase).astype(np.float32)
    
    # Envelope
    envelope = np.ones_like(tone)
    fade_samples = int(TARGET_SR * 0.01)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    
    return tone * envelope * 0.3

# Pre-generate beep sound
BEEP_SOUND = generate_rising_beep()


# =========================
# Audio Output
# =========================

class OutputAudio:
    """Persistent output stream."""
    
    def __init__(self, samplerate=TARGET_SR, channels=1):
        stream_kwargs = dict(
            samplerate=samplerate,
            channels=channels,
            dtype='float32',
        )
        try:
            self.stream = sd.OutputStream(latency='low', **stream_kwargs)
        except Exception:
            self.stream = sd.OutputStream(**stream_kwargs)
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


# =========================
# Latency Tracker
# =========================

class LatencyTracker:
    """Enhanced latency tracking."""
    
    def __init__(self):
        self.events: list[tuple[str, float]] = []
        self.turn_start = time.perf_counter()

    @contextlib.contextmanager
    def track(self, label: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self.events.append((label, max(0.0, end - start)))

    def clear(self):
        self.events.clear()
        self.turn_start = time.perf_counter()

    def report(self, title: str = "⏱️ Latency breakdown"):
        if not self.events:
            return
        
        total_turn = time.perf_counter() - self.turn_start
        sum_tracked = sum(d for _, d in self.events)
        untracked = max(0.0, total_turn - sum_tracked)
        
        print()
        print("=" * 60)
        print(title)
        print("=" * 60)
        
        for label, duration in self.events:
            pct_of_total = (duration / total_turn * 100.0) if total_turn > 0 else 0.0
            bar_width = int(pct_of_total / 2)
            bar = "█" * bar_width
            print(f"  {label:<25} {duration*1000:7.1f}ms  {pct_of_total:5.1f}%  {bar}")
        
        if untracked > 0.01:
            pct_untracked = (untracked / total_turn * 100.0)
            bar_width = int(pct_untracked / 2)
            bar = "░" * bar_width
            print(f"  {'[untracked overhead]':<25} {untracked*1000:7.1f}ms  {pct_untracked:5.1f}%  {bar}")
        
        print("-" * 60)
        print(f"  {'TOTAL TURN TIME':<25} {total_turn*1000:7.1f}ms  100.0%")
        print("=" * 60)
        print()


# =========================
# Audio Utilities
# =========================

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


def pick_input_device(prefer_name_substr=None, prefer_index=None):
    devices = sd.query_devices()
    if isinstance(prefer_index, int) and 0 <= prefer_index < len(devices):
        if devices[prefer_index]['max_input_channels'] > 0:
            return prefer_index
    if prefer_name_substr:
        p = prefer_name_substr.lower()
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0 and p in d.get('name', '').lower():
                return i
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            return i
    raise RuntimeError("No input audio devices with capture channels.")


def play_audio_feedback(out: OutputAudio, tracker: LatencyTracker):
    """Play beep sound followed by mhm with configured delays."""
    with tracker.track("audio_feedback"):
        time.sleep(BEEP_DELAY_MS / 1000.0)
        out.write_float_np(BEEP_SOUND)


# =========================
# Audio Capture
# =========================

class ContinuousListener:
    """Audio capture with RMS VAD."""
    
    def __init__(self):
        self.device_index = pick_input_device(PREFERRED_INPUT_NAME, INPUT_DEVICE_INDEX)
        self.dev_info = sd.query_devices(self.device_index)
        self.device_sr = TARGET_SR

    def record_utterance(self):
        """Capture one utterance based on RMS VAD."""
        print("\n🎙️ Listening… (speak; short pause = end)")
        chunks = []
        start_time = time.time()
        last_above = None
        frame_samples = max(256, int(TARGET_SR * FRAME_DURATION_MS / 1000))
        started_speaking = False

        def callback(indata, frames, time_info, status):
            nonlocal last_above, started_speaking
            if status:
                print(status, file=sys.stderr)
            mono = indata.copy().reshape(-1)
            rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
            chunks.append(mono.tobytes())
            if rms > RMS_THRESH:
                last_above = time.time()
                started_speaking = True

        stream_kwargs = dict(
            device=self.device_index,
            channels=CHANNELS,
            samplerate=self.device_sr,
            dtype='float32',
            blocksize=frame_samples,
            callback=callback,
        )

        try:
            try:
                stream = sd.InputStream(latency='low', **stream_kwargs)
            except Exception:
                stream = sd.InputStream(**stream_kwargs)

            with stream:
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
# Remote Agent Client
# =========================

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
    def __init__(self, base_url: str, token: Optional[str] = None, openai_api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/api"):
            self.base_url = f"{self.base_url}/api"
        self.token = token.strip() if token else None
        self.openai_api_key = openai_api_key.strip() if openai_api_key else None
        self.session_id: Optional[str] = None
        self.session = _configure_http_session(requests.Session())

    def _headers(self):
        headers = {}
        if self.token:
            headers["X-Auth"] = self.token
        if self.openai_api_key:
            headers["X-OpenAI-Key"] = self.openai_api_key
        return headers

    def ensure_session(self):
        if self.session_id:
            return
        resp = self.session.post(
            f"{self.base_url}/session",
            headers=self._headers(),
            timeout=HTTP_TIMEOUT,
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
            if self.openai_api_key:
                data["openai_api_key"] = self.openai_api_key
            
            resp = self.session.post(
                f"{self.base_url}/process",
                headers=self._headers(),
                files=files,
                data=data,
                timeout=HTTP_TIMEOUT,
            )
            
            if resp.status_code in (401, 403):
                raise RuntimeError("Remote agent authentication failed")
            
            if resp.status_code in (404, 410):
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


# =========================
# Main Loop
# =========================

def main():
    # Validate configuration
    if not REMOTE_AGENT_URL:
        print("\n" + "=" * 60)
        print("❌ ERROR: REMOTE_AGENT_URL is not set!")
        print("=" * 60)
        print("\nThis minimal version requires a remote agent server.")
        print("Please set REMOTE_AGENT_URL in your .env file.")
        print("\nExample:")
        print("  REMOTE_AGENT_URL=https://your-runpod-url.proxy.runpod.net/")
        print("=" * 60)
        sys.exit(1)

    print(f"\n✅ Using remote voice agent backend at: {REMOTE_AGENT_URL}")
    
    remote_client = RemoteAgentClient(
        REMOTE_AGENT_URL,
        REMOTE_AGENT_TOKEN,
        openai_api_key=REMOTE_AGENT_OPENAI_KEY,
    )
    
    if REMOTE_AGENT_OPENAI_KEY:
        print("✅ Forwarding OpenAI API key to remote agent server.")

    out = OutputAudio(samplerate=TARGET_SR, channels=1)
    listener = ContinuousListener()

    print("\n🎙️ Bilingual voice agent ready. (HR/EN)")
    print("Tips:")
    print("- Speak naturally; a short pause ends your turn.")
    if WAKE_WORD:
        print(f"- Wake word enabled: say \"{WAKE_WORD}\" to start a turn.")

    try:
        while True:
            try:
                tracker = LatencyTracker()
                
                # Capture audio
                with tracker.track("capture"):
                    wav_buf = listener.record_utterance()
                
                if wav_buf.getbuffer().nbytes < 32000:
                    tracker.report("⏱️ Turn skipped (no usable audio)")
                    continue

                # Play audio feedback while waiting for remote response
                play_audio_feedback(out, tracker)
                
                # Send to remote agent
                try:
                    with tracker.track("remote_agent"):
                        result = remote_client.process(wav_buf)
                except Exception as e:
                    print(f"❌ Remote agent error: {e}")
                    tracker.report()
                    continue
                
                if not result or not result.user_text:
                    tracker.report("⏱️ Turn skipped (remote empty)")
                    continue
                
                if result.skipped:
                    if (result.reason == "wake_word_missing") and WAKE_WORD:
                        print(f"(Ignored — missing wake word '{WAKE_WORD}')")
                    tracker.report("⏱️ Turn skipped (wake word)")
                    continue
                
                # Display results
                flag = "🇭🇷" if (result.lang or "").startswith("hr") else "🇬🇧"
                print(f"{flag} You: {result.user_text}")
                print(f"🤖 Assistant: {result.assistant_text}")
                
                # Play response audio
                if result.audio_pcm16:
                    with tracker.track("playback"):
                        if result.sample_rate != TARGET_SR:
                            pcm = np.frombuffer(result.audio_pcm16, dtype=np.int16).astype(np.float32) / 32768.0
                            pcm_16k = resample_to_16k(pcm, result.sample_rate)
                            out.write_float_np(pcm_16k)
                        else:
                            out.write_int16_bytes(result.audio_pcm16)
                else:
                    print("⚠️ (Remote agent returned no audio)")
                
                tracker.report()

            except KeyboardInterrupt:
                print("\n👋 Exit. Bye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                tracker.report("⏱️ Turn errored")
                time.sleep(0.2)
    finally:
        out.close()


if __name__ == "__main__":
    main()