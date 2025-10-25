# voice_agent_minimal.py
# Minimal bilingual voice agent - Remote-only version
# - Audio capture with VAD
# - Audio feedback (beep)
# - Remote agent communication
# - Enhanced latency diagnostics
# - Audio playback
# Version: 2025-10-25 (Minimal) - Modified to play audio feedback during remote agent wait

import os
import io
import sys
import time
import wave
import base64
import contextlib
import threading
import multiprocessing
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
BEEP_DELAY_MS = int(os.getenv("BEEP_DELAY_MS", "2000"))

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
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(int16.tobytes())
    buf.seek(0)
    return buf


def play_audio_feedback_process(beep_delay_ms: int, stop_flag_value):
    """Play feedback sounds in separate process - doesn't interfere with main thread."""
    try:
        # Create fresh output stream in this process
        stream = sd.OutputStream(samplerate=TARGET_SR, channels=1, dtype='float32')
        stream.start()
        
        # Play beep
        stream.write(BEEP_SOUND.reshape(-1, 1))
        
        # Wait for delay
        start = time.time()
        while (time.time() - start) < (beep_delay_ms / 1000.0):
            if stop_flag_value.value == 1:  # Check if we should stop
                stream.stop()
                stream.close()
                return
            time.sleep(0.05)

        stream.stop()
        stream.close()
    except Exception:
        pass  # Silently fail if audio doesn't work in subprocess


# =========================
# Audio Capture
# =========================

def _select_input_device():
    devices = sd.query_devices()
    
    if INPUT_DEVICE_INDEX is not None:
        if 0 <= INPUT_DEVICE_INDEX < len(devices):
            return INPUT_DEVICE_INDEX, devices[INPUT_DEVICE_INDEX]["default_samplerate"]
        print(f"⚠️  Device index {INPUT_DEVICE_INDEX} out of range. Using default.")
    
    if PREFERRED_INPUT_NAME:
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                if PREFERRED_INPUT_NAME.lower() in d["name"].lower():
                    return i, d["default_samplerate"]
        print(f"⚠️  No device matching '{PREFERRED_INPUT_NAME}'. Using default.")
    
    default_input = sd.query_devices(kind="input")
    return None, default_input["default_samplerate"]


class ContinuousListener:
    """Continuous audio capture with VAD."""
    
    def __init__(self):
        self.device_index, self.device_sr = _select_input_device()
        self.device_sr = int(self.device_sr)
        
        device_info = sd.query_devices(self.device_index, kind="input")
        device_name = device_info.get("name", "Unknown")
        print(f"\n🎤 Using input device: {device_name} @ {self.device_sr} Hz")

    def record_utterance(self) -> io.BytesIO:
        print("\n🎧 Listening...", end=" ", flush=True)
        
        frame_samples = int(self.device_sr * FRAME_DURATION_MS / 1000.0)
        max_frames = int((MAX_UTTERANCE_SECS * 1000.0) / FRAME_DURATION_MS)
        
        chunks = []
        last_above: Optional[float] = None
        start_time: Optional[float] = None
        
        try:
            with sd.InputStream(
                device=self.device_index,
                samplerate=self.device_sr,
                channels=1,
                dtype='float32',
                latency='low',
                blocksize=frame_samples,
            ) as stream:
                for _ in range(max_frames):
                    buf, _ = stream.read(frame_samples)
                    rms = float(np.sqrt(np.mean(buf**2)))
                    now = time.perf_counter()
                    
                    if rms > RMS_THRESH:
                        if start_time is None:
                            start_time = now
                        last_above = now
                        chunks.append(buf.tobytes())
                    else:
                        if last_above is not None:
                            chunks.append(buf.tobytes())
                            if (now - last_above) > max(SILENCE_TIMEOUT_SECS, RMS_HANGOVER):
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

                # Create stop flag for audio feedback process
                stop_flag = multiprocessing.Value('i', 0)
                result = None
                exception = None
                
                # Start audio feedback in separate PROCESS (not thread)
                feedback_process = multiprocessing.Process(
                    target=play_audio_feedback_process,
                    args=(BEEP_DELAY_MS, stop_flag),
                    daemon=True
                )
                feedback_process.start()
                
                # Send to remote agent (happens in parallel, no GIL interference)
                try:
                    with tracker.track("remote_agent"):
                        result = remote_client.process(wav_buf)
                except Exception as e:
                    exception = e
                finally:
                    # Signal audio feedback to stop
                    stop_flag.value = 1
                    feedback_process.join(timeout=0.5)
                    if feedback_process.is_alive():
                        feedback_process.terminate()
                
                # Handle any exception from remote agent
                if exception:
                    print(f"❌ Remote agent error: {exception}")
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